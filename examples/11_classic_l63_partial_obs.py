r"""Example 11 -- the textbook positive control: a single chaotic Lorenz-63,
observe ONE variable, reconstruct the UNOBSERVED ones.

This is the clean, visually decisive "the particle filter OBVIOUSLY works" case.
It is *separate* from example 10 (the coupled fast--slow L63, our on-narrative
partial-skill case): here a single canonical Lorenz-63 attractor is the whole
world. We observe only :math:`x` (with noise) and reconstruct the **unobserved**
:math:`z` (and, for completeness, :math:`y`). With only 3 state dimensions and a
few hundred particles there is no curse of dimensionality, so the SIR particle
filter -- the *identical* :class:`~pypfda.ParticleFilter` /
:class:`~pypfda.driver.CycleDriver` engine used for the GCM cores, no filter
changes -- reconstructs the hidden dimension almost perfectly while keeping a
healthy, exercised genealogy.

Setup
-----
* **TRUTH**: one Lorenz-63 trajectory (sigma=10, rho=28, beta=8/3; the standard
  chaotic regime), RK4 with dt=0.01, one assimilation cycle every ``window``
  model-time units.
* **FREE**: an :math:`N`-member ensemble from diverse initial conditions, NO
  assimilation -> its mean collapses to the attractor's centroid and does not
  track the truth's chaotic oscillation.
* **DA**: the SAME ensemble assimilating ONLY noisy :math:`x` observations
  through the identical SIR / systematic-resampling engine. It identifies which
  wing of the attractor the truth currently occupies and reconstructs the
  unobserved :math:`z`.

The headline target is the UNOBSERVED variable -- inferring the hidden dimension
from a single observed one is the impressive part.

Acceptance gate (asserted at the end)
-------------------------------------
1. DA clearly tracks the unobserved variable: ``r_DA >= +0.85``.
2. FREE clearly fails: ``r_FREE <= +0.30``.
3. Real tracking, not a level effect: anomaly RMSE (time-mean removed) on the
   unobserved variable cut by >= 50 % (DA vs FREE).
4. Healthy, exercised genealogy: median ESS >~ 0.3 N, median EAS healthy and
   varying (std(EAS) > 0), ~0 % of cycles with EAS <= 1.5 (no collapse).
5. Multi-seed robustness: passes on >= 5 seeds (min/median r_DA reported).
6. Honest: the gain is an observational constraint, not one particle memorising
   the truth -- the healthy ESS/EAS confirm this.

Outputs (clean PNGs -- NO titles / panel letters / model names / statistics
baked into the image; all of that goes in the LaTeX caption):

* ``docs/_static/l63_classic_timeseries.png``   -- TRUTH vs DA-mean vs FREE-mean
  of the unobserved :math:`z`, with ensemble bands;
* ``docs/_static/l63_classic_attractor.png``    -- the :math:`x`--:math:`z`
  attractor with the DA ensemble sitting ON it vs the FREE ensemble scattered;
* ``docs/_static/l63_classic_filterhealth.png`` -- per-cycle ESS and EAS;
* ``docs/_static/l63_classic_metrics.json``     -- the headline numbers.

Run::

    python examples/11_classic_l63_partial_obs.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pypfda import ParticleFilter
from pypfda.driver import CycleDriver, SerialBackend
from pypfda.models.base import ForwardModel
from pypfda.verify import scan_osse_result

FloatArray = NDArray[np.floating[Any]]

# --- the chosen "textbook works" regime (every knob fixed here) ------------
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0  # standard chaotic Lorenz-63
DT = 0.01  # RK4 step
OBS_IDX = 0  # observe ONLY x
TARGET_IDX = 2  # reconstruct the UNOBSERVED z
N_MEMBERS = 1000
N_CYCLES = 500
WINDOW = 0.08  # assimilation every 8 model steps -- frequent enough to constrain,
SPINUP = 3000  #   chaotic between obs; tuned with the scan in figs/_l63_classic_work
OBS_SIGMA = 0.8  # Gaussian noise on x (a few % of the x range ~[-20, 20])
ETA = 3.0  # observation-error tempering (likelihood uses eta*sigma): flattens the
#            likelihood just enough that the filter does NOT resample every cycle,
#            so the 3-D genealogy stays exercised (EAS varies, never collapses)
INFLATION = 2.5  # post-resample inflation: re-diversifies clones so EAS recovers
SEED = 42  # representative seed (multi-seed robustness asserted below)
SEEDS = (42, 101, 202, 303, 404, 505, 606)  # for the multi-seed gate (>= 5)
SPIN_EVAL = 50  # cycles dropped (filter spin-up) before scoring

STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"
FIG_TS = STATIC / "l63_classic_timeseries.png"
FIG_ATTR = STATIC / "l63_classic_attractor.png"
FIG_HEALTH = STATIC / "l63_classic_filterhealth.png"
OUT_JSON = STATIC / "l63_classic_metrics.json"

TRUTH_C, FREE_C, DA_C, BAND_C = "#1a1a1a", "#3B6EA5", "#E8743B", "#BCBCBC"
VAR = ("x", "y", "z")[TARGET_IDX]


# ===========================================================================
# Self-contained standalone Lorenz-63 adapter (NOT added to the package).
# Same ForwardModel contract as pypfda.models.lorenz.CoupledLorenz63, but a
# single, *uncoupled* L63: observe x, reconstruct the unobserved z.
# ===========================================================================
def l63_rhs(s: FloatArray, *, sigma: float, rho: float, beta: float) -> FloatArray:
    """Right-hand side of the classic Lorenz-63 system."""
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], float)


class Lorenz63(ForwardModel):
    r"""Standard chaotic Lorenz-63 as an in-process :class:`ForwardModel`.

    Observe one variable (``obs_idx``); reconstruct another (``target_idx``).
    Ensemble held as an array ``(n_members, 3)``. The target diagnostic is the
    FORECAST value (set in :meth:`forecast`, NOT swapped by :meth:`set_state`),
    so the driver records forecast skill -- the conservative OSSE convention.
    """

    DIM = 3

    def __init__(
        self,
        n_members: int,
        *,
        sigma: float = SIGMA,
        rho: float = RHO,
        beta: float = BETA,
        dt: float = DT,
        obs_idx: int = OBS_IDX,
        target_idx: int = TARGET_IDX,
        inflate_sigma: float = 1.0,
    ) -> None:
        self.n_members = int(n_members)
        self.sigma, self.rho, self.beta = float(sigma), float(rho), float(beta)
        self.dt = float(dt)
        self.obs_idx = np.array([int(obs_idx)])
        self.target_idx = int(target_idx)
        self.inflate_sigma = float(inflate_sigma)
        self._state = np.zeros((self.n_members, self.DIM))
        self._wmean = np.zeros((self.n_members, self.DIM))
        self._diag = np.zeros(self.n_members)
        self._diag_state = np.zeros((self.n_members, self.DIM))  # full forecast state (for plots)

    def _rhs(self, s: FloatArray) -> FloatArray:
        return l63_rhs(s, sigma=self.sigma, rho=self.rho, beta=self.beta)

    def _rk4(self, s: FloatArray) -> FloatArray:
        dt = self.dt
        k1 = self._rhs(s)
        k2 = self._rhs(s + 0.5 * dt * k1)
        k3 = self._rhs(s + 0.5 * dt * k2)
        k4 = self._rhs(s + dt * k3)
        return cast(FloatArray, s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))

    def integrate(self, s: FloatArray, n_steps: int) -> FloatArray:
        """Integrate one state vector ``n_steps`` RK4 steps (public, for spin-up)."""
        for _ in range(n_steps):
            s = self._rk4(s)
        return s

    def spin_up(self, rng: np.random.Generator, n_steps: int = SPINUP) -> FloatArray:
        """Return one on-attractor state, seeded from ``rng`` (for diverse ICs)."""
        s = rng.normal(0.0, 1.0, self.DIM)
        s[2] = abs(s[2]) + 10.0  # start above the z floor of the attractor
        return self.integrate(s, n_steps)

    # -- ForwardModel contract -------------------------------------------
    def initialize_member(self, member_id: int, ic_spec: Any) -> None:
        """Set member's full state from a 3-vector IC."""
        self._state[member_id] = np.asarray(ic_spec, float)
        self._diag[member_id] = self._state[member_id, self.target_idx]
        self._diag_state[member_id] = self._state[member_id]

    def forecast(self, member_id: int, window: float) -> None:
        """Advance member by ``window`` and accumulate a window-mean."""
        n = max(1, round(window / self.dt))
        s = self._state[member_id].copy()
        acc = np.zeros_like(s)
        for _ in range(n):
            s = self._rk4(s)
            acc += s
        self._state[member_id] = s
        self._wmean[member_id] = acc / n
        self._diag[member_id] = s[self.target_idx]  # forecast diagnostic (forecast convention)
        self._diag_state[member_id] = s

    def observe(self, member_id: int, window: float) -> FloatArray:
        """Return the window-mean of the single observed variable."""
        return cast(FloatArray, self._wmean[member_id, self.obs_idx])

    def get_state(self, member_id: int) -> FloatArray:
        """Return an independent copy of the member's full state."""
        return cast(FloatArray, self._state[member_id].copy())

    def set_state(self, member_id: int, state: FloatArray) -> None:
        """Overwrite the member's full state with a parent's snapshot."""
        self._state[member_id] = np.asarray(state, float)

    def inflate(self, member_id: int, amplitude: float, seed: int) -> None:
        """Add Gaussian noise to re-diversify the member after resampling."""
        if amplitude == 0:
            return
        rng = np.random.default_rng(seed)
        self._state[member_id] += rng.normal(0.0, amplitude * self.inflate_sigma, self.DIM)

    def target_diagnostic(self, member_id: int) -> float:
        """Unobserved target variable (forecast value); never assimilated."""
        return float(self._diag[member_id])


# ===========================================================================
# Twin-OSSE driver (observe one variable, reconstruct another) on the engine.
# ===========================================================================
def _corr(a: FloatArray, b: FloatArray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _anom_rmse(a: FloatArray, b: FloatArray) -> float:
    """RMSE after removing each series' time-mean: variability skill, not level."""
    a = np.asarray(a, float) - np.nanmean(a)
    b = np.asarray(b, float) - np.nanmean(b)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def _drive(
    model: Lorenz63,
    obs: Any,
    *,
    n_cycles: int,
    window: float,
    resample: bool,
    inflation: float,
    seed: int,
) -> tuple[FloatArray, dict, np.ndarray]:
    pf = ParticleFilter(
        ess_threshold=0.5 if resample else 1e-9,
        resampling="systematic",
        max_weight=0.3 if resample else None,
        rng=np.random.default_rng(seed),
    )
    driver = CycleDriver(
        model=model,
        pf=pf,
        observations=obs,
        n_cycles=n_cycles,
        window=window,
        inflation_amplitude=inflation if resample else 0.0,
        backend=SerialBackend(),
        base_seed=seed,
    )
    hist = driver.run()
    targets = np.asarray(hist["targets"], float)  # (n_cycles, n_members) unobserved var
    # Final-cycle full-state snapshot of every member, for the attractor plot.
    final_state = model._diag_state.copy()
    return targets, hist, final_state


def run_classic_l63_osse(
    *,
    n_members: int,
    n_cycles: int,
    window: float,
    spinup: int,
    obs_sigma: float,
    eta: float,
    inflation: float,
    seed: int,
    obs_idx: int = OBS_IDX,
    target_idx: int = TARGET_IDX,
) -> dict:
    """Observe one variable, reconstruct another, on the identical pypfda engine."""
    rng = np.random.default_rng(seed)

    def mk(n: int) -> Lorenz63:
        return Lorenz63(n, obs_idx=obs_idx, target_idx=target_idx)

    # TRUTH: one trajectory supplies the unobserved target and the noisy obs.
    truth = mk(1)
    truth.initialize_member(0, truth.spin_up(rng, spinup))
    truth_target, truth_obs, truth_xz = [], [], []
    for _ in range(n_cycles):
        truth.forecast(0, window)
        truth_obs.append(truth.observe(0, window).copy())
        truth_target.append(truth.target_diagnostic(0))
        truth_xz.append(truth._diag_state[0, [0, 2]].copy())  # (x, z) for the attractor plot
    truth_target = np.asarray(truth_target)
    truth_obs = np.asarray(truth_obs)
    truth_xz = np.asarray(truth_xz)

    obs_series = truth_obs + rng.normal(0.0, obs_sigma, truth_obs.shape)
    eff_err = float(eta * obs_sigma)

    def obs_provider(cycle: int) -> tuple[FloatArray, float]:
        return obs_series[cycle], eff_err

    # Diverse-IC ensemble SHARED by FREE and DA (only DA assimilates).
    spin = mk(1)
    ics = [spin.spin_up(np.random.default_rng(seed + 1000 + m), spinup) for m in range(n_members)]
    free_model, da_model = mk(n_members), mk(n_members)
    for m in range(n_members):
        free_model.initialize_member(m, ics[m])
        da_model.initialize_member(m, ics[m])

    free_t, _, free_final = _drive(
        free_model,
        obs_provider,
        n_cycles=n_cycles,
        window=window,
        resample=False,
        inflation=inflation,
        seed=seed + 7,
    )
    da_t, da_h, da_final = _drive(
        da_model,
        obs_provider,
        n_cycles=n_cycles,
        window=window,
        resample=True,
        inflation=inflation,
        seed=seed + 7,
    )

    free_mean, da_mean = np.nanmean(free_t, 1), np.nanmean(da_t, 1)
    eas = np.asarray(da_h["eas"], float)
    gate = scan_osse_result(da_mean, truth_target, eas=eas, free_ens=free_mean, label="DA")
    return {
        "gate": gate,
        "cycle": np.arange(1, n_cycles + 1),
        "truth_target": truth_target,
        "truth_xz": truth_xz,
        "free_targets": free_t,
        "da_targets": da_t,
        "free_mean": free_mean,
        "da_mean": da_mean,
        "free_final_xz": free_final[:, [0, 2]],
        "da_final_xz": da_final[:, [0, 2]],
        "ess_da": np.asarray(da_h["ess"], float),
        "eas_da": eas,
        "r_free": _corr(free_mean, truth_target),
        "r_da": _corr(da_mean, truth_target),
    }


def _scored(r: dict, spin: int = SPIN_EVAL) -> dict:
    """Post-spin-up skill metrics on the unobserved variable."""
    tt = r["truth_target"][spin:]
    fm, dm = r["free_mean"][spin:], r["da_mean"][spin:]
    return {
        "r_free": _corr(fm, tt),
        "r_da": _corr(dm, tt),
        "rmse_free": float(np.sqrt(np.nanmean((fm - tt) ** 2))),
        "rmse_da": float(np.sqrt(np.nanmean((dm - tt) ** 2))),
        "armse_free": _anom_rmse(fm, tt),
        "armse_da": _anom_rmse(dm, tt),
    }


def main() -> None:
    r = run_classic_l63_osse(
        n_members=N_MEMBERS,
        n_cycles=N_CYCLES,
        window=WINDOW,
        spinup=SPINUP,
        obs_sigma=OBS_SIGMA,
        eta=ETA,
        inflation=INFLATION,
        seed=SEED,
    )
    s = _scored(r)
    ess = np.asarray(r["ess_da"], float)[SPIN_EVAL:]
    eas = np.asarray(r["eas_da"], float)[SPIN_EVAL:]
    dr = s["r_da"] - s["r_free"]
    armse_red = 1.0 - s["armse_da"] / s["armse_free"]
    std_truth = float(np.std(r["truth_target"][SPIN_EVAL:]))

    # ---- multi-seed robustness ----------------------------------------
    seed_r_da = []
    for sd in SEEDS:
        rr = run_classic_l63_osse(
            n_members=N_MEMBERS,
            n_cycles=N_CYCLES,
            window=WINDOW,
            spinup=SPINUP,
            obs_sigma=OBS_SIGMA,
            eta=ETA,
            inflation=INFLATION,
            seed=sd,
        )
        seed_r_da.append(_scored(rr)["r_da"])
    seed_r_da = np.asarray(seed_r_da)

    # ---- metrics file (nothing is drawn on the PNGs) ------------------
    metrics = {
        "model": "Lorenz63 (standalone, single chaotic attractor)",
        "regime": "classic-partial-obs",
        "observed_variable": ("x", "y", "z")[OBS_IDX],
        "reconstructed_variable": VAR,
        "params": {
            "sigma": SIGMA,
            "rho": RHO,
            "beta": BETA,
            "dt": DT,
            "window": WINDOW,
            "steps_per_cycle": round(WINDOW / DT),
            "n_members": N_MEMBERS,
            "n_cycles": N_CYCLES,
            "spinup_steps": SPINUP,
            "obs_sigma": OBS_SIGMA,
            "eta": ETA,
            "inflation": INFLATION,
            "seed": SEED,
            "spin_eval": SPIN_EVAL,
            "resampling": "systematic",
            "ess_threshold": 0.5,
            "max_weight": 0.3,
        },
        "std_truth_target": std_truth,
        "truth_target_min": float(np.min(r["truth_target"])),
        "truth_target_max": float(np.max(r["truth_target"])),
        "r_free": s["r_free"],
        "r_da": s["r_da"],
        "delta_r": dr,
        "rmse_free": s["rmse_free"],
        "rmse_da": s["rmse_da"],
        "rmse_reduction_frac": 1.0 - s["rmse_da"] / s["rmse_free"],
        "anom_rmse_free": s["armse_free"],
        "anom_rmse_da": s["armse_da"],
        "anom_rmse_reduction_frac": armse_red,
        "ess_median": float(np.median(ess)),
        "ess_min": float(ess.min()),
        "eas_median": float(np.median(eas)),
        "eas_min": float(eas.min()),
        "eas_std": float(np.std(eas)),
        "frac_cycles_eas_le_1p5": float(np.mean(eas <= 1.5)),
        "seeds": list(SEEDS),
        "seed_r_da": [float(v) for v in seed_r_da],
        "seed_r_da_min": float(seed_r_da.min()),
        "seed_r_da_median": float(np.median(seed_r_da)),
        "gate_ok": bool(r["gate"]["ok"]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print("metrics written to", OUT_JSON)

    # ---- Figure 1: reconstruction of the UNOBSERVED variable ----------
    x = r["cycle"]
    win = slice(SPIN_EVAL, SPIN_EVAL + 90)  # legible window: ~90 cycles, oscillation visible
    fig, ax = plt.subplots(figsize=(6.8, 2.7), constrained_layout=True)
    flo = np.nanpercentile(r["free_targets"], 5, 1)
    fhi = np.nanpercentile(r["free_targets"], 95, 1)
    dlo = np.nanpercentile(r["da_targets"], 5, 1)
    dhi = np.nanpercentile(r["da_targets"], 95, 1)
    ax.fill_between(x[win], flo[win], fhi[win], color=BAND_C, alpha=0.45, lw=0, zorder=1)
    ax.fill_between(x[win], dlo[win], dhi[win], color=DA_C, alpha=0.25, lw=0, zorder=3)
    ax.plot(x[win], r["da_mean"][win], color=DA_C, lw=1.8, zorder=5, label="DA")
    ax.plot(x[win], r["truth_target"][win], color=TRUTH_C, lw=1.2, zorder=6, label="TRUTH")
    ax.plot(
        x[win], r["free_mean"][win], color=FREE_C, ls=(0, (5, 2)), lw=1.4, zorder=8, label="FREE"
    )
    ax.set_xlabel("assimilation cycle")
    ax.set_ylabel(f"unobserved variable  ${VAR}$")
    ax.margins(x=0.005)
    ax.legend(
        loc="upper center",
        ncol=3,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(FIG_TS, dpi=300)
    plt.close(fig)
    print("figure written to", FIG_TS)

    # ---- Figure 2: phase-space attractor (x-z) ------------------------
    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    txz = r["truth_xz"]
    ax.plot(txz[:, 0], txz[:, 1], color=TRUTH_C, lw=0.6, alpha=0.55, zorder=2, label="TRUTH")
    ax.scatter(
        r["free_final_xz"][:, 0],
        r["free_final_xz"][:, 1],
        s=10,
        color=FREE_C,
        alpha=0.55,
        lw=0,
        zorder=3,
        label="FREE",
    )
    ax.scatter(
        r["da_final_xz"][:, 0],
        r["da_final_xz"][:, 1],
        s=10,
        color=DA_C,
        alpha=0.75,
        lw=0,
        zorder=4,
        label="DA",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.legend(
        loc="upper center",
        ncol=3,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(FIG_ATTR, dpi=300)
    plt.close(fig)
    print("figure written to", FIG_ATTR)

    # ---- Figure 3: filter health (twin axis: ESS ~0.6 N, EAS modest) ---
    fig, ax = plt.subplots(figsize=(6.8, 2.7), constrained_layout=True)
    ax.plot(x, r["ess_da"], color="#2E8B57", lw=1.3, label="effective sample size")
    ax.axhline(0.5 * N_MEMBERS, color="#999999", lw=1.0, ls=":")
    ax.set_xlabel("assimilation cycle")
    ax.set_ylabel("effective sample size")
    ax.set_ylim(0, N_MEMBERS * 1.03)
    ax.margins(x=0.005)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, r["eas_da"], color="#7D5BA6", lw=1.3, label="effective ancestor size")
    ax2.set_ylabel("effective ancestor size")
    # Scale to the post-spin-up regime so the cycle-to-cycle EAS variation is
    # legible (cycle 1 starts at N distinct ancestors -- excluded from the range).
    eas_arr = np.asarray(r["eas_da"], float)
    eas_hi = (
        float(np.nanmax(eas_arr[SPIN_EVAL:]))
        if eas_arr.size > SPIN_EVAL
        else float(np.nanmax(eas_arr))
    )
    ax2.set_ylim(0, max(5.0, eas_hi * 1.25))
    ax2.grid(False)
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(
        lines,
        [ln.get_label() for ln in lines],
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(FIG_HEALTH, dpi=300)
    plt.close(fig)
    print("figure written to", FIG_HEALTH)

    # ---- assert the acceptance gate (1-6) ------------------------------
    if s["r_da"] < 0.85:
        raise SystemExit(f"GATE 1 FAIL: r_DA={s['r_da']:+.3f} < 0.85 on unobserved {VAR}.")
    if s["r_free"] > 0.30:
        raise SystemExit(f"GATE 2 FAIL: r_FREE={s['r_free']:+.3f} > 0.30 (FREE should fail).")
    if armse_red < 0.50:
        raise SystemExit(
            f"GATE 3 FAIL: anomaly-RMSE reduction {armse_red:+.0%} < 50 % "
            f"(FREE {s['armse_free']:.2f} -> DA {s['armse_da']:.2f}); gain would be a level effect."
        )
    # Gate 4: a per-cycle weight spread that keeps many members alive
    # (ESS median >= 0.3 N), and a genealogy that is exercised (EAS varies) but
    # never collapses (no cycle with EAS <= 1.5). For a 3-D system the absolute
    # surviving-ancestor count is modest, so "healthy" here is "no collapse +
    # varying", in contrast to the high-dimensional Lorenz-96 case where EAS -> 1.
    if np.median(ess) < 0.3 * N_MEMBERS:
        raise SystemExit(
            f"GATE 4 FAIL: per-cycle weight spread too low (median ESS={np.median(ess):.0f} "
            f"< 0.3 N = {0.3 * N_MEMBERS:.0f})."
        )
    if np.std(eas) <= 0.0 or np.mean(eas <= 1.5) > 0.01 or eas.min() <= 1.5:
        raise SystemExit(
            f"GATE 4 FAIL: genealogy not healthy (EAS std={np.std(eas):.1f}, "
            f"min={eas.min():.0f}, %EAS<=1.5={100 * np.mean(eas <= 1.5):.1f}); "
            f"it must vary and never collapse."
        )
    if (seed_r_da >= 0.85).sum() < 5:
        raise SystemExit(
            f"GATE 5 FAIL: only {(seed_r_da >= 0.85).sum()}/{len(SEEDS)} seeds reach r_DA>=0.85 "
            f"(min {seed_r_da.min():+.2f}, median {np.median(seed_r_da):+.2f})."
        )
    print(
        f"OK (gate 1-6): observe x, reconstruct UNOBSERVED {VAR} -- "
        f"r {s['r_free']:+.2f}->{s['r_da']:+.2f} (Delta r {dr:+.2f}); "
        f"anomaly RMSE {s['armse_free']:.2f}->{s['armse_da']:.2f} ({armse_red:+.0%}); "
        f"median ESS {np.median(ess):.0f}/{N_MEMBERS}, median EAS {np.median(eas):.0f} "
        f"(std {np.std(eas):.0f}, %EAS<=1.5 {100 * np.mean(eas <= 1.5):.1f}); "
        f"multi-seed r_DA min {seed_r_da.min():+.2f} / median {np.median(seed_r_da):+.2f} "
        f"(>={(seed_r_da >= 0.85).sum()}/{len(SEEDS)} seeds pass)."
    )


if __name__ == "__main__":
    main()
