r"""Example 10 -- a clear, working idealised benchmark for the online particle filter.

This is the *positive* companion to examples 08 (two-scale Lorenz-96) and 09
(coupled Lorenz-63): a regime in which the same ``pypfda`` particle filter
**clearly works** on a fast--slow twin OSSE -- the DA ensemble-mean reconstructs a
genuinely time-varying *unobserved* slow variable with a large, unambiguous
TRACKING gain over the FREE baseline, *while the genealogy stays healthy* (high
effective sample size, many surviving ancestors, varying cycle-to-cycle). It is
the minimal, laptop-runnable analogue of reconstructing slow ocean-overturning
(AMOC) variability from fast surface observations (SST), and is meant to be a
reusable benchmark.

Setup
-----
Coupled fast--slow Lorenz-63 (Pena & Kalnay, 2004): a fast "atmosphere"
:math:`(x_f,y_f,z_f)` two-way coupled (strength :math:`\kappa`) to a slow
"ocean" :math:`(x_s,y_s,z_s)` evolving at rate :math:`\varepsilon`. We observe
ONLY the fast subsystem (with noise) and reconstruct ONLY the slow variable
:math:`x_s`. TRUTH is one trajectory; FREE is a diverse-IC ensemble run forward
with no assimilation; DA is the *same* ensemble assimilating the fast
pseudo-observations through the identical engine used for the GCM cores.

The "clearly-working" regime (all knobs below)
---------------------------------------------
* a CHAOTIC slow subsystem (:math:`\varepsilon=0.9`): the slow variable
  :math:`x_s` genuinely oscillates and switches attractor lobes throughout the
  run (standard deviation :math:`\approx 8`, range roughly :math:`[-17,18]`), so
  the reconstruction target is a real time-varying signal -- NOT a near-frozen
  fixed point (the artifact that made an earlier tuning of this example look
  good for the wrong reason; correlations over a flat truth are meaningless);
* strong two-way coupling (:math:`\kappa=3`) so the fast observations genuinely
  inform the slow variable AND so the likelihood does not become so peaked that
  the genealogy collapses;
* a large ensemble (:math:`N=300`), moderate observation noise
  (:math:`\sigma_\mathrm{obs}=1`), generous observation-error tempering
  (:math:`\eta=8`, which flattens the likelihood and keeps many ancestors
  alive), a short assimilation window (0.2 model-time units) and modest
  inflation (0.8).

The mechanism (and why it is not over-fitting)
---------------------------------------------
With diverse initial conditions the FREE ensemble mean drifts onto the *wrong*
attractor lobe and ends up anti-correlated with the truth's slow variable
(:math:`r_\mathrm{FREE}<0`). The DA filter, fed only noisy *fast* observations,
identifies the lobe the slow subsystem currently occupies and pulls the ensemble
onto it: the reconstructed slow variable then tracks the truth's slow envelope
(:math:`r_\mathrm{FREE}\approx-0.50 \to r_\mathrm{DA}\approx+0.69`,
:math:`\Delta r\approx+1.2` on the representative seed; median
:math:`r_\mathrm{DA}\approx+0.67`, :math:`r\ge0.5` on :math:`\sim`90 % of seeds).
The gain is a TRACKING gain on a varying signal, not a constant-level offset --
the ANOMALY RMSE (after removing each series' time-mean) also drops,
:math:`\sim 8.2\to 6.1`. Crucially, the effective sample size and the effective
ancestor size stay HIGH and VARYING throughout (median ESS :math:`\approx 0.8\,N`,
median EAS :math:`\approx 0.46\,N`, minimum EAS :math:`\approx 45`, never
collapsing to a single lineage): the gain is a genuine constraint from the
observations, not one particle memorising the truth. (On seeds where FREE already
happens to be neutral, DA correctly does not hurt -- the same "DA adds value only
when the prior is wrong" logic as the main study.)

Contrast with the harder two-scale Lorenz-96 case (example 08), where the same
engine still beats FREE but drives the genealogy to near-collapse (effective
ancestor size :math:`\to 1`). The two examples bracket the diversity--memory /
cycle-length trade-off the package is built to manage, *without a single change
to the filter*.

Outputs
-------
Two CLEAN figures (no titles / panel letters / statistics baked into the image
-- those belong in a caption) and a small metrics file:

* ``docs/_static/l63_clearly_works_timeseries.png`` -- TRUTH vs FREE-mean vs
  DA-mean of the unobserved slow variable, with ensemble bands;
* ``docs/_static/l63_clearly_works_filterhealth.png`` -- per-cycle effective
  sample size and effective ancestor size (the filter stays healthy);
* ``docs/_static/l63_clearly_works_metrics.json`` -- the headline numbers.

Run::

    python examples/10_coupled_l63_clearly_works.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _coupled_lorenz_common import report, rmse_skill, run_fast_slow_osse

from pypfda.models.lorenz import CoupledLorenz63

# ---- the clearly-working regime (every knob fixed here) -------------------
EPS = 0.9  # CHAOTIC slow subsystem -> x_s genuinely varies (std ~ 8), not a fixed point
KAPPA = 3.0  # strong fast<->slow coupling: observable AND genealogy does not collapse
DT = 0.01  # RK4 step
N_MEMBERS = 300  # large ensemble -> healthy ESS/EAS after a peaked likelihood
N_CYCLES = 900
WINDOW = 0.20  # short assimilation window
SPINUP = 3000
OBS_SIGMA = 1.0  # moderate observation noise on the fast subsystem
ETA = 8.0  # observation-error tempering: flattens the likelihood, keeps ancestors alive
INFLATION = 0.8  # post-resample inflation
SEED = 314  # representative "FREE in the wrong lobe, DA rescues it" seed (see header)

STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"
FIG_TS = STATIC / "l63_clearly_works_timeseries.png"
FIG_HEALTH = STATIC / "l63_clearly_works_filterhealth.png"
OUT_JSON = STATIC / "l63_clearly_works_metrics.json"

TRUTH_C, FREE_C, DA_C = "#1a1a1a", "#3B6EA5", "#E8743B"


def _anom_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE after removing each series' time-mean: variability skill, not level."""
    a = np.asarray(a, float) - np.nanmean(a)
    b = np.asarray(b, float) - np.nanmean(b)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def main() -> None:
    r = run_fast_slow_osse(
        lambda n: CoupledLorenz63(n, eps=EPS, kappa=KAPPA, dt=DT),
        n_members=N_MEMBERS,
        n_cycles=N_CYCLES,
        window=WINDOW,
        spinup_steps=SPINUP,
        obs_sigma=OBS_SIGMA,
        eta=ETA,
        inflation=INFLATION,
        seed=SEED,
    )
    report("coupled L63 (works)", r)
    ef, ed, skill = rmse_skill(r)
    ts = r["truth_slow"]
    af, ad = _anom_rmse(r["free_mean"], ts), _anom_rmse(r["da_mean"], ts)
    ess, eas = np.asarray(r["ess_da"], float), np.asarray(r["eas_da"], float)
    dr = r["r_da"] - r["r_free"]
    std_truth = float(np.std(ts))

    # ---- metrics file (the headline numbers; nothing is drawn on the PNGs) --
    metrics = {
        "model": "CoupledLorenz63",
        "regime": "clearly-working",
        "params": {
            "eps": EPS,
            "kappa": KAPPA,
            "dt": DT,
            "n_members": N_MEMBERS,
            "n_cycles": N_CYCLES,
            "window": WINDOW,
            "spinup_steps": SPINUP,
            "obs_sigma": OBS_SIGMA,
            "eta": ETA,
            "inflation": INFLATION,
            "seed": SEED,
            "resampling": "systematic",
            "ess_threshold": 0.5,
            "max_weight": 0.3,
        },
        "std_truth_slow": std_truth,
        "truth_slow_min": float(np.min(ts)),
        "truth_slow_max": float(np.max(ts)),
        "r_free": r["r_free"],
        "r_da": r["r_da"],
        "delta_r": dr,
        "rmse_free": ef,
        "rmse_da": ed,
        "rmse_reduction_frac": skill,
        "anom_rmse_free": af,
        "anom_rmse_da": ad,
        "anom_rmse_reduction_frac": 1.0 - ad / af,
        "ess_median": float(np.median(ess)),
        "ess_min": float(ess.min()),
        "eas_median": float(np.median(eas)),
        "eas_min": float(eas.min()),
        "eas_std": float(np.std(eas)),
        "frac_cycles_eas_le_1p5": float(np.mean(eas <= 1.5)),
        "gate_ok": bool(r["gate"]["ok"]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print("metrics written to", OUT_JSON)

    x = r["cycle"]

    # ---- Panel 1: reconstruction of the UNOBSERVED slow variable -----------
    # CLEAN: axis labels + a line-identity legend only. Titles, panel letters,
    # model names and r / Delta-r / RMSE numbers all go in the LaTeX caption.
    fig, ax = plt.subplots(figsize=(6.8, 2.7), constrained_layout=True)
    flo, fhi = np.nanpercentile(r["free_targets"], 5, 1), np.nanpercentile(r["free_targets"], 95, 1)
    dlo, dhi = np.nanpercentile(r["da_targets"], 5, 1), np.nanpercentile(r["da_targets"], 95, 1)
    ax.fill_between(x, flo, fhi, color="#BCBCBC", alpha=0.45, lw=0, zorder=1)
    ax.fill_between(x, dlo, dhi, color=DA_C, alpha=0.28, lw=0, zorder=3)
    ax.plot(x, r["da_mean"], color=DA_C, lw=1.7, zorder=5, label="DA")
    ax.plot(x, r["truth_slow"], color=TRUTH_C, lw=1.1, zorder=6, label="TRUTH")
    ax.plot(x, r["free_mean"], color=FREE_C, ls=(0, (5, 2)), lw=1.3, zorder=8, label="FREE")
    ax.set_xlabel("assimilation cycle")
    ax.set_ylabel(r"slow variable  $x_s$")
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

    # ---- Panel 2: filter health (NOT degenerate in this regime) ------------
    fig, ax = plt.subplots(figsize=(6.8, 2.7), constrained_layout=True)
    ax.plot(x, ess, color="#2E8B57", lw=1.3, label="effective sample size")
    ax.plot(x, eas, color="#7D5BA6", lw=1.3, label="effective ancestor size")
    ax.axhline(0.5 * N_MEMBERS, color="#999999", lw=1.0, ls=":")  # resample threshold
    ax.set_xlabel("assimilation cycle")
    ax.set_ylabel("ensemble members")
    ax.set_ylim(0, N_MEMBERS * 1.03)
    ax.margins(x=0.005)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, facecolor="white", edgecolor="none")
    fig.savefig(FIG_HEALTH, dpi=300)
    plt.close(fig)
    print("figure written to", FIG_HEALTH)

    # ---- assert the "works" claim is real (and honest) ---------------------
    if std_truth < 3.0:
        raise SystemExit(
            f"truth slow variable barely varies (std={std_truth:.2f}); "
            f"correlations would be meaningless -- pick a chaotic regime."
        )
    if r["r_da"] < 0.6:
        raise SystemExit(f"DA did not track the slow variable (r_DA={r['r_da']:+.2f}).")
    if dr <= 0.5:
        raise SystemExit(f"DA did not clearly beat FREE on the slow variable (Delta r={dr:+.2f}).")
    if ad >= af:
        raise SystemExit(
            f"DA did not cut the ANOMALY RMSE (FREE {af:.2f} -> DA {ad:.2f}); "
            f"the gain would be a level offset, not variability tracking."
        )
    if np.median(eas) < 0.3 * N_MEMBERS:
        raise SystemExit(
            f"genealogy degenerate (median EAS={np.median(eas):.1f} < 0.3N); not the healthy regime."
        )
    if np.std(eas) <= 0.0:
        raise SystemExit("EAS is pinned (std=0): the filter never exercised the genealogy.")
    print(
        f"OK: DA clearly reconstructs the time-varying slow variable "
        f"(std={std_truth:.1f}) -- r {r['r_free']:+.2f}->{r['r_da']:+.2f} (Delta r {dr:+.2f}), "
        f"anomaly RMSE {af:.2f}->{ad:.2f}, median ESS {np.median(ess):.0f}/{N_MEMBERS}, "
        f"median EAS {np.median(eas):.0f} (std {np.std(eas):.0f}; healthy and exercised)."
    )


if __name__ == "__main__":
    main()
