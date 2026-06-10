"""Shared twin-OSSE driver for the coupled fast--slow Lorenz examples (08, 09).

Both examples plug a :class:`~pypfda.models.base.ForwardModel` adapter from
:mod:`pypfda.models.lorenz` into the *same* :class:`~pypfda.driver.CycleDriver`
that orchestrates the GCM cores, and run a three-experiment OSSE:

* **TRUTH** -- one trajectory; its FAST variable (plus noise) is the
  pseudo-observation, its SLOW variable is the reconstruction target.
* **FREE**  -- an ensemble run forward with no assimilation (the baseline).
* **DA**    -- the same ensemble assimilating the fast pseudo-observations with
  observation-error tempering, a max-weight cap, and post-resample inflation.

Only the FAST variable is observed and only the SLOW variable is scored --- the
minimal, laptop-runnable analogue of reconstructing slow ocean overturning
(AMOC) from fast surface temperature (SST). Reconstruction skill is the Pearson
correlation between the *forecast* ensemble-mean slow variable and the truth
(the same forecast convention the engine enforces via :mod:`pypfda.verify`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pypfda import ParticleFilter
from pypfda.driver import CycleDriver, SerialBackend
from pypfda.verify import scan_osse_result


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _drive(model: Any, obs_provider: Callable[[int], Any], *,
           n_cycles: int, window: float, resample: bool,
           inflation: float, seed: int) -> tuple[np.ndarray, dict]:
    """Run one ensemble: FREE (no resampling/inflation) or DA (full filter)."""
    pf = ParticleFilter(
        ess_threshold=0.5 if resample else 1e-9,
        resampling="systematic",
        max_weight=0.3 if resample else None,
        rng=np.random.default_rng(seed),
    )
    driver = CycleDriver(
        model=model,
        pf=pf,
        observations=obs_provider,
        n_cycles=n_cycles,
        window=window,
        inflation_amplitude=inflation if resample else 0.0,
        backend=SerialBackend(),
        base_seed=seed,
    )
    hist = driver.run()
    return np.asarray(hist["targets"], dtype=float), hist


def run_fast_slow_osse(make_model: Callable[[int], Any], *, n_members: int,
                       n_cycles: int, window: float, spinup_steps: int,
                       obs_sigma: float, eta: float, inflation: float,
                       seed: int = 20260609) -> dict:
    """Generic observe-fast / reconstruct-slow twin OSSE.

    ``make_model(n)`` builds a fresh adapter for ``n`` members. Returns a results
    dict with per-cycle truth/FREE/DA slow series, skill, and the
    :func:`pypfda.verify.scan_osse_result` gate verdict.
    """
    rng = np.random.default_rng(seed)

    # TRUTH: one trajectory supplies the slow target and the fast pseudo-obs.
    truth = make_model(1)
    truth.initialize_member(0, truth.spin_up(rng, spinup_steps))
    truth_slow, truth_obs = [], []
    for _ in range(n_cycles):
        truth.forecast(0, window)
        truth_obs.append(truth.observe(0, window).copy())
        truth_slow.append(truth.target_diagnostic(0))
    truth_slow = np.asarray(truth_slow)
    truth_obs = np.asarray(truth_obs)

    # Pseudo-observations and observation-error tempering (likelihood uses eta*sigma).
    obs_series = truth_obs + rng.normal(0.0, obs_sigma, truth_obs.shape)
    eff_err = float(eta * obs_sigma)

    def obs_provider(cycle: int) -> tuple[np.ndarray, float]:
        return obs_series[cycle], eff_err

    # Diverse initial ensemble, SHARED by FREE and DA (only DA assimilates).
    spin = make_model(1)
    ics = [spin.spin_up(np.random.default_rng(seed + 1000 + m), spinup_steps)
           for m in range(n_members)]
    free_model, da_model = make_model(n_members), make_model(n_members)
    for m in range(n_members):
        free_model.initialize_member(m, ics[m])
        da_model.initialize_member(m, ics[m])

    free_t, _ = _drive(free_model, obs_provider, n_cycles=n_cycles, window=window,
                       resample=False, inflation=inflation, seed=seed + 7)
    da_t, da_h = _drive(da_model, obs_provider, n_cycles=n_cycles, window=window,
                        resample=True, inflation=inflation, seed=seed + 7)

    free_mean, da_mean = np.nanmean(free_t, axis=1), np.nanmean(da_t, axis=1)
    eas = np.asarray(da_h["eas"], float)

    # Adversarial result gate: guard against clone / stale-diagnostic artifacts.
    gate = scan_osse_result(da_mean, truth_slow, eas=eas, free_ens=free_mean, label="DA")
    return {
        "gate": gate,
        "cycle": np.arange(1, n_cycles + 1),
        "truth_slow": truth_slow,
        "free_targets": free_t, "da_targets": da_t,
        "free_mean": free_mean, "da_mean": da_mean,
        "ess_da": np.asarray(da_h["ess"], float), "eas_da": eas,
        "r_free": _corr(free_mean, truth_slow), "r_da": _corr(da_mean, truth_slow),
    }


def rmse_skill(r: dict) -> tuple[float, float, float]:
    """RMSE of FREE and DA against the truth, and the fractional RMSE reduction.

    For a highly persistent slow variable (low effective DOF) correlation is a
    noisy skill metric; RMSE is the robust one (see example 09).
    """
    ef = float(np.sqrt(np.nanmean((r["free_mean"] - r["truth_slow"]) ** 2)))
    ed = float(np.sqrt(np.nanmean((r["da_mean"] - r["truth_slow"]) ** 2)))
    return ef, ed, 1.0 - ed / ef


def report(name: str, r: dict) -> None:
    """Print a one-line skill + gate summary for a finished OSSE."""
    g, eas = r["gate"], r["eas_da"]
    ef, ed, sk = rmse_skill(r)
    print(
        f"{name:18s}  r_FREE={r['r_free']:+.3f}  r_DA={r['r_da']:+.3f}  "
        f"Dr={r['r_da'] - r['r_free']:+.3f}  | RMSE {ef:.2f}->{ed:.2f} ({sk:+.0%})  "
        f"| gate ok={g['ok']} clones={max(g['clones'].values())} nan={g['nan_frac']:.0%}  "
        f"| EAS min={eas.min():.1f}"
    )
    for c in g["caveats"]:
        print("   caveat:", c)
    for p in g["pathologies"]:
        print("   PATHOLOGY:", p)
