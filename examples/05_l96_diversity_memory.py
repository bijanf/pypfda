"""Diversity--memory trade-off on Lorenz-96 — the paper's headline claim.

The companion paper argues that, in a finite-ensemble SIR filter,
resampling concentrates the particles on high-likelihood trajectories
and thereby destroys ensemble diversity. Post-resampling inflation can
restore the diversity, but it simultaneously erases the information
that had been accumulated in the ensemble state — the *memory* of past
assimilated observations. The central finding (paper §5) is that no
amount of inflation avoids the trade-off: zero inflation collapses the
filter, too-much inflation washes out the very signal the filter was
correcting for.

On a coupled ocean--atmosphere system the effect is striking because
the ocean memory is multi-decadal. On Lorenz-96, the memory timescale
is only the Lyapunov time (~0.42 model time units), so the trade-off
shows up fast: the skill curve as a function of inflation amplitude is
U-shaped, and the minimum is sharp.

This script sweeps the post-resampling inflation standard deviation
from zero to large and records

* the ensemble-mean RMSE of the analysis (skill),
* the ensemble standard deviation at the end of each cycle
  (diversity),
* the fraction of cycles in which resampling occurred.

Output: ``docs/_static/l96_diversity_memory.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _l96_common import (  # type: ignore[import-not-found]
    FilterConfig,
    N_VARS,
    integrate,
    spin_up,
)

from pypfda import ParticleFilter

DT = 0.01
OBS_INTERVAL = 0.05  # the short-cycle regime where DA wins clearly
N_STEPS_PER_OBS = int(OBS_INTERVAL / DT)
N_CYCLES = 100
BURN_IN = 25
SEED = 20260423

INFLATIONS = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])

OUT_FIG = Path(__file__).resolve().parents[1] / "docs" / "_static" / "l96_diversity_memory.png"


def _run_one(inflation: float, cfg: FilterConfig, rng: np.random.Generator) -> dict:
    truth = spin_up()
    ensemble = truth + rng.normal(0, cfg.initial_spread, size=(cfg.n_members, N_VARS))

    obs_idx = np.arange(0, N_VARS, cfg.obs_stride)

    pf = ParticleFilter(
        ess_threshold=cfg.ess_threshold,
        resampling="systematic",
        max_weight=cfg.max_weight,
        rng=rng,
    )

    rmse: list[float] = []
    spread: list[float] = []
    n_resample = 0

    for _ in range(N_CYCLES):
        truth = integrate(truth, N_STEPS_PER_OBS, DT)
        ensemble = integrate(ensemble, N_STEPS_PER_OBS, DT)

        obs = truth[obs_idx] + rng.normal(0, cfg.obs_sigma, obs_idx.size)
        ensemble_obs = ensemble[:, obs_idx]
        effective_err = cfg.obs_tempering_eta * cfg.obs_sigma

        ensemble, info = pf.assimilate(ensemble, ensemble_obs, obs, obs_err=effective_err)

        if info.resampled:
            n_resample += 1
            if inflation > 0:
                ensemble = ensemble + rng.normal(0, inflation, ensemble.shape)

        mean = (
            ensemble.mean(axis=0)
            if info.resampled
            else np.average(ensemble, axis=0, weights=info.weights)
        )
        rmse.append(float(np.sqrt(np.mean((mean - truth) ** 2))))
        spread.append(float(ensemble.std(axis=0).mean()))

    return {
        "inflation": inflation,
        "mean_rmse": float(np.mean(rmse[BURN_IN:])),
        "mean_spread": float(np.mean(spread[BURN_IN:])),
        "resample_fraction": n_resample / N_CYCLES,
    }


def main() -> None:
    cfg = FilterConfig()
    results = []
    for sigma in INFLATIONS:
        rng = np.random.default_rng(SEED)
        r = _run_one(float(sigma), cfg, rng)
        print(
            f"inflation={sigma:5.2f}  RMSE={r['mean_rmse']:5.2f}  "
            f"spread={r['mean_spread']:5.2f}  resample_frac={r['resample_fraction']:.2f}"
        )
        results.append(r)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_rmse, ax_spread) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    sigmas = np.array([r["inflation"] for r in results])
    rmses = np.array([r["mean_rmse"] for r in results])
    spreads = np.array([r["mean_spread"] for r in results])

    # Left: skill (RMSE) vs inflation.
    ax_rmse.plot(sigmas, rmses, "o-", color="#1f77b4", lw=1.8, ms=7)
    ax_rmse.set_xlabel("post-resample inflation $\\sigma$")
    ax_rmse.set_ylabel("mean analysis RMSE")
    ax_rmse.set_title("Skill vs inflation")
    ax_rmse.grid(alpha=0.3)

    best = int(np.argmin(rmses))
    ax_rmse.annotate(
        f"optimum\n$\\sigma={sigmas[best]:g}$, RMSE={rmses[best]:.2f}",
        xy=(sigmas[best], rmses[best]),
        xytext=(sigmas[best] + 0.3, rmses[best] + 0.8),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        fontsize=9,
    )

    # Right: skill vs diversity, parameterised by inflation.
    ax_spread.plot(spreads, rmses, "o-", color="#2ca02c", lw=1.8, ms=7)
    for s, sig, rm in zip(spreads, sigmas, rmses, strict=True):
        ax_spread.annotate(f"$\\sigma={sig:g}$", (s, rm), fontsize=7, alpha=0.7)
    ax_spread.set_xlabel("ensemble spread (end-of-cycle $\\sigma$)")
    ax_spread.set_ylabel("mean analysis RMSE")
    ax_spread.set_title("Trade-off: skill vs diversity")
    ax_spread.grid(alpha=0.3)

    fig.suptitle(
        "Diversity--memory trade-off on Lorenz-96  "
        "(paper's headline claim, on the standard benchmark)",
        fontsize=11,
        y=1.04,
    )
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"figure written to {OUT_FIG}")


if __name__ == "__main__":
    main()
