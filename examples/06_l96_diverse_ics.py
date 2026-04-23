"""Diverse initial conditions on Lorenz-96 — analogue of T13.

The paper's T13 ensemble starts each member from a different 50-year
slice of a 5000-year control run, spanning AMOC strengths from 17 to
26 Sv. The question it answers is whether the particle filter's skill
depends on the members sharing initial-state memory, or whether the
filter can pull a widely-dispersed ensemble toward the truth using
observations alone. T13 shows the latter: starting from independent
climates, DA achieves strong positive correlation with truth that the
free ensemble cannot reach.

This script does the corresponding Lorenz-96 experiment: each of N
members is spun up from an independent random seed onto the attractor,
so the initial ensemble is maximally diverse across the attractor. We
then compare analysis-mean RMSE against a free ensemble with the same
diverse ICs.

If DA can pull the ensemble together from distinct attractor states,
the skill advantage over the free run should be *larger* with diverse
ICs than with the near-identical ICs used in earlier demos, because
the free ensemble mean tends toward the climatology whereas the DA
ensemble locks onto the truth trajectory.

Output: ``docs/_static/l96_diverse_ics.png``.
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
OBS_INTERVAL = 0.05
N_STEPS_PER_OBS = int(OBS_INTERVAL / DT)
N_CYCLES = 200
N_MEMBERS = 200
POST_RESAMPLE_INFLATION = 0.75  # the optimum from experiment 05
SEED = 20260423

OUT_FIG = Path(__file__).resolve().parents[1] / "docs" / "_static" / "l96_diverse_ics.png"


def _build_diverse_ensemble(n_members: int, rng: np.random.Generator) -> np.ndarray:
    """Spin up each member independently so they land on different
    attractor states."""
    ensemble = np.empty((n_members, N_VARS))
    for m in range(n_members):
        state = np.full(N_VARS, 8.0)
        state += rng.normal(0, 3.0, N_VARS)  # large kick so members start elsewhere
        # Long spin-up so each member is an independent draw from the
        # Lorenz-96 attractor measure.
        ensemble[m] = integrate(state, n_steps=3000, dt=DT)
    return ensemble


def main() -> None:
    cfg = FilterConfig(n_members=N_MEMBERS, post_resample_inflation=POST_RESAMPLE_INFLATION)

    rng = np.random.default_rng(SEED)
    truth = spin_up()
    ensemble_da = _build_diverse_ensemble(cfg.n_members, rng)
    ensemble_free = ensemble_da.copy()

    obs_idx = np.arange(0, N_VARS, cfg.obs_stride)
    pf = ParticleFilter(
        ess_threshold=cfg.ess_threshold,
        resampling="systematic",
        max_weight=cfg.max_weight,
        rng=rng,
    )

    rmse_da: list[float] = []
    rmse_free: list[float] = []
    spread_da: list[float] = []
    spread_free: list[float] = []

    # Diagnostic scalar: the first variable of the truth, and the
    # ensemble means, plotted over time so the "pulling-together"
    # behaviour is visible.
    truth_x0: list[float] = []
    da_mean_x0: list[float] = []
    free_mean_x0: list[float] = []

    for _ in range(N_CYCLES):
        truth = integrate(truth, N_STEPS_PER_OBS, DT)
        ensemble_da = integrate(ensemble_da, N_STEPS_PER_OBS, DT)
        ensemble_free = integrate(ensemble_free, N_STEPS_PER_OBS, DT)

        obs = truth[obs_idx] + rng.normal(0, cfg.obs_sigma, obs_idx.size)
        ensemble_obs = ensemble_da[:, obs_idx]
        effective_err = cfg.obs_tempering_eta * cfg.obs_sigma

        ensemble_da, info = pf.assimilate(ensemble_da, ensemble_obs, obs, obs_err=effective_err)
        if info.resampled:
            ensemble_da = ensemble_da + rng.normal(
                0, cfg.post_resample_inflation, ensemble_da.shape
            )

        mean_da = (
            ensemble_da.mean(axis=0)
            if info.resampled
            else np.average(ensemble_da, axis=0, weights=info.weights)
        )
        mean_free = ensemble_free.mean(axis=0)
        rmse_da.append(float(np.sqrt(np.mean((mean_da - truth) ** 2))))
        rmse_free.append(float(np.sqrt(np.mean((mean_free - truth) ** 2))))
        spread_da.append(float(ensemble_da.std(axis=0).mean()))
        spread_free.append(float(ensemble_free.std(axis=0).mean()))

        truth_x0.append(float(truth[0]))
        da_mean_x0.append(float(mean_da[0]))
        free_mean_x0.append(float(mean_free[0]))

    print(f"Mean RMSE  DA   (cycles 20+) = {np.mean(rmse_da[20:]):.3f}")
    print(f"Mean RMSE  FREE (cycles 20+) = {np.mean(rmse_free[20:]):.3f}")
    print(f"Mean spread DA   (end) = {np.mean(spread_da[-20:]):.3f}")
    print(f"Mean spread FREE (end) = {np.mean(spread_free[-20:]):.3f}")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_ts, ax_rmse) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    t_obs = np.arange(N_CYCLES) * OBS_INTERVAL

    # Left: first variable trajectory. Black = truth, blue = DA mean,
    # grey = free mean. Shows DA locking onto truth while free drifts.
    ax_ts.plot(t_obs, truth_x0, color="black", lw=1.4, label="truth", zorder=3)
    ax_ts.plot(t_obs, da_mean_x0, color="#1f77b4", lw=1.2, label="DA mean", zorder=2)
    ax_ts.plot(t_obs, free_mean_x0, color="#999999", lw=1.0, label="free mean", zorder=1)
    ax_ts.set_xlabel("model time")
    ax_ts.set_ylabel("$x_0$")
    ax_ts.set_title("First-variable trajectory from diverse ICs")
    ax_ts.legend(frameon=False, loc="upper right")
    ax_ts.grid(alpha=0.3)

    # Right: RMSE time series.
    ax_rmse.plot(t_obs, rmse_free, color="#999999", lw=1.2, label="Free ensemble")
    ax_rmse.plot(t_obs, rmse_da, color="#1f77b4", lw=1.6, label="Particle filter")
    ax_rmse.axhline(cfg.obs_sigma, color="#d62728", lw=0.9, ls="--", label="obs. noise")
    ax_rmse.set_xlabel("model time")
    ax_rmse.set_ylabel("ensemble-mean RMSE")
    ax_rmse.set_title("Skill from diverse ICs")
    ax_rmse.legend(frameon=False, loc="upper right")
    ax_rmse.grid(alpha=0.3)
    ax_rmse.set_ylim(bottom=0)

    fig.suptitle(
        "Diverse-IC ensemble on Lorenz-96  (analogue of T13)",
        fontsize=11,
        y=1.05,
    )
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"figure written to {OUT_FIG}")


if __name__ == "__main__":
    main()
