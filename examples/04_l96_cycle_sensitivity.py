"""Cycle-length sensitivity on Lorenz-96 — analogue of T10 / T11 / T12.

The companion paper observes on AMOC that assimilation skill depends
non-trivially on cycle length: T10 (1-year cycles) gives marginal
skill, T11 (5-year cycles) is best, T12 (10-year cycles) falls below
the free baseline because it samples below the Nyquist frequency of
the dominant ~13.3-year AMOC oscillation.

We ask the analogous question on Lorenz-96 with *exactly the same
filter recipe* (η = 4 observation-error tempering, post-resample
inflation, max-weight cap 0.3, N = 400 members).

The Lorenz-96 attractor at F = 8 carries both a Lyapunov timescale
(~0.42 model time units, the deterministic-predictability horizon)
and a spectral peak at T ~ 1.65 (see ``07_l96_nyquist.py``), which
sets a Nyquist sampling bound at T/2 ~ 0.83. Two bounds apply:

* for observation intervals well below the Lyapunov time, DA
  comfortably beats the free run;
* at roughly the Lyapunov time the filter is *worse* than free,
  because forecast uncertainty has grown faster than the tempered
  likelihood can usefully constrain;
* for intervals near the Nyquist bound and beyond, the filter
  degenerates toward the free baseline — the dominant mode is
  aliased and the observations no longer carry coherent information.

On Lorenz-96 the Lyapunov bound is the tighter of the two and bites
first: the curve is not a clean Nyquist U-curve but a
Lyapunov-bounded degradation. The paper's T10 / T11 / T12 U-curve is
the regime where the ordering is reversed — the coupled system's
Lyapunov time is decades, so Nyquist (driven by the 13.3-yr AMOC
oscillation) is the tighter bound and the optimum at T11 (5-year
cycles) appears. Same mechanism, different ordering of bounds.

Output: ``docs/_static/l96_cycle_sensitivity.png``.
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
N_CYCLES_PER_CELL = 40  # fixed number of DA cycles per observation interval
BURN_IN_FRACTION = 0.25  # skip the first 25 % of cycles for summary stats
SEED = 20260423

# Observation intervals probed. Chosen to span roughly one Lyapunov time
# (~0.42 for Lorenz-96 with F = 8) from well-below to well-above the
# Lyapunov bound.
OBS_INTERVALS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]

OUT_FIG = Path(__file__).resolve().parents[1] / "docs" / "_static" / "l96_cycle_sensitivity.png"


def _run_one(obs_interval: float, cfg: FilterConfig, rng: np.random.Generator) -> dict:
    n_steps_per_obs = max(1, int(round(obs_interval / DT)))
    n_cycles = N_CYCLES_PER_CELL

    truth = spin_up()
    ensemble_da = truth + rng.normal(0, cfg.initial_spread, size=(cfg.n_members, N_VARS))
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

    for _ in range(n_cycles):
        truth = integrate(truth, n_steps_per_obs, DT)
        ensemble_da = integrate(ensemble_da, n_steps_per_obs, DT)
        ensemble_free = integrate(ensemble_free, n_steps_per_obs, DT)

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
        rmse_da.append(float(np.sqrt(np.mean((mean_da - truth) ** 2))))
        rmse_free.append(float(np.sqrt(np.mean((ensemble_free.mean(0) - truth) ** 2))))

    burn_in = int(BURN_IN_FRACTION * n_cycles)
    return {
        "obs_interval": obs_interval,
        "n_cycles": n_cycles,
        "mean_rmse_da": float(np.mean(rmse_da[burn_in:])),
        "mean_rmse_free": float(np.mean(rmse_free[burn_in:])),
        "rmse_da_series": rmse_da,
        "rmse_free_series": rmse_free,
    }


def main() -> None:
    cfg = FilterConfig()
    results = []
    for dt_obs in OBS_INTERVALS:
        rng = np.random.default_rng(SEED)  # same IC + same obs noise across rows
        r = _run_one(dt_obs, cfg, rng)
        print(
            f"obs_interval={dt_obs:5.2f}  cycles={r['n_cycles']:4d}  "
            f"DA={r['mean_rmse_da']:5.2f}  FREE={r['mean_rmse_free']:5.2f}  "
            f"ratio={r['mean_rmse_da'] / r['mean_rmse_free']:.2f}"
        )
        results.append(r)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), constrained_layout=True)

    dts = np.array([r["obs_interval"] for r in results])
    da_means = np.array([r["mean_rmse_da"] for r in results])
    free_means = np.array([r["mean_rmse_free"] for r in results])

    ax.plot(dts, da_means, "o-", color="#1f77b4", lw=1.8, ms=7, label="Particle filter")
    ax.plot(dts, free_means, "s--", color="#999999", lw=1.2, ms=6, label="Free ensemble")
    ax.axhline(cfg.obs_sigma, color="#d62728", lw=1.0, ls=":", label="obs. noise")

    ax.set_xscale("log")
    ax.set_xlabel("observation interval $\\Delta t_{\\mathrm{obs}}$ (model time units)")
    ax.set_ylabel("mean analysis RMSE")
    ax.set_title(
        "Lorenz-96 cycle-length sensitivity  (analogue of T10 / T11 / T12)\n"
        f"N={cfg.n_members}, $\\eta$={cfg.obs_tempering_eta:g}, "
        f"max_weight={cfg.max_weight:g}"
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # Overlay the two intrinsic timescales of the system.
    lyapunov_time = 0.42
    nyquist_bound = 0.83
    ax.axvline(lyapunov_time, color="#d62728", lw=0.9, ls="-.", alpha=0.8)
    ax.axvline(nyquist_bound, color="#555555", lw=0.9, ls=":", alpha=0.8)
    ymax = ax.get_ylim()[1]
    ax.text(
        lyapunov_time * 1.03,
        ymax * 0.94,
        r"Lyapunov" + "\n" + r"$\sim$0.42",
        fontsize=8,
        color="#d62728",
    )
    ax.text(
        nyquist_bound * 1.03,
        ymax * 0.94,
        r"Nyquist" + "\n" + r"$\sim$0.83",
        fontsize=8,
        color="#555555",
    )

    best = int(np.argmin(da_means))
    ax.annotate(
        f"best DA: $\\Delta t_{{\\mathrm{{obs}}}} = {dts[best]:g}$",
        xy=(dts[best], da_means[best]),
        xytext=(dts[best] * 2.2, da_means[best] * 1.8),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        fontsize=9,
    )

    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"figure written to {OUT_FIG}")


if __name__ == "__main__":
    main()
