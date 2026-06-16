"""Lorenz-96 twin experiment with observation tempering + inflation.

Lorenz-96 is the standard atmospheric-DA benchmark (Lorenz, 1996): a
40-variable chaotic ODE whose dimension is high enough to expose the
classical curse-of-dimensionality failure mode of a vanilla SIR
particle filter. The three techniques defended by the paper that
motivated this package *are* enough to make the filter beat the free
run on Lorenz-96:

1. **Observation-error tempering.** The likelihood is evaluated with an
   inflated error ``eta * sigma_obs`` so that no single particle
   accumulates all the weight. ``eta = 4`` in the reference
   configuration below matches the value used in the paper.
2. **Post-resample Gaussian inflation.** After resampling, Gaussian
   noise is added to every member to break the genealogical collapse
   that otherwise freezes the ensemble into duplicates.
3. **Max-weight cap** (``max_weight=0.3``). A hard safeguard against
   one particle dominating even after tempering; the same heuristic the
   paper calls DA_T9b.

The script produces a two-panel figure ``docs/_static/lorenz96_demo.png``:
left, DA-mean vs FREE-mean RMSE over time; right, effective sample size
over time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypfda import ParticleFilter

# Model.
N_VARS = 40
FORCING = 8.0

# Integration.
DT = 0.01
OBS_INTERVAL = 0.05
N_STEPS_PER_OBS = int(OBS_INTERVAL / DT)
N_CYCLES = 200
SPINUP_STEPS = 5000

# Ensemble.
N_MEMBERS = 400
INITIAL_SPREAD = 1.0

# Observations: every other variable, Gaussian noise.
OBS_STRIDE = 2
OBS_SIGMA = 0.5

# Filter knobs used by the paper.
OBS_TEMPERING_ETA = 4.0  # inflate obs_err in the likelihood only
POST_RESAMPLE_INFLATION = 0.35
MAX_WEIGHT = 0.3
ESS_THRESHOLD = 0.5

SEED = 20260423
OUT_FIG = Path(__file__).resolve().parents[1] / "docs" / "_static" / "lorenz96_demo.png"


def lorenz96_rhs(state: np.ndarray) -> np.ndarray:
    """Right-hand side of Lorenz-96. Vectorised over the leading (ensemble) axis."""
    return (
        (np.roll(state, -1, axis=-1) - np.roll(state, 2, axis=-1)) * np.roll(state, 1, axis=-1)
        - state
        + FORCING
    )


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = lorenz96_rhs(state)
    k2 = lorenz96_rhs(state + 0.5 * dt * k1)
    k3 = lorenz96_rhs(state + 0.5 * dt * k2)
    k4 = lorenz96_rhs(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state: np.ndarray, n_steps: int, dt: float = DT) -> np.ndarray:
    for _ in range(n_steps):
        state = rk4_step(state, dt)
    return state


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Spin up the truth onto the attractor.
    truth = np.full(N_VARS, FORCING)
    truth[0] += 0.01
    truth = integrate(truth, n_steps=SPINUP_STEPS)

    # Initial ensemble.
    ensemble_da = truth + rng.normal(0, INITIAL_SPREAD, size=(N_MEMBERS, N_VARS))
    ensemble_free = ensemble_da.copy()

    obs_idx = np.arange(0, N_VARS, OBS_STRIDE)
    n_obs = obs_idx.size

    pf = ParticleFilter(
        ess_threshold=ESS_THRESHOLD,
        resampling="systematic",
        max_weight=MAX_WEIGHT,
        rng=rng,
    )

    rmse_da: list[float] = []
    rmse_free: list[float] = []
    ess_history: list[float] = []

    for cycle in range(N_CYCLES):
        truth = integrate(truth, N_STEPS_PER_OBS)
        ensemble_da = integrate(ensemble_da, N_STEPS_PER_OBS)
        ensemble_free = integrate(ensemble_free, N_STEPS_PER_OBS)

        obs = truth[obs_idx] + rng.normal(0, OBS_SIGMA, n_obs)
        ensemble_obs = ensemble_da[:, obs_idx]

        # Observation-error tempering: evaluate the likelihood with an
        # inflated error so the weights remain informative instead of
        # collapsing onto a single particle.
        effective_err = OBS_TEMPERING_ETA * OBS_SIGMA

        ensemble_da, info = pf.assimilate(ensemble_da, ensemble_obs, obs, obs_err=effective_err)

        # Inflation: break genealogical collapse after resampling.
        if info.resampled:
            ensemble_da = ensemble_da + rng.normal(0, POST_RESAMPLE_INFLATION, ensemble_da.shape)

        mean_da = (
            ensemble_da.mean(axis=0)
            if info.resampled
            else np.average(ensemble_da, axis=0, weights=info.weights)
        )

        rmse_da.append(float(np.sqrt(np.mean((mean_da - truth) ** 2))))
        rmse_free.append(float(np.sqrt(np.mean((ensemble_free.mean(0) - truth) ** 2))))
        ess_history.append(info.ess)

        if cycle % 20 == 0:
            print(
                f"cycle={cycle:3d}  ESS={info.ess:6.1f}  "
                f"RMSE_DA={rmse_da[-1]:5.2f}  RMSE_FREE={rmse_free[-1]:5.2f}"
            )

    # Skip a brief spin-up for summary statistics.
    burn_in = 30
    da_mean = float(np.mean(rmse_da[burn_in:]))
    free_mean = float(np.mean(rmse_free[burn_in:]))

    print()
    print(f"Mean RMSE (cycles {burn_in}+)  DA   = {da_mean:.3f}")
    print(f"Mean RMSE (cycles {burn_in}+)  FREE = {free_mean:.3f}")

    # --- Figure ------------------------------------------------------
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_rmse, ax_ess) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    t_obs = np.arange(N_CYCLES) * OBS_INTERVAL

    ax_rmse.plot(t_obs, rmse_free, color="#999999", lw=1.4, label="Free ensemble")
    ax_rmse.plot(t_obs, rmse_da, color="#1f77b4", lw=1.6, label="Particle filter")
    ax_rmse.axhline(OBS_SIGMA, color="#d62728", lw=1.0, ls="--", label="obs. noise")
    ax_rmse.set_xlabel("model time")
    ax_rmse.set_ylabel("RMSE (ensemble mean − truth)")
    ax_rmse.set_title(
        f"Lorenz-96 twin experiment  (N={N_MEMBERS}, {n_obs}/{N_VARS} obs, η={OBS_TEMPERING_ETA:g})"
    )
    ax_rmse.legend(frameon=False, loc="upper right")
    ax_rmse.grid(alpha=0.25)
    ax_rmse.set_ylim(bottom=0)

    ax_ess.plot(t_obs, ess_history, color="#2ca02c", lw=1.4)
    ax_ess.axhline(
        ESS_THRESHOLD * N_MEMBERS,
        color="#d62728",
        lw=1.0,
        ls="--",
        label=f"threshold = {ESS_THRESHOLD:g} N",
    )
    ax_ess.set_xlabel("model time")
    ax_ess.set_ylabel("effective sample size")
    ax_ess.set_title("Filter health")
    ax_ess.set_ylim(0, N_MEMBERS * 1.05)
    ax_ess.legend(frameon=False, loc="upper right")
    ax_ess.grid(alpha=0.25)

    fig.suptitle(
        "pypfda on Lorenz-96:  observation tempering + inflation + max-weight cap",
        fontsize=11,
        y=1.04,
    )
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"figure written to {OUT_FIG}")

    if da_mean >= free_mean:
        raise SystemExit("Particle filter failed to beat the free run.")
    print("Particle filter beat the free run.")


if __name__ == "__main__":
    main()
