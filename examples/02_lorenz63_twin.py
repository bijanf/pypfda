"""Lorenz-63 twin experiment: does the particle filter beat the free run?

Lorenz-63 is the classic 3-D chaotic system (Lorenz, 1963). With three
state dimensions and one observed variable, a vanilla SIR particle
filter is well within the regime where it should robustly beat a free
ensemble.

The truth and the ensemble integrate the same Lorenz-63 ODE; the truth
is observed in its first variable with Gaussian noise; the free
ensemble is not assimilated; the analysis ensemble is corrected with a
particle filter every ``OBS_INTERVAL`` time units.

The observation interval is deliberately long (~ one Lyapunov time) so
that the ensemble spread grows enough between cycles to make the
particle weights informative.

Expected outcome: DA RMSE drops well below the free RMSE within a few
cycles.
"""

from __future__ import annotations

import numpy as np

from pypfda import ParticleFilter

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.01
OBS_INTERVAL = 0.5  # frequent enough to keep the filter on track
N_STEPS_PER_OBS = int(OBS_INTERVAL / DT)
N_CYCLES = 80
N_MEMBERS = 500
OBS_ERR = 1.0
INITIAL_SPREAD = 3.0
INFLATION = 0.5  # Gaussian noise stdev added after each resampling
SEED = 20260423


def lorenz63_rhs(state: np.ndarray) -> np.ndarray:
    """Right-hand side of Lorenz-63. Vectorized over leading dims."""
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dxdt = SIGMA * (y - x)
    dydt = x * (RHO - z) - y
    dzdt = x * y - BETA * z
    return np.stack([dxdt, dydt, dzdt], axis=-1)


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = lorenz63_rhs(state)
    k2 = lorenz63_rhs(state + 0.5 * dt * k1)
    k3 = lorenz63_rhs(state + 0.5 * dt * k2)
    k4 = lorenz63_rhs(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state: np.ndarray, n_steps: int, dt: float = DT) -> np.ndarray:
    for _ in range(n_steps):
        state = rk4_step(state, dt)
    return state


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Spin up the truth onto the attractor.
    truth = np.array([1.0, 1.0, 1.0])
    truth = integrate(truth, n_steps=2000)

    # Initial ensembles: large perturbation around the truth so members
    # explore different attractor lobes.
    ensemble_da = truth + rng.normal(0, INITIAL_SPREAD, size=(N_MEMBERS, 3))
    ensemble_free = ensemble_da.copy()

    pf = ParticleFilter(ess_threshold=0.5, resampling="systematic", rng=rng)

    rmse_da: list[float] = []
    rmse_free: list[float] = []

    for cycle in range(N_CYCLES):
        truth = integrate(truth, N_STEPS_PER_OBS)
        ensemble_da = integrate(ensemble_da, N_STEPS_PER_OBS)
        ensemble_free = integrate(ensemble_free, N_STEPS_PER_OBS)

        obs = np.array([truth[0] + rng.normal(0, OBS_ERR)])
        ensemble_obs = ensemble_da[:, [0]]

        ensemble_da, info = pf.assimilate(ensemble_da, ensemble_obs, obs, obs_err=OBS_ERR)

        # After resampling, particles are duplicates: add small Gaussian
        # inflation noise so the ensemble does not collapse to a single
        # trajectory. Without this step the filter degenerates after the
        # first heavy resampling event (the diversity-memory trade-off).
        if info.resampled:
            ensemble_da = ensemble_da + rng.normal(0, INFLATION, ensemble_da.shape)

        # Use the weighted mean if no resampling, plain mean otherwise
        # (after resampling all weights are uniform).
        mean_da = (
            ensemble_da.mean(axis=0)
            if info.resampled
            else np.average(ensemble_da, axis=0, weights=info.weights)
        )
        rmse_da.append(float(np.sqrt(np.mean((mean_da - truth) ** 2))))
        rmse_free.append(float(np.sqrt(np.mean((ensemble_free.mean(0) - truth) ** 2))))

        print(
            f"cycle={cycle:3d}  ESS={info.ess:6.1f}  resample={str(info.resampled):5}  "
            f"RMSE_DA={rmse_da[-1]:5.2f}  RMSE_FREE={rmse_free[-1]:5.2f}"
        )

    # Skip the first 10 cycles for filter spin-up.
    da_mean = float(np.mean(rmse_da[10:]))
    free_mean = float(np.mean(rmse_free[10:]))

    print()
    print(f"Mean RMSE (cycles 10+)  DA   = {da_mean:.3f}")
    print(f"Mean RMSE (cycles 10+)  FREE = {free_mean:.3f}")
    if da_mean >= free_mean:
        raise SystemExit("Particle filter failed to beat the free run.")
    print("Particle filter beat the free run.")


if __name__ == "__main__":
    main()
