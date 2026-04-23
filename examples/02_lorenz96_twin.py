"""Lorenz-96 twin experiment: does the particle filter beat the free run?

The truth and the ensemble integrate the same Lorenz-96 ODE; the truth
is observed with Gaussian noise, the free ensemble is not assimilated,
and the analysis ensemble is corrected with a particle filter every
``OBS_INTERVAL`` time units.

Expected outcome: analysis-mean RMSE drops below the observation noise
within a few cycles, while the free ensemble saturates near the
climatological standard deviation.
"""

from __future__ import annotations

import numpy as np

from pypfda import ParticleFilter

N_VARS = 40
FORCING = 8.0
DT = 0.01
OBS_INTERVAL = 0.05
N_STEPS_PER_OBS = int(OBS_INTERVAL / DT)
N_CYCLES = 60
N_MEMBERS = 100
OBS_ERR = 0.5
SEED = 20260423


def lorenz96_rhs(x: np.ndarray, forcing: float = FORCING) -> np.ndarray:
    """Right-hand side of the Lorenz-96 system."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def rk4_step(x: np.ndarray, dt: float) -> np.ndarray:
    """One classical fourth-order Runge–Kutta step."""
    k1 = lorenz96_rhs(x)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2)
    k4 = lorenz96_rhs(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(x: np.ndarray, n_steps: int, dt: float = DT) -> np.ndarray:
    """Integrate ``n_steps`` RK4 steps."""
    for _ in range(n_steps):
        x = rk4_step(x, dt)
    return x


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Spin-up the truth.
    truth = np.full(N_VARS, FORCING)
    truth[0] += 0.01
    truth = integrate(truth, n_steps=2000)

    # Observation operator: every other variable.
    obs_idx = np.arange(0, N_VARS, 2)
    n_obs = obs_idx.size

    # Initial ensemble: perturb the truth.
    ensemble_da = truth + rng.normal(0, 1.0, size=(N_MEMBERS, N_VARS))
    ensemble_free = ensemble_da.copy()

    pf = ParticleFilter(ess_threshold=0.5, resampling="systematic", rng=rng)

    rmse_da, rmse_free = [], []
    ess_history = []

    for cycle in range(N_CYCLES):
        # Forward integrate everything.
        truth = integrate(truth, N_STEPS_PER_OBS)
        for m in range(N_MEMBERS):
            ensemble_da[m] = integrate(ensemble_da[m], N_STEPS_PER_OBS)
            ensemble_free[m] = integrate(ensemble_free[m], N_STEPS_PER_OBS)

        # Generate observations.
        obs = truth[obs_idx] + rng.normal(0, OBS_ERR, n_obs)

        # Assimilate.
        ensemble_obs = ensemble_da[:, obs_idx]
        ensemble_da, info = pf.assimilate(ensemble_da, ensemble_obs, obs, obs_err=OBS_ERR)

        rmse_da.append(float(np.sqrt(np.mean((ensemble_da.mean(0) - truth) ** 2))))
        rmse_free.append(
            float(np.sqrt(np.mean((ensemble_free.mean(0) - truth) ** 2)))
        )
        ess_history.append(info.ess)

        print(
            f"cycle={cycle:3d}  ESS={info.ess:6.1f}  "
            f"RMSE_DA={rmse_da[-1]:5.2f}  RMSE_FREE={rmse_free[-1]:5.2f}"
        )

    # Final summary.
    print()
    print(f"Mean RMSE (last 30 cycles)  DA   = {np.mean(rmse_da[-30:]):.3f}")
    print(f"Mean RMSE (last 30 cycles)  FREE = {np.mean(rmse_free[-30:]):.3f}")
    if np.mean(rmse_da[-30:]) >= np.mean(rmse_free[-30:]):
        raise SystemExit("Particle filter failed to beat the free run.")
    print("Particle filter beat the free run.")


if __name__ == "__main__":
    main()
