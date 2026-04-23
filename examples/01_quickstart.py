"""Minimal working example: assimilate noisy linear observations.

Run this script with ``python examples/01_quickstart.py`` after
installing pypfda.
"""

from __future__ import annotations

import numpy as np

from pypfda import ParticleFilter


def main() -> None:
    rng = np.random.default_rng(0)
    n_members = 100
    n_obs = 5
    state_dim = 4

    H = rng.normal(size=(n_obs, state_dim))
    truth = rng.normal(size=state_dim)
    ensemble = rng.normal(size=(n_members, state_dim))

    pf = ParticleFilter(ess_threshold=0.5, resampling="systematic", rng=rng)

    for step in range(50):
        # 1. Forward integration of the truth and the ensemble.
        truth = 0.95 * truth + rng.normal(0, 0.5, state_dim)
        ensemble = 0.95 * ensemble + rng.normal(0, 0.5, ensemble.shape)

        # 2. Generate observations from the truth.
        obs = H @ truth + rng.normal(0, 0.1, n_obs)

        # 3. Predict observations from each ensemble member.
        obs_pred = ensemble @ H.T

        # 4. Analysis (and optional resampling).
        ensemble, info = pf.assimilate(ensemble, obs_pred, obs, obs_err=0.1)

        if step % 5 == 0:
            mean = ensemble.mean(axis=0)
            rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
            print(
                f"step={step:3d}  ESS={info.ess:6.1f}  "
                f"resampled={info.resampled!s:5}  RMSE={rmse:.3f}"
            )


if __name__ == "__main__":
    main()
