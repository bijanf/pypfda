# Quickstart

A minimal example: assimilate noisy linear observations into a small
ensemble of 1-D states.

```{code-block} python
import numpy as np
from pypfda import ParticleFilter

rng = np.random.default_rng(0)
n_members = 100
n_obs = 5

# A toy linear-Gaussian forward model.
def forecast(state):
    return 0.95 * state + rng.normal(0, 0.5, state.shape)

H = rng.normal(size=(n_obs, 4))
truth = rng.normal(size=4)
ensemble = rng.normal(size=(n_members, 4))

pf = ParticleFilter(ess_threshold=0.5, resampling="systematic")

for step in range(50):
    # 1. Forward integration.
    truth = 0.95 * truth + rng.normal(0, 0.5, 4)
    ensemble = np.array([forecast(m) for m in ensemble])

    # 2. Generate observations.
    obs = H @ truth + rng.normal(0, 0.1, n_obs)

    # 3. Predict observations from each ensemble member.
    obs_pred = ensemble @ H.T

    # 4. Analysis (and optional resampling).
    ensemble, info = pf.assimilate(ensemble, obs_pred, obs, obs_err=0.1)

    print(f"step={step:3d}  ESS={info.ess:6.1f}  resampled={info.resampled}")
```

The {class}`~pypfda.ParticleFilter` does not run your forward model — you
do, between calls. This separation of concerns means the same filter
works with a 1-D toy, a Lorenz-96 system, or a coupled climate model.

## What `assimilate` returns

The second return value is an {class}`~pypfda.AssimilationInfo`
dataclass with the analysis weights, effective sample size, entropy,
whether resampling occurred, and (if so) the resampling indices. Use it
for diagnostics; ignore it if you do not care.

## Where to next

- The [Lorenz-96 tutorial](tutorials/01_lorenz96.md) shows a full twin
  experiment with skill metrics.
- The [particle filter theory page](theory/particle_filter.md) walks
  through the SIR derivation and explains the design choices behind the
  `ParticleFilter` class.
- The [API reference](api/index.md) documents every public function.
