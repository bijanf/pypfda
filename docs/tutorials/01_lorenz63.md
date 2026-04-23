# Tutorial: Lorenz-63 twin experiment

Lorenz-63 (Lorenz, 1963) is the canonical 3-D chaotic system — the
"butterfly" attractor. With three state variables and one observed
variable it sits squarely in the regime where a vanilla SIR particle
filter should comfortably beat a free ensemble. In this tutorial we
run a *twin experiment*: a single "truth" trajectory generates
synthetic observations, and we ask whether the particle filter can
recover the truth from a perturbed ensemble.

The script lives at `examples/02_lorenz63_twin.py` in the repository.

## The model

$$
\dot x = \sigma (y - x), \qquad
\dot y = x (\rho - z) - y, \qquad
\dot z = x y - \beta z,
$$

with $\sigma = 10$, $\rho = 28$, $\beta = 8/3$. We integrate with
fourth-order Runge–Kutta at $\Delta t = 0.01$.

## Experimental setup

- 3-dimensional state, 1 observed variable ($x$).
- 500 ensemble members.
- Gaussian observation error, $\sigma_o = 1$.
- Observation interval $\Delta t_{\mathrm{obs}} = 0.5$ (about half a
  Lyapunov time).
- 80 analysis cycles.
- After each resampling, add Gaussian inflation noise with standard
  deviation 0.5 to all members; this keeps the ensemble from
  collapsing to a single trajectory after a heavy resampling event.

## Expected outcome

The DA analysis-mean RMSE settles around the observation noise
($\approx 1$), while the free-running ensemble drifts to the
climatological RMSE ($\approx 7$). On the reference run the script
prints

```
Mean RMSE (cycles 10+)  DA   = 1.394
Mean RMSE (cycles 10+)  FREE = 7.340
Particle filter beat the free run.
```

a roughly 5× error reduction.

## Why inflation?

A SIR particle filter is *consistent* with the Bayesian posterior in
the limit of infinitely many particles, but with a finite ensemble a
heavy resampling event leaves many duplicated members behind. Without
any process noise these duplicates evolve identically and the filter
collapses. Adding small Gaussian noise after resampling is the
simplest fix and is enough for Lorenz-63. For systems with long
memory (such as the deep ocean) inflation creates a different
problem — the *diversity–memory trade-off* discussed in
{doc}`../theory/particle_filter`.

> The script is not yet rendered as an executable notebook; it will be
> in the next minor release once `myst-nb` and the data dependencies
> are wired up.
