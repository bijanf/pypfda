# Tutorial: Lorenz-96 twin experiment

The Lorenz-96 system is the standard data-assimilation benchmark: a
40-variable chaotic ODE with a single nonlinearity that captures the
flavor of atmospheric dynamics in a tractable system. In this tutorial
we run a *twin experiment*: a single "truth" trajectory generates
synthetic observations, and we ask whether the particle filter can
recover the truth from a perturbed ensemble.

The script lives at `examples/02_lorenz96_twin.py` in the repository.

## The model

The 40-variable Lorenz-96 ODE is

$$
\dot x_i \;=\; (x_{i+1} - x_{i-2}) \, x_{i-1} \;-\; x_i \;+\; F,
\qquad i=1,\dots,40,
$$

with cyclic indices and forcing $F=8$.

## Experimental setup

- 40-dimensional state.
- 100 ensemble members.
- Observe every other variable with Gaussian noise of standard
  deviation $\sigma_o = 0.5$.
- Observation interval: $\Delta t = 0.05$.
- Compare the analysis ensemble mean against a *free* ensemble (no
  observations).

## Expected outcome

After ~20 cycles the analysis-mean RMSE drops below the observation
noise and stays there, while the free ensemble RMSE saturates at the
climatological standard deviation (≈ 3). The full script in
`examples/02_lorenz96_twin.py` prints these RMSEs and produces a
diagnostic plot of ESS over time.

> The script is not yet rendered as an executable notebook; it will be
> in the next minor release once `myst-nb` and the data dependencies
> are wired up.
