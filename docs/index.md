# pypfda

Particle filter data assimilation in pure Python, with first-class support
for paleoclimate Observing System Simulation Experiments.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 1
:caption: Theory

theory/particle_filter
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/01_lorenz63
```

```{toctree}
:maxdepth: 1
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
contributing
```

## Why pypfda?

Sequential importance resampling particle filters are conceptually
simple but easy to implement badly: log-domain weight underflow,
inadvertent ensemble collapse, sensitivity to particle ordering.
`pypfda` ships well-tested primitives (log-sum-exp weights, four
resampling schemes, Gaspari–Cohn localization, max-weight degeneracy
prevention) so you can focus on your science problem rather than
debugging the filter.

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
