# pypfda

**Particle filter data assimilation in pure Python**, with first-class
support for paleoclimate Observing System Simulation Experiments (OSSEs).

[![CI](https://github.com/bijanf/pypfda/actions/workflows/ci.yml/badge.svg)](https://github.com/bijanf/pypfda/actions/workflows/ci.yml)
[![Docs](https://github.com/bijanf/pypfda/actions/workflows/docs.yml/badge.svg)](https://bijanf.github.io/pypfda)
[![Lint](https://github.com/bijanf/pypfda/actions/workflows/lint.yml/badge.svg)](https://github.com/bijanf/pypfda/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/bijanf/pypfda/branch/main/graph/badge.svg)](https://codecov.io/gh/bijanf/pypfda)
[![PyPI](https://img.shields.io/pypi/v/pypfda.svg)](https://pypi.org/project/pypfda/)
[![Python](https://img.shields.io/pypi/pyversions/pypfda.svg)](https://pypi.org/project/pypfda/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![status: work in progress](https://img.shields.io/badge/status-work_in_progress-orange.svg)](#status)

> **Status — work in progress.**
> `pypfda` is the open-source companion to a paper currently under
> review at *npj Climate and Atmospheric Science*
> ([Fallah et al., 2026](#citing-pypfda)). The core particle filter,
> weight / ESS / resampling primitives, observation tempering,
> post-resample inflation, max-weight degeneracy cap and Gaspari–Cohn
> localization are in place and tested; higher-level diagnostics
> (genealogy tracking, Welch / Nyquist spectral tools) and the paleo
> forward-model subpackage are on the roadmap. Public APIs may evolve
> before `v1.0`. Pin a specific version in production code.

`pypfda` provides a clean, model-agnostic implementation of the sequential
importance resampling (SIR) particle filter, plus the building blocks needed
to deploy it on real Earth-system problems: spatial localization, ensemble
inflation, degeneracy prevention, multi-year assimilation windows, and
ensemble diagnostics (effective sample size, weight entropy, genealogy,
spectral analysis).

The core filter knows nothing about climate models. You bring a forward
model — a Lorenz-96 toy, a coupled GCM, anything callable — and `pypfda`
runs the analysis cycle.

## It works on Lorenz-96

![Lorenz-96 twin experiment: DA vs free ensemble and ESS](docs/_static/lorenz96_demo.png)

A 40-variable Lorenz-96 twin experiment with 400 members, observations
on every other variable, observation-error tempering (η = 4),
post-resample Gaussian inflation, and a max-weight cap (0.3). The
particle filter drives ensemble-mean RMSE down to roughly half of the
free-ensemble RMSE and keeps the effective sample size well above the
degeneracy threshold. Reproduce with

```bash
python examples/03_lorenz96_twin.py
```

This is the same combination of techniques the companion paper applies
in an online coupled-climate-model OSSE; Lorenz-96 is the smallest
chaotic benchmark on which those techniques can be exhibited end-to-end
without a climate model in the loop.

## Paper claims reproduced on Lorenz-96

Four of the paper's central methodological findings re-expressed on
Lorenz-96. Each is a self-contained script that runs on a laptop in
seconds to a few minutes and writes a single figure to
`docs/_static/`. These are analogues, not reproductions — Lorenz-96 is
not AMOC — but they show that the filter's qualitative behaviour is
not an artefact of the coupled model.

### 1. Diversity–memory trade-off (§5, headline claim)

![Diversity-memory trade-off on Lorenz-96](docs/_static/l96_diversity_memory.png)

A post-resampling Gaussian inflation sweep, repeated across **5
independent rng seeds** so that the U-curve is demonstrably not a
single-seed artefact. Zero inflation collapses the ensemble to
duplicates and the filter dies (RMSE ≈ 4.75 ± 0.09); too-much
inflation washes out the accumulated information (RMSE rises back to
≈ 4.12 ± 0.21 at σ = 2). A clean U-curve with optimum near σ ≈ 0.35,
RMSE ≈ 1.66 ± 0.41. The across-seed envelope (shaded blue) narrows at
the endpoints (robust collapse and robust over-noise) and widens at
the optimum (the filter is on the edge of its stable regime, where
different seeds land in different minima of the same U). The right
panel re-parameterises the seed-mean sweep as skill vs achieved
diversity. Reproduce with `python examples/05_l96_diversity_memory.py`.

### 2. Cycle-length sensitivity — the two intrinsic bounds behind T10 / T11 / T12

![Lorenz-96 cycle-length sensitivity](docs/_static/l96_cycle_sensitivity.png)

The paper's cycle-length finding for AMOC is that two intrinsic
timescales constrain the usable observation interval:

* the **Lyapunov time** (how fast forecast uncertainty grows without
  observations), and
* the **Nyquist bound** (half the period of the dominant spectral
  mode).

The *tighter* of the two determines where the filter breaks down. On
AMOC, the coupled-system Lyapunov time is decades while Nyquist is
~6.7 years (from the 13.3-year AMOC mode), so Nyquist binds and a
5-year cycle (T11) is optimum.

This figure shows the same two bounds on Lorenz-96, correctly marked.
On L96 the ordering is reversed: Lyapunov is ~0.42 and the Nyquist
bound (from the spectral peak in figure 4 below) is ~0.83, so
Lyapunov binds and the filter breaks down at ~0.4, well before any
aliasing kicks in. We are **not** claiming L96 reproduces the AMOC
U-curve — it doesn't, and it cannot, because L96's chaos horizon is
short relative to its oscillation period. We are claiming both bounds
exist here, they are measurable, they have the values the theory
predicts, and which one matters is set by the system, not the
algorithm. That is the point the paper makes, made visible on a
benchmark that fits on a laptop. Reproduce with
`python examples/04_l96_cycle_sensitivity.py`.

### 3. Diverse-IC ensemble (analogue of T13)

![Diverse-IC ensemble on Lorenz-96](docs/_static/l96_diverse_ics.png)

Each of 200 members is independently spun up on the attractor, so
the initial ensemble is maximally diverse. Left: the first state
variable over time. DA (blue) tracks the truth (black); the free
ensemble mean (grey) collapses to climatology because averaging
uncorrelated attractor trajectories returns the attractor mean.
Right: ensemble-mean RMSE over time. DA pulls the scattered ensemble
onto the truth trajectory without shared initial-state memory,
matching the paper's T13 finding. Reproduce with
`python examples/06_l96_diverse_ics.py`.

### 4. Welch spectrum and the Nyquist argument (§4, Figure 2 analogue)

![Welch spectrum and autocorrelation of Lorenz-96](docs/_static/l96_spectrum.png)

A 2000-unit control integration of Lorenz-96 shows a genuine spectral
peak at T ≈ 1.65 model time units with peak / median power ratio
≈ 134 — sharp, not broadband. The corresponding Nyquist bound (T/2
≈ 0.83) is the same one overlaid on figure 2 above. The
autocorrelation panel independently reports an e-folding time of
0.30, consistent with the Lyapunov-limited predictability horizon.
Reproduce with `python examples/07_l96_nyquist.py`.

The walk-through with derivations and extra figures lives in the docs
at [*Paper claims reproduced on Lorenz-96*](https://bijanf.github.io/pypfda/tutorials/02_paper_claims_on_l96.html).

## Scope — what `pypfda` is and is not

`pypfda` ships the **data-assimilation method**, not a turnkey coupling
to any specific Earth-system model. In concrete terms:

**What is here, today.** The particle filter itself (SIR), numerically
stable weight and ESS computation, four resampling schemes, Gaspari–Cohn
localization, observation-error tempering, a max-weight degeneracy cap,
and post-resample inflation. Demonstrated end-to-end on Lorenz-63 and
Lorenz-96.

**What is not here, today, and will not arrive by magic.**

- Integration glue for any specific climate model (CM2Mc-BLING, MITgcm,
  CESM, ICON, …). Nothing in this repository knows how to write an FV3
  restart file, launch a SLURM job array, or cycle ensemble state
  between `INPUT/` and `RESTART/` directories.
- The HPC orchestration used by the companion paper — that is a
  separate, cluster-specific driver that calls `pypfda` functions but
  lives elsewhere.
- A coral δ¹⁸O proxy forward model, a PAGES 2k loader, or any
  paleo-specific I/O. These are planned for `pypfda.paleo` but are
  not yet implemented.
- The 100-member ensemble output underlying the paper's figures (that
  sits on the authors' cluster; see [Paper data](#paper-data) below).

**Using `pypfda` with your own model is real engineering work.** You
are expected to write the driver that runs your forward model, reads
its state into a NumPy array, calls `pf.assimilate(...)`, writes the
resampled state back, and handles your own HPC scheduling. For
Lorenz-style ODEs that driver is a few dozen lines (see
`examples/`). For a coupled climate model with distributed restart
files and a queue system, expect weeks of integration effort per
model. If you are attempting this and want guidance,
[open an issue](https://github.com/bijanf/pypfda/issues) or email the
authors — we are happy to help, but there is no plug-and-play path.

## Highlights

- **Pure Python.** No Fortran, no compilation. Works on Linux, macOS, and
  Windows.
- **Model-agnostic.** The filter is decoupled from any specific simulator;
  bring your own forward step.
- **Numerically careful.** Log-domain weight computation, well-tested
  resampling routines (systematic, stratified, residual, multinomial),
  numerically stable ESS.
- **Diagnostics.** ESS, weight entropy, genealogical diversity, rank
  histograms, spectral / Nyquist analysis for cycle-length design.
- **Paleoclimate-ready.** Optional `paleo` extra includes coral δ¹⁸O
  proxy system models and a PAGES 2k loader.
- **Production tooling.** Strict typing (`mypy --strict`), property-based
  tests, ≥ 80 % coverage, conventional commits, semantic versioning,
  reproducible builds via `hatchling`.

## Installation

```bash
pip install pypfda                 # core only
pip install 'pypfda[io,plot]'      # + NetCDF and matplotlib helpers
pip install 'pypfda[paleo]'        # + coral PSM and PAGES 2k loader
pip install 'pypfda[all]'          # everything including dev + docs
```

`pypfda` requires Python ≥ 3.10.

## 60-second example

```python
import numpy as np
from pypfda import ParticleFilter

rng = np.random.default_rng(0)
n_members, n_obs = 100, 5

# Linear Gaussian toy: x_{t+1} = 0.95 x_t + w,   y_t = H x_t + v
def forecast(state):
    return 0.95 * state + rng.normal(0, 0.5, state.shape)

H = rng.normal(size=(n_obs, 4))
truth = rng.normal(size=4)
ensemble = rng.normal(size=(n_members, 4))

pf = ParticleFilter(ess_threshold=0.5, resampling="systematic")

for _ in range(50):
    truth = 0.95 * truth + rng.normal(0, 0.5, 4)
    ensemble = np.array([forecast(m) for m in ensemble])
    obs = H @ truth + rng.normal(0, 0.1, n_obs)
    obs_pred = ensemble @ H.T
    ensemble, info = pf.assimilate(ensemble, obs_pred, obs, obs_err=0.1)
    print(f"ESS={info.ess:.1f}  resampled={info.resampled}")
```

See the [quickstart](https://bijanf.github.io/pypfda/quickstart.html) and
[Lorenz-63 tutorial](https://bijanf.github.io/pypfda/tutorials/01_lorenz63.html)
for full walk-throughs.

## Documentation

- **Theory** — derivation of the SIR update, comparison of resampling
  schemes, the diversity/memory trade-off, choosing inflation and
  localization parameters.
- **Tutorials** — Lorenz-63 twin experiment and, for high-dimensional
  chaos, the Lorenz-96 demo used by the figure above.
- **API reference** — every public function and class, generated by
  Sphinx + autosummary.

Read the docs at <https://bijanf.github.io/pypfda>.

## Citing pypfda

If you use `pypfda` in published work, please cite the software (via the
`CITATION.cff` button on GitHub) **and** the methodological paper:

> Fallah, B., et al. (2026). Bidirectional AMOC–SST coupling on fast and
> slow timescales: Causal discovery and particle filter perspectives for
> paleoclimate reconstruction. *npj Climate and Atmospheric Science*, in
> review.

A BibTeX snippet is provided in [CITATION.cff](CITATION.cff).

## Paper data

The companion paper is an Observing System Simulation Experiment built
on the coupled CM2Mc-BLING climate model with 100-member ensembles
integrated for ~100 years each (several months of cluster wall time per
experiment, of order one terabyte of netCDF output). That raw ensemble
is **not** distributed with this repository and would be impractical
to re-generate from scratch. It currently resides on the Potsdam
Institute for Climate Impact Research (PIK) cluster; interested
researchers are welcome to contact the authors for access or for
processed diagnostics.

What *is* in this repository is the **method**: a model-agnostic
implementation of the techniques the paper applies (SIR, observation
tempering, post-resample inflation, max-weight cap, Gaspari–Cohn
localization), validated on Lorenz-96 (see the figure above). That is
what the paper's Code Availability statement points to.

Scripts that operate on small processed diagnostics — for example the
Welch / Nyquist spectral analysis of the control AMOC time series —
may be added under `reproduce/` in a later release, driven by a small
Zenodo deposit. Regeneration of the full figure set is **not** a goal
of this package.

## Related work

- [`DA_offline_PF`](https://github.com/dalaiden/DA_offline_PF) — the
  Fortran offline particle filter from Dalaiden et al. that motivated
  this Python implementation.
- Dubinkina, S. et al. (2011), *Testing a particle filter to reconstruct
  climate changes over the past centuries*, IJBC 21, 3611.
- Goosse, H. et al. (2010), *Reconstructing surface temperature changes
  over the past 600 years using climate model simulations with data
  assimilation*, JGR 115.

## Contributing

Bug reports, feature requests, and pull requests are welcome. Please
read [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of
Conduct](CODE_OF_CONDUCT.md) before opening an issue or PR.

## License

`pypfda` is distributed under the [MIT License](LICENSE).
