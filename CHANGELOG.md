# Changelog

All notable changes to `pypfda` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release skeleton:
  - `ParticleFilter` core class with sequential importance resampling.
  - Numerically stable weight computation via log-sum-exp.
  - Effective sample size (ESS) and weight entropy diagnostics.
  - Systematic, stratified, residual, and multinomial resampling.
  - Gaspari–Cohn spatial localization (great-circle distance).
  - Property-based tests for weight and resampling invariants.
  - End-to-end Lorenz-63 twin experiment.
  - Lorenz-96 twin experiment combining observation-error tempering,
    post-resample Gaussian inflation, and max-weight degeneracy cap;
    produces the reference figure embedded in the README
    (`docs/_static/lorenz96_demo.png`).
  - Paper-claim analogues on Lorenz-96: cycle-length sensitivity
    (`04_l96_cycle_sensitivity.py`), diversity–memory trade-off
    (`05_l96_diversity_memory.py`, now a 5-seed sweep with ±1σ
    envelope), diverse-IC robustness (`06_l96_diverse_ics.py`, with
    every member's first-variable trajectory overplotted), Welch /
    Nyquist spectral analysis (`07_l96_nyquist.py`). Shared
    Lorenz-96 helpers factored into `examples/_l96_common.py`.
  - Dedicated docs tutorial page
    `docs/tutorials/02_paper_claims_on_l96.md` walking through all
    four experiments with derivations and explicit caveats. The
    cycle-length section is framed as a demonstration that the
    paper's two intrinsic bounds (Lyapunov and Nyquist) both exist
    on Lorenz-96 with the predicted values, rather than as a
    reproduction of the AMOC U-curve itself.
  - Sphinx documentation skeleton with Furo theme.
  - Continuous integration on Linux, macOS, and Windows for Python
    3.10–3.12.
  - PyPI trusted publishing on tagged releases.

[Unreleased]: https://github.com/bijanf/pypfda/compare/main...HEAD
