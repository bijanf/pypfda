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
  - Sphinx documentation skeleton with Furo theme.
  - Continuous integration on Linux, macOS, and Windows for Python
    3.10–3.12.
  - PyPI trusted publishing on tagged releases.

[Unreleased]: https://github.com/bijanf/pypfda/compare/main...HEAD
