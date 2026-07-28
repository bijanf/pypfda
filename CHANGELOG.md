# Changelog

All notable changes to `pypfda` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-07-28

Adds transient-forcing support to the CLIMBER-X adapter and two idealized Lorenz-63
examples. No change to the assimilation core, so results produced with 1.0.0 are
reproduced exactly by this release.

### Added

- `ClimberXConfig.year_ini_start`: the model calendar year the first segment starts
  at. `forecast()` now advances `year_ini = year_ini_start + elapsed` each cycle, so
  the calendar runs continuously across segments and prescribed solar/volcanic/CO2
  forcing tracks real years. Set it to e.g. `-1000` for a last-millennium forced
  run; the default `0` keeps the constant-forcing behaviour used in the paper.
- `ClimberXConfig.archive_sst`: when true, `observe()` writes each member's
  window-mean SST field to `<member>/sst_archive/cycle_NNN.npy`, for validating a
  reconstructed SST field against observations. Off by default.
- `examples/11_classic_l63_partial_obs.py`: the classic Lorenz-63 twin experiment
  under partial observation, the cleanest laptop-scale demonstration that the filter
  recovers an unobserved component.
- `examples/10_coupled_l63_clearly_works.py`: a coupled fast--slow Lorenz-63 twin
  OSSE in a well-posed regime, with its diagnostic figures under `docs/_static/`.
- `tests/test_climberx_hosing.py`: regression tests pinning the hosing forcing file
  to the segment's calendar (see Fixed).

### Fixed

- The stochastic-hosing forcing file is now written on the segment's own calendar.
  `write_hosing_file()` gained a `year0` argument and its time axis runs
  `year0 .. year0+years-1` instead of always starting at 0. With an advancing
  `year_ini` a 0-based axis leaves every segment after the first entirely outside
  the file, so the model would not find its forcing. Only affects runs with
  `hosing_sigma > 0`; `year0` defaults to 0, so the 1.0.0 behaviour is unchanged.

### Changed

- CLIMBER-X restarts are resolved by absolute end year (`year_ini + nyears`) rather
  than by `nyears`, matching how the model names them once the calendar advances.
- `OMP_STACKSIZE` for the CLIMBER-X subprocess raised from 256M to 512M.
- README: expanded usage documentation.

## [1.0.0] - 2026-06-10

First tagged release, accompanying the methodological paper (Fallah et al.,
*Online Particle Filter Data Assimilation for AMOC Reconstruction*).

### Added

- Coupled fast--slow reference adapters in `pypfda.models.lorenz`:
  `TwoScaleLorenz96` (two-level Lorenz-96; Lorenz 1996, Lorenz & Emanuel
  1998) and `CoupledLorenz63` (two-timescale coupled Lorenz-63; Peña &
  Kalnay 2004). Both are first-class `ForwardModel`s driven by the *same*
  `CycleDriver` as the coupled-GCM cores: observe the FAST variable,
  reconstruct the SLOW one — the minimal, laptop-runnable analogue of
  reconstructing AMOC from SST. They record the *forecast* diagnostic
  (deliberately not swapped by `set_state`), so they pass the
  forecast-convention conformance check in `pypfda.verify`.
- Worked twin-OSSE examples `08_two_scale_l96_fast_slow.py` and
  `09_coupled_l63_fast_slow.py` (TRUTH/FREE/DA), each gated by
  `pypfda.verify.scan_osse_result`, sharing a `_coupled_lorenz_common.py`
  driver.
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

[Unreleased]: https://github.com/bijanf/pypfda/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bijanf/pypfda/releases/tag/v1.0.0
