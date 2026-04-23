# reproduce/

This directory is intentionally almost empty.

The companion paper (Fallah et al., 2026, npj Climate and Atmospheric
Science, in review) is an Observing System Simulation Experiment using
100-member ensembles of the coupled CM2Mc-BLING Earth system model,
integrated for ~100 years each. The raw ensemble output is on the
order of a terabyte and required several months of cluster wall time
to produce; it cannot be shipped with this package and re-running it
would require comparable resources.

We therefore make no promise that `pypfda` reproduces every figure in
the paper from scratch. What this repository does provide:

- the **methods** the paper relies on (sequential importance
  resampling, observation tempering, post-resample inflation,
  max-weight degeneracy cap, Gaspari–Cohn localization), validated on
  Lorenz-96 in `examples/03_lorenz96_twin.py`;
- a clean open-source citation target for the algorithm.

If you need the ensemble data itself — for a review, a follow-up
analysis, or to reproduce a specific figure — please contact the
authors (<bijan.fallah@gmail.com>).

A small Zenodo deposit of *processed* diagnostics (time series of
AMOC, ESS, weights, spectra — tens of MB, not terabytes) may be minted
post-acceptance. If that happens, the scripts that read it will land
here with a matching DOI.
