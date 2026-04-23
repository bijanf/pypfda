# Reproducing the figures from the paper

This directory contains scripts that regenerate every figure in

> Fallah, B. et al. (2026). *Bidirectional AMOC–SST coupling on fast and
> slow timescales: Causal discovery and particle filter perspectives for
> paleoclimate reconstruction*. npj Climate and Atmospheric Science (in
> review).

The scripts read a small Zenodo data deposit (~5–10 GB) of pre-computed
ensemble diagnostics; they do **not** require the underlying CM2Mc-BLING
climate model.

## Workflow

```bash
pip install 'pypfda[paleo,plot]'
cd reproduce
bash download_data.sh     # ~5–10 GB to data/
python fig01_proxy_network.py
python fig02_nyquist.py
# ...
```

Output figures land in `figures/` (created on first run).

## Status

Scripts will be added in v0.2.0, alongside the Zenodo deposit. Until
then this directory is a placeholder and `download_data.sh` is a stub.
