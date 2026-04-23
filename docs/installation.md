# Installation

`pypfda` requires Python ≥ 3.10. We recommend installing into a virtual
environment.

## From PyPI

```bash
pip install pypfda                 # core only
pip install 'pypfda[io,plot]'      # + NetCDF and matplotlib helpers
pip install 'pypfda[paleo]'        # + coral PSM and PAGES 2k loader
pip install 'pypfda[all]'          # everything including dev + docs
```

## From source (development install)

```bash
git clone https://github.com/bijanf/pypfda.git
cd pypfda
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[all]'
pre-commit install
```

After installing in development mode you can run the test suite:

```bash
pytest
mypy src/pypfda
ruff check src/ tests/
```

## Optional dependencies

| Extra    | Pulls in                                | Used for                          |
| -------- | --------------------------------------- | --------------------------------- |
| `io`     | `netCDF4`, `xarray`, `pyyaml`           | NetCDF I/O, YAML config loader    |
| `plot`   | `matplotlib`                            | Diagnostic plots                  |
| `paleo`  | `pandas` (+ `io`, `plot`)               | Coral PSM, PAGES 2k loader        |
| `dev`    | `pytest`, `mypy`, `ruff`, `hypothesis`  | Development & contribution        |
| `docs`   | `sphinx`, `furo`, `myst-parser`         | Building documentation            |
| `all`    | Everything above                        | One-stop install                  |
