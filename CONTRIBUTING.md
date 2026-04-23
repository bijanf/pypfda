# Contributing to pypfda

Thank you for your interest in `pypfda`. This document outlines how to set
up a development environment, the conventions the codebase follows, and
how to propose changes.

## Code of conduct

Participation in this project is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By contributing,
you agree to abide by its terms.

## Development environment

`pypfda` targets Python ≥ 3.10. We recommend using a dedicated virtual
environment.

```bash
git clone https://github.com/bijanf/pypfda.git
cd pypfda
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
pre-commit install
```

## Running the test suite

```bash
pytest                       # full suite
pytest -k weights            # subset
pytest --cov                 # with coverage
mypy src/pypfda              # type check
ruff check src/ tests/       # lint
ruff format --check .        # format check
sphinx-build -W docs docs/_build  # docs
```

`pre-commit run --all-files` runs the same checks that CI runs.

## Conventions

- **Style**: enforced by `ruff format` (configured in `pyproject.toml`).
- **Linting**: `ruff check` with the rule selection in `pyproject.toml`.
- **Typing**: every public function has type hints; `mypy --strict` must
  pass.
- **Docstrings**: NumPy style, validated by `pydocstyle` rules in `ruff`.
- **Tests**: every public function gets at least one test; resampling and
  weight invariants are covered by `hypothesis` property tests.
- **Commits**: follow [Conventional Commits](https://www.conventionalcommits.org/).
  Examples: `feat(filter): add stratified resampling`,
  `fix(weights): correct underflow in log-sum-exp`,
  `docs(theory): expand SIR derivation`.

## Pull requests

1. Open an issue first for non-trivial changes so we can discuss the
   design.
2. Branch from `main`. Keep PRs focused; smaller PRs review faster.
3. Add or update tests, docs, and the `[Unreleased]` section of
   `CHANGELOG.md`.
4. CI must pass on all matrix cells before merge.
5. A maintainer will squash-merge with a Conventional Commit title.

## Reporting issues

Use the provided issue templates. Minimal reproducers (small Python
snippets, no external data) help us help you faster.

## Releasing

Releases are cut by maintainers via `git tag vX.Y.Z` on `main`. The
release workflow publishes to PyPI via OIDC trusted publishing and creates
a GitHub Release that triggers a Zenodo DOI.
