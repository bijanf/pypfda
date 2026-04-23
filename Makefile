.PHONY: help install dev lint format test cov type docs clean

help:
	@echo "Common targets:"
	@echo "  install   pip-install the package (no extras)"
	@echo "  dev       pip-install with dev + docs + all extras"
	@echo "  lint      run ruff check"
	@echo "  format    run ruff format"
	@echo "  type      run mypy"
	@echo "  test      run pytest"
	@echo "  cov       run pytest with coverage"
	@echo "  docs      build Sphinx HTML docs"
	@echo "  clean     remove build artifacts"

install:
	pip install -e .

dev:
	pip install -e '.[all]'
	pre-commit install

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/ examples/ reproduce/

type:
	mypy src/pypfda

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing --cov-report=xml

docs:
	sphinx-build -W -b html docs docs/_build/html

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} +
