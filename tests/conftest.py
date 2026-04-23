"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(20260423)


@pytest.fixture
def uniform_weights() -> np.ndarray:
    return np.full(100, 1.0 / 100)


@pytest.fixture
def degenerate_weights() -> np.ndarray:
    w = np.zeros(100)
    w[42] = 1.0
    return w


@pytest.fixture
def skewed_weights(rng: np.random.Generator) -> np.ndarray:
    raw = rng.dirichlet(np.full(100, 0.1))  # heavy concentration on a few members
    return raw / raw.sum()
