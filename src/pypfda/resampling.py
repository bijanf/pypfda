"""Particle resampling schemes.

Each scheme takes a vector of normalized weights and a random generator
and returns an integer index array of the same length whose values are
in ``[0, n_members)``. The expected number of times particle :math:`i`
is selected equals :math:`N w_i`. Variance differs across schemes;
systematic resampling has the lowest variance and is the recommended
default for most applications.

References
----------
Douc, R., Cappé, O. & Moulines, E. (2005).
*Comparison of resampling schemes for particle filtering*.
ISPA 2005.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _check_weights(weights: ArrayLike) -> NDArray[np.floating]:
    """Validate and return weights as a float array."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D; got shape {w.shape}")
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = float(w.sum())
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"weights must be normalized (sum={s})")
    return w


def systematic(
    weights: ArrayLike,
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Systematic resampling.

    Draws a single uniform random offset and walks through the inverse
    cumulative distribution function with deterministic, evenly-spaced
    strides. Lowest variance among standard schemes; preserves the
    expected count exactly when weights are rationals with a common
    denominator that divides ``N``.

    Parameters
    ----------
    weights : array_like, shape (n_members,)
        Normalized weights.
    rng : numpy.random.Generator, optional
        Random number generator. If ``None``, ``np.random.default_rng()``
        is used.

    Returns
    -------
    indices : ndarray of int, shape (n_members,)
        Indices into the input weight vector.
    """
    w = _check_weights(weights)
    n = w.size
    rng = rng if rng is not None else np.random.default_rng()
    u0 = rng.uniform(0.0, 1.0 / n)
    points = u0 + np.arange(n) / n
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0  # guard against floating-point drift
    return np.searchsorted(cumulative, points).astype(np.intp)


def stratified(
    weights: ArrayLike,
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Stratified resampling.

    Like :func:`systematic` but draws an independent uniform offset
    inside each of the ``N`` equal-width strata. Slightly higher
    variance than systematic, but eliminates the (theoretical) failure
    mode where systematic resampling can be sensitive to particle
    ordering.
    """
    w = _check_weights(weights)
    n = w.size
    rng = rng if rng is not None else np.random.default_rng()
    points = (rng.uniform(0.0, 1.0, size=n) + np.arange(n)) / n
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, points).astype(np.intp)


def residual(
    weights: ArrayLike,
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Residual resampling.

    Allocates ``floor(N * w_i)`` deterministic copies of each particle
    and resamples the remainder multinomially with normalized residual
    weights. Often the lowest variance for small ensembles where many
    weights are close to the uniform value.
    """
    w = _check_weights(weights)
    n = w.size
    rng = rng if rng is not None else np.random.default_rng()

    deterministic_counts = np.floor(n * w).astype(np.intp)
    remaining = n - int(deterministic_counts.sum())
    indices = np.repeat(np.arange(n, dtype=np.intp), deterministic_counts)

    if remaining > 0:
        residual_weights = n * w - deterministic_counts
        residual_weights = residual_weights / residual_weights.sum()
        extras = rng.choice(n, size=remaining, p=residual_weights)
        indices = np.concatenate([indices, extras.astype(np.intp)])

    rng.shuffle(indices)
    return indices


def multinomial(
    weights: ArrayLike,
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Plain multinomial resampling.

    Draws ``N`` independent samples from the categorical distribution
    defined by the weights. Highest variance of the standard schemes;
    included mainly as a baseline.
    """
    w = _check_weights(weights)
    n = w.size
    rng = rng if rng is not None else np.random.default_rng()
    return rng.choice(n, size=n, p=w).astype(np.intp)


_SCHEMES: dict[str, Callable[..., NDArray[np.intp]]] = {
    "systematic": systematic,
    "stratified": stratified,
    "residual": residual,
    "multinomial": multinomial,
}


def resample(
    weights: ArrayLike,
    method: str = "systematic",
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Dispatch by name to one of the resampling routines.

    Parameters
    ----------
    weights : array_like, shape (n_members,)
        Normalized weights.
    method : {'systematic', 'stratified', 'residual', 'multinomial'}, default 'systematic'
        Resampling scheme.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    indices : ndarray of int, shape (n_members,)
    """
    try:
        fn = _SCHEMES[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown resampling method {method!r}; choose from {sorted(_SCHEMES)}"
        ) from exc
    return fn(weights, rng=rng)
