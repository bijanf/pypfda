"""Particle weight computation and ensemble health diagnostics.

All weight math lives in the log domain to avoid underflow when many
particles have negligible likelihood. The public functions accept and
return float arrays of shape ``(n_members,)``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp


def gaussian_log_likelihood(
    ensemble_obs: ArrayLike,
    observations: ArrayLike,
    obs_err: ArrayLike | float,
) -> NDArray[np.floating]:
    r"""Log-likelihood of each ensemble member under a diagonal Gaussian observation model.

    For ensemble member :math:`m` with predicted observations
    :math:`\hat y_m \in \mathbb R^{p}` and actual observations
    :math:`y \in \mathbb R^{p}`, the log-likelihood under
    :math:`y \mid x \sim \mathcal N(\hat y_m, \mathrm{diag}(\sigma^2))`
    reduces (up to an additive constant that is the same for every
    member) to

    .. math::

        \log p(y \mid x_m) \;\propto\; -\frac{1}{2} \sum_{i=1}^{p}
        \left( \frac{\hat y_{m,i} - y_i}{\sigma_i} \right)^{2}.

    The constant is dropped because only weight *ratios* matter for the
    particle filter.

    Parameters
    ----------
    ensemble_obs : array_like, shape (n_members, n_obs)
        Predicted observations for each ensemble member.
    observations : array_like, shape (n_obs,)
        Actual observations.
    obs_err : float or array_like of shape (n_obs,)
        Observation-error standard deviation. Scalar values are
        broadcast over all observations.

    Returns
    -------
    log_likelihood : ndarray, shape (n_members,)
        Unnormalized log-likelihood per member.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``obs_err`` contains non-positive
        values.
    """
    pred = np.asarray(ensemble_obs, dtype=float)
    obs = np.asarray(observations, dtype=float)
    sigma = np.broadcast_to(np.asarray(obs_err, dtype=float), obs.shape)

    if pred.ndim != 2:
        raise ValueError(f"ensemble_obs must be 2-D (n_members, n_obs); got shape {pred.shape}")
    if obs.shape != (pred.shape[1],):
        raise ValueError(f"observations must have shape ({pred.shape[1]},); got {obs.shape}")
    if np.any(sigma <= 0):
        raise ValueError("obs_err must be strictly positive")

    residuals = (pred - obs) / sigma
    return cast("NDArray[np.floating]", -0.5 * np.sum(residuals**2, axis=1))


def normalize_log_weights(log_weights: ArrayLike) -> NDArray[np.floating]:
    """Normalize log-weights to a probability vector summing to one.

    Uses the log-sum-exp trick so the result is numerically stable even
    when the largest log-weight is very negative.

    Parameters
    ----------
    log_weights : array_like, shape (n_members,)
        Unnormalized log-weights.

    Returns
    -------
    weights : ndarray, shape (n_members,)
        Normalized weights, ``weights.sum() == 1`` to within floating-point
        rounding.
    """
    lw = np.asarray(log_weights, dtype=float)
    if lw.ndim != 1:
        raise ValueError(f"log_weights must be 1-D; got shape {lw.shape}")
    if not np.all(np.isfinite(lw)):
        raise ValueError("log_weights must be finite")
    return cast("NDArray[np.floating]", np.exp(lw - logsumexp(lw)))


def effective_sample_size(weights: ArrayLike) -> float:
    r"""Effective sample size of a normalized weight vector.

    .. math::

        N_{\mathrm{eff}} = \frac{1}{\sum_{m=1}^{N} w_m^{2}}.

    Ranges from 1 (all weight on one particle) to ``len(weights)`` (uniform).

    Parameters
    ----------
    weights : array_like, shape (n_members,)
        Normalized weights.

    Returns
    -------
    ess : float
        Effective sample size, in particles.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D; got shape {w.shape}")
    s = float(np.sum(w))
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"weights must be normalized (sum={s})")
    return 1.0 / float(np.sum(w**2))


def weight_entropy(weights: ArrayLike) -> float:
    r"""Shannon entropy (in nats) of a normalized weight vector.

    .. math::

        H(w) = -\sum_{m=1}^{N} w_m \log w_m.

    Zero-weight particles contribute zero (the convention :math:`0 \log 0 = 0`).

    Parameters
    ----------
    weights : array_like, shape (n_members,)
        Normalized weights.

    Returns
    -------
    entropy : float
        Entropy in nats. Maximum is ``log(N)`` for the uniform distribution.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D; got shape {w.shape}")
    nonzero = w[w > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def cap_max_weight(
    log_weights: ArrayLike,
    max_weight: float,
) -> NDArray[np.floating]:
    """Clip the largest particle weight to prevent degeneracy.

    Implements the simple "max-weight" heuristic from `DA_T9b`: if the
    largest normalized weight exceeds ``max_weight``, redistribute the
    excess uniformly to the others. Operates on log-weights and returns
    log-weights so it composes with downstream calls.

    Parameters
    ----------
    log_weights : array_like, shape (n_members,)
        Unnormalized log-weights.
    max_weight : float
        Maximum allowed normalized weight for any single particle, in
        ``(1/N, 1]``. Values outside this range are rejected.

    Returns
    -------
    capped_log_weights : ndarray, shape (n_members,)
        Log of the capped, re-normalized weight distribution.
    """
    lw = np.asarray(log_weights, dtype=float)
    if lw.ndim != 1:
        raise ValueError(f"log_weights must be 1-D; got shape {lw.shape}")
    n = lw.size
    if not (1.0 / n) < max_weight <= 1.0:
        raise ValueError(f"max_weight must be in (1/N, 1]; got {max_weight} for N={n}")

    w = normalize_log_weights(lw)
    if np.max(w) <= max_weight:
        return cast("NDArray[np.floating]", np.log(w))

    # Cap the top weight; spread the surplus across the remaining particles
    # in proportion to their current weights.
    top = np.argmax(w)
    surplus = float(w[top] - max_weight)
    rest_sum = float(w.sum() - w[top])
    if rest_sum <= 0:
        # All remaining mass is exactly zero (extreme degeneracy);
        # distribute surplus uniformly to keep the distribution proper.
        capped = np.full_like(w, surplus / (n - 1))
        capped[top] = max_weight
    else:
        capped = w * (1.0 + surplus / rest_sum)
        capped[top] = max_weight
    capped = np.clip(capped, a_min=1e-300, a_max=None)
    capped /= capped.sum()
    return cast("NDArray[np.floating]", np.log(capped))
