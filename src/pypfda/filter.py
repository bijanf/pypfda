"""High-level particle filter orchestration.

The :class:`ParticleFilter` class wires together the primitives in
:mod:`pypfda.weights` and :mod:`pypfda.resampling`. Its
:meth:`ParticleFilter.assimilate` method takes one analysis step
(weight + optional resampling) and returns the updated ensemble
together with diagnostic information.

The filter is intentionally model-agnostic: it does not run the forward
model itself. Callers are expected to integrate their own model between
analysis steps and pass the resulting state and predicted observations
back in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pypfda.resampling import resample
from pypfda.weights import (
    cap_max_weight,
    effective_sample_size,
    gaussian_log_likelihood,
    normalize_log_weights,
    weight_entropy,
)

VALID_RESAMPLING: Final[frozenset[str]] = frozenset(
    {"systematic", "stratified", "residual", "multinomial"}
)


@dataclass(frozen=True)
class AssimilationInfo:
    """Diagnostics returned by :meth:`ParticleFilter.assimilate`.

    Attributes
    ----------
    weights : ndarray, shape (n_members,)
        Normalized analysis weights.
    log_weights : ndarray, shape (n_members,)
        Unnormalized log-weights, useful for downstream log-domain ops.
    ess : float
        Effective sample size in particles.
    ess_fraction : float
        ``ess / n_members``, in :math:`[0, 1]`.
    entropy : float
        Shannon entropy of the weight distribution in nats.
    resampled : bool
        Whether the ensemble was resampled this step.
    indices : ndarray of int or None, shape (n_members,)
        Resampling indices, or ``None`` if no resampling occurred.
    """

    weights: NDArray[np.floating]
    log_weights: NDArray[np.floating]
    ess: float
    ess_fraction: float
    entropy: float
    resampled: bool
    indices: NDArray[np.intp] | None


class ParticleFilter:
    """Sequential importance resampling (SIR) particle filter.

    Parameters
    ----------
    ess_threshold : float, default 0.5
        Resample whenever the effective sample size fraction drops below
        this value. Must lie in :math:`(0, 1]`. Setting it to 1 forces
        resampling at every step; setting it close to 0 disables
        resampling almost entirely.
    resampling : {'systematic', 'stratified', 'residual', 'multinomial'}, default 'systematic'
        Resampling scheme to use when triggered.
    max_weight : float, optional
        If given, the largest normalized weight is capped at this value
        and the surplus is redistributed among the remaining particles.
        Useful as a degeneracy-prevention safeguard. Must lie strictly
        above ``1 / n_members`` and not exceed 1.
    rng : numpy.random.Generator, optional
        Random number generator used for resampling. If omitted, a
        default generator is created and reused across calls.

    Notes
    -----
    The current implementation assumes a diagonal Gaussian observation
    error model; richer likelihoods can be obtained by passing
    pre-computed log-weights to :meth:`assimilate_log_weights` instead.
    """

    def __init__(
        self,
        ess_threshold: float = 0.5,
        resampling: str = "systematic",
        max_weight: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 < ess_threshold <= 1.0:
            raise ValueError(f"ess_threshold must be in (0, 1]; got {ess_threshold}")
        if resampling not in VALID_RESAMPLING:
            raise ValueError(
                f"Unknown resampling {resampling!r}; choose from {sorted(VALID_RESAMPLING)}"
            )
        if max_weight is not None and not 0.0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1]; got {max_weight}")

        self.ess_threshold = float(ess_threshold)
        self.resampling = resampling
        self.max_weight = max_weight
        self.rng = rng if rng is not None else np.random.default_rng()

    def assimilate(
        self,
        ensemble: ArrayLike,
        ensemble_obs: ArrayLike,
        observations: ArrayLike,
        obs_err: ArrayLike | float,
    ) -> tuple[NDArray[np.floating], AssimilationInfo]:
        """Run one analysis (and possibly resampling) step.

        Parameters
        ----------
        ensemble : array_like, shape (n_members, ...)
            Ensemble state. Any trailing dimensions are preserved when
            resampling.
        ensemble_obs : array_like, shape (n_members, n_obs)
            Predicted observations for each ensemble member.
        observations : array_like, shape (n_obs,)
            Actual observations.
        obs_err : float or array_like of shape (n_obs,)
            Observation-error standard deviations.

        Returns
        -------
        ensemble_out : ndarray, shape (n_members, ...)
            Resampled ensemble (or the input unchanged if no resampling
            occurred).
        info : AssimilationInfo
            Diagnostics for this step.
        """
        log_weights = gaussian_log_likelihood(ensemble_obs, observations, obs_err)
        return self.assimilate_log_weights(ensemble, log_weights)

    def assimilate_log_weights(
        self,
        ensemble: ArrayLike,
        log_weights: ArrayLike,
    ) -> tuple[NDArray[np.floating], AssimilationInfo]:
        """Like :meth:`assimilate` but takes externally-computed log-weights.

        Use this when you have a non-Gaussian likelihood, a custom proxy
        forward model, or pre-localized weights that do not factor as a
        single Gaussian product.

        Parameters
        ----------
        ensemble : array_like, shape (n_members, ...)
        log_weights : array_like, shape (n_members,)
            Unnormalized log-weights.

        Returns
        -------
        ensemble_out : ndarray, shape (n_members, ...)
        info : AssimilationInfo
        """
        ens = np.asarray(ensemble)
        lw = np.asarray(log_weights, dtype=float)
        if lw.ndim != 1 or lw.size != ens.shape[0]:
            raise ValueError(
                f"log_weights shape {lw.shape} incompatible with ensemble shape {ens.shape}"
            )

        if self.max_weight is not None:
            lw = cap_max_weight(lw, self.max_weight)

        weights = normalize_log_weights(lw)
        ess = effective_sample_size(weights)
        ess_fraction = ess / weights.size

        if ess_fraction < self.ess_threshold:
            indices = resample(weights, method=self.resampling, rng=self.rng)
            ensemble_out = ens[indices]
            resampled = True
        else:
            indices = None
            ensemble_out = ens
            resampled = False

        info = AssimilationInfo(
            weights=weights,
            log_weights=lw,
            ess=ess,
            ess_fraction=ess_fraction,
            entropy=weight_entropy(weights),
            resampled=resampled,
            indices=indices,
        )
        return ensemble_out, info
