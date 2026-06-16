"""Online particle-filter cycle orchestration over a :class:`ForwardModel`.

:class:`CycleDriver` runs the analysis loop that turns the model-agnostic
:class:`~pypfda.ParticleFilter` into a full *online* reconstruction:

.. code-block:: text

    for each cycle:
        forecast all members one window        (via an ExecutionBackend)
        observe   -> predicted obs (n_members, n_obs)
        weight    -> NaN-aware Gaussian log-likelihood (+ optional max-weight cap)
        ESS       -> resample if below threshold
        if resampled:
            clone parent -> child states   (get_state / set_state)
            inflate every member           (distinct seed per member)
        record genealogy / EAS / target diagnostic; checkpoint

The driver is intentionally thin: the numerics are
:mod:`pypfda.weights`/:mod:`pypfda.resampling` (via the supplied
:class:`~pypfda.ParticleFilter`), the model is a
:class:`~pypfda.models.base.ForwardModel` adapter, and parallelism is an
:class:`ExecutionBackend`. This mirrors the two production pipelines that
motivated ``pypfda`` (a coupled GCM and an ocean model), which share exactly
this choreography.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pypfda.filter import ParticleFilter
from pypfda.models.base import ForwardModel

# A provider of the assimilated observations for a given cycle index:
# returns (y, obs_err) where y has shape (n_obs,) and obs_err is a scalar or
# an (n_obs,) array of standard deviations. For an OSSE this draws from the
# synthetic truth; for a real reconstruction it returns the proxy values.
ObsProvider = Callable[[int], "tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]] | float]"]

#: Run history: named lists, one entry appended per completed cycle.
History = dict[str, list[Any]]


# ---------------------------------------------------------------------------
# Execution backends — how the per-member forecast for one cycle is run.
# ---------------------------------------------------------------------------
class ExecutionBackend:
    """Runs the forecast of every member for one cycle.

    Subclass to add parallelism (threads, a process pool, or a SLURM job
    array). The contract: on return, every member in ``member_ids`` has been
    advanced by ``window`` and is ready for :meth:`ForwardModel.observe`.
    """

    def run_forecasts(self, model: ForwardModel, member_ids: Sequence[int], window: float) -> None:
        """Forecast every member in ``member_ids`` by ``window`` (override for parallelism)."""
        raise NotImplementedError


class SerialBackend(ExecutionBackend):
    """Forecast members one after another. Simplest; correct everywhere."""

    def run_forecasts(self, model: ForwardModel, member_ids: Sequence[int], window: float) -> None:
        """Forecast members sequentially."""
        for i in member_ids:
            model.forecast(i, window)


class ThreadPoolBackend(ExecutionBackend):
    """Forecast members concurrently with a bounded thread pool.

    Ideal for adapters whose :meth:`ForwardModel.forecast` blocks on an external
    process (a subprocess model run): the GIL is released during the blocking
    call, so ``max_workers`` members integrate at once. Size ``max_workers`` to
    ``cores // threads_per_member`` for a subprocess model. The first forecast to
    raise aborts the cycle (its exception propagates).
    """

    def __init__(self, max_workers: int = 8):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers

    def run_forecasts(self, model: ForwardModel, member_ids: Sequence[int], window: float) -> None:
        """Forecast every member concurrently, re-raising the first failure."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(model.forecast, i, window) for i in member_ids]
            for fut in futures:
                fut.result()  # propagate the first exception


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def effective_ancestor_size(genealogy: NDArray[np.integer[Any]], n_ancestors: int) -> float:
    r"""Effective number of surviving ancestors.

    .. math:: \mathrm{EAS} = 1 \big/ \sum_a f_a^2,

    where :math:`f_a` is the fraction of current members descended from
    original ancestor :math:`a`. Drops from ``n_members`` toward 1 as the
    ensemble's genealogy collapses onto a few lineages — the failure mode a
    regularized / inflated filter is designed to resist.
    """
    counts = np.bincount(np.asarray(genealogy, dtype=int), minlength=n_ancestors).astype(float)
    total = counts.sum()
    if total <= 0:
        return float(len(genealogy))
    fracs = counts / total
    s2 = float(np.sum(fracs**2))
    return 1.0 / s2 if s2 > 0 else float(len(genealogy))


def gaussian_loglik_nan(
    ensemble_obs: NDArray[np.floating[Any]],
    observations: NDArray[np.floating[Any]],
    obs_err: NDArray[np.floating[Any]] | float,
) -> NDArray[np.floating[Any]]:
    r"""NaN-aware diagonal-Gaussian log-likelihood per member.

    Like :func:`pypfda.weights.gaussian_log_likelihood` but sums only over
    proxies that are finite in *both* the prediction and the observation, so
    missing proxies (``NaN``) simply drop out — matching the production cost
    functions. A member with no valid proxies gets ``-inf`` (zero weight).

    Parameters
    ----------
    ensemble_obs : ndarray, shape (n_members, n_obs)
    observations : ndarray, shape (n_obs,)
    obs_err : float or ndarray, shape (n_obs,)
    """
    pred = np.asarray(ensemble_obs, dtype=float)
    obs = np.asarray(observations, dtype=float)
    sigma = np.broadcast_to(np.asarray(obs_err, dtype=float), obs.shape)
    if pred.ndim != 2 or obs.shape != (pred.shape[1],):
        raise ValueError(f"shape mismatch: ensemble_obs {pred.shape}, observations {obs.shape}")
    if np.any(sigma <= 0):
        raise ValueError("obs_err must be strictly positive")

    resid = (pred - obs) / sigma  # (n_members, n_obs)
    valid = np.isfinite(resid)
    resid = np.where(valid, resid, 0.0)
    n_valid = valid.sum(axis=1)
    loglik = -0.5 * np.sum(resid**2, axis=1)
    loglik = np.where(n_valid > 0, loglik, -np.inf)
    return loglik.astype(float)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
@dataclass
class CycleDriver:
    """Run an online particle-filter reconstruction over a forward model.

    Parameters
    ----------
    model : ForwardModel
        The adapter wrapping your forward model.
    pf : ParticleFilter
        Supplies the resampling scheme, ESS threshold, and optional
        max-weight cap. To run a **FREE** (no-assimilation) baseline, pass a
        ``ParticleFilter`` with ``ess_threshold`` effectively disabling
        resampling (e.g. a tiny value) — the loop then just forecasts the
        diverse ensemble.
    observations : callable
        ``observations(cycle) -> (y, obs_err)`` for each cycle index.
    n_cycles, window : int, float
        Number of assimilation cycles and the model-time length of each.
    inflation_amplitude : float, default 0
        Passed to :meth:`ForwardModel.inflate` after each resample.
    backend : ExecutionBackend, default SerialBackend
        How the per-cycle forecasts are executed.
    base_seed : int, default 2000
        Reproducible per-member inflation seeds derive from this.
    outdir : str, optional
        If given, the driver writes a small JSON checkpoint (genealogy +
        history + completed cycle) here each cycle and resumes from it.
    resume : bool, default True
        Resume from an existing checkpoint in ``outdir`` if present.
    """

    model: ForwardModel
    pf: ParticleFilter
    observations: ObsProvider
    n_cycles: int
    window: float
    inflation_amplitude: float = 0.0
    backend: ExecutionBackend = field(default_factory=SerialBackend)
    base_seed: int = 2000
    outdir: str | None = None
    resume: bool = True

    def _ckpt_path(self) -> Path | None:
        """Return the JSON checkpoint path, or ``None`` if no ``outdir`` was set."""
        return Path(self.outdir) / "pypfda_driver_state.json" if self.outdir else None

    def _maybe_resume(self, n: int) -> tuple[int, NDArray[np.integer[Any]], History]:
        """Return ``(start_cycle, genealogy, history)``, resuming if a checkpoint exists."""
        path = self._ckpt_path()
        if self.resume and path and path.exists():
            with path.open() as f:
                d = json.load(f)
            if d.get("n_members") == n:
                return (
                    int(d["completed_cycle"]) + 1,
                    np.asarray(d["genealogy"], dtype=int),
                    {k: list(v) for k, v in d["history"].items()},
                )
        genealogy = np.arange(n, dtype=int)
        history: History = {
            "cycle": [],
            "ess": [],
            "ess_fraction": [],
            "resampled": [],
            "eas": [],
            "mean_loglik": [],
            "targets": [],
            "parents": [],
        }
        return 0, genealogy, history

    def _checkpoint(
        self, cycle: int, genealogy: NDArray[np.integer[Any]], history: History
    ) -> None:
        """Atomically write the driver state (genealogy + history + cycle) to ``outdir``."""
        path = self._ckpt_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(
                {
                    "completed_cycle": int(cycle),
                    "n_members": int(self.model.n_members),
                    "genealogy": [int(x) for x in genealogy],
                    "history": history,
                },
                f,
            )
        tmp.replace(path)  # atomic

    def run(self) -> History:
        """Run all cycles and return the ``history`` dict."""
        n = self.model.n_members
        start, genealogy, history = self._maybe_resume(n)

        for cycle in range(start, self.n_cycles):
            # ---- FORECAST (all members one window) ----
            self.backend.run_forecasts(self.model, range(n), self.window)

            # ---- OBSERVE ----
            obs_pred = np.array(
                [np.asarray(self.model.observe(i, self.window), float) for i in range(n)]
            )  # (n_members, n_obs)
            y, obs_err = self.observations(cycle)

            # ---- WEIGHT + ESS + (maybe) RESAMPLE ----
            log_weights = gaussian_loglik_nan(obs_pred, y, obs_err)
            _, info = self.pf.assimilate_log_weights(np.arange(n), log_weights)

            # ---- RESAMPLE state + INFLATE ----
            if info.resampled:
                assert info.indices is not None  # guaranteed non-None when resampled
                parents = np.asarray(info.indices, dtype=int)
                # Snapshot every distinct parent BEFORE applying any clone so
                # swaps/cycles among members are safe (see get_state contract).
                snaps = {int(p): self.model.get_state(int(p)) for p in np.unique(parents)}
                for child, parent in enumerate(parents):
                    if int(parent) != child:
                        self.model.set_state(child, snaps[int(parent)])
                genealogy = genealogy[parents]
                for k in range(n):
                    self.model.inflate(
                        k,
                        self.inflation_amplitude,
                        seed=self.base_seed + 100_000 + cycle * 1000 + k,
                    )

            # ---- RECORD + CHECKPOINT ----
            eas = effective_ancestor_size(genealogy, n)
            finite = log_weights[np.isfinite(log_weights)]
            history["cycle"].append(int(cycle))
            history["ess"].append(float(info.ess))
            history["ess_fraction"].append(float(info.ess_fraction))
            history["resampled"].append(bool(info.resampled))
            history["eas"].append(float(eas))
            history["mean_loglik"].append(float(np.mean(finite)) if finite.size else float("nan"))
            history["targets"].append([float(self.model.target_diagnostic(i)) for i in range(n)])
            # Persist the per-cycle resample mapping (parent index chosen for each
            # child; identity when no resampling). With these, the ANALYSIS
            # (weighted/resampled) ensemble mean can be reconstructed offline from
            # the per-member forecast diagnostics WITHOUT re-running the model.
            history.setdefault("parents", []).append(
                [
                    int(p)
                    for p in (
                        info.indices if info.resampled and info.indices is not None else range(n)
                    )
                ]
            )
            self._checkpoint(cycle, genealogy, history)

        self.model.finalize()
        return history
