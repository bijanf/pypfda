"""The forward-model adapter contract.

``pypfda``'s :class:`~pypfda.ParticleFilter` is deliberately model-agnostic: it
sees only *predicted observations* and returns analysis weights plus resampling
*indices*. To run a full **online** reconstruction you also need four
model-specific operations:

1. **forecast** an ensemble member one assimilation window,
2. **observe** it — apply the observation operator :math:`H` at the proxy
   network to get predicted observations,
3. **get_state / set_state** — clone a resampled "parent" member's full state
   onto a "child" (this is how the filter's resampling indices act on a model
   that is too large to hold as a plain array), and
4. **inflate** — re-diversify the duplicated members.

Those operations are the adapter contract defined by :class:`ForwardModel`. The
cycle orchestration (weights → ESS → resample → clone → inflate → checkpoint)
lives in :class:`pypfda.driver.CycleDriver`; the numerics live in
:mod:`pypfda.weights` / :mod:`pypfda.resampling`.

The same contract is already satisfied, in spirit, by two independent
production pipelines that motivated ``pypfda``: a coupled GCM (file/restart and
SLURM-based) and an ocean model (in-process). Concrete reference adapters live
alongside this module in :mod:`pypfda.models`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Opaque, clonable handle to a member's full forward-model state. For
#: file-based models this is typically a path to a (copied) restart
#: directory/file; for in-process models it can be the state array(s)
#: themselves. The only requirement is the contract documented on
#: :meth:`ForwardModel.get_state`.
StateRef = Any


class ForwardModel(ABC):
    """Adapter that plugs a forward model into the particle-filter cycle.

    Ensemble members are addressed by integer id ``0 .. n_members - 1``.
    Implement the abstract methods for your model; everything else — weights,
    effective sample size, resampling, genealogy tracking, checkpointing,
    and (optionally parallel) execution — is provided by ``pypfda``.

    Subclasses must set :attr:`n_members` before the driver runs.

    Notes
    -----
    The driver treats members as opaque: it never inspects model state. It
    only (a) asks the execution backend to forecast members, (b) collects
    :meth:`observe` vectors, (c) on resample, snapshots each *parent* with
    :meth:`get_state` and applies it to the corresponding *child* with
    :meth:`set_state`, then (d) calls :meth:`inflate` on every member with a
    distinct seed.
    """

    #: Number of ensemble members. Subclasses must set this.
    n_members: int

    # -- ensemble setup ---------------------------------------------------
    @abstractmethod
    def initialize_member(self, member_id: int, ic_spec: Any) -> None:
        """Create / prepare member ``member_id`` from an initial-condition spec.

        ``ic_spec`` is opaque to ``pypfda`` — e.g. a restart-file path, a
        control-run year, or a perturbation seed. For a **diverse-IC**
        ensemble (the high-skill regime in the motivating study) pass a
        different ``ic_spec`` per member; for a **similar-IC** ensemble pass
        the same one. Callers must initialize the FREE and DA ensembles with
        *identical* args so the only difference is assimilation.
        """

    # -- forecast ---------------------------------------------------------
    @abstractmethod
    def forecast(self, member_id: int, window: float) -> None:
        """Advance member ``member_id`` forward by ``window`` model-time units.

        Must leave the member ready for :meth:`observe`, :meth:`get_state`,
        and a subsequent :meth:`forecast`. To de-alias the observation
        operator, accumulate a *window-mean* of the observed field here rather
        than using an end-of-window snapshot.
        """

    # -- observation operator --------------------------------------------
    @abstractmethod
    def observe(self, member_id: int, window: float) -> NDArray[np.floating]:
        """Return member ``member_id``'s predicted observations, shape ``(n_obs,)``.

        Apply the observation operator :math:`H` at the fixed proxy network
        (shared by truth and every member) to the just-integrated window.
        Missing proxies may be ``NaN``; pair them out in the likelihood
        (see the driver's ``log_likelihood`` hook). The ordering must be
        stable across members and cycles.
        """

    # -- state I/O for resampling ----------------------------------------
    @abstractmethod
    def get_state(self, member_id: int) -> StateRef:
        """Return an **independent, clonable** snapshot of the member's state.

        The returned handle must remain valid and unchanged even after the
        source member is subsequently overwritten by :meth:`set_state` — the
        driver snapshots all parents *before* applying any clone, so swaps and
        cycles among members are safe. For file-based models, copy the restart
        to a fresh location and return its path; for in-process models, return
        a deep copy of the state arrays.
        """

    @abstractmethod
    def set_state(self, member_id: int, state: StateRef) -> None:
        """Overwrite member ``member_id``'s full state with ``state``.

        ``state`` comes from :meth:`get_state` on a (resampled) parent and is
        guaranteed to be on the same model configuration.
        """

    # -- optional hooks ---------------------------------------------------
    def inflate(self, member_id: int, amplitude: float, seed: int) -> None:
        """Re-diversify a resampled member (default: no-op).

        Called on every member after a resample, with a **distinct** ``seed``
        per member so duplicated particles separate again. Override to add a
        smooth, dynamically-balanced perturbation to the member's state (e.g.
        density-compensated T/S in an ocean). ``amplitude == 0`` should leave
        the state untouched.
        """
        return

    def target_diagnostic(self, member_id: int) -> float:
        """Return an optional scalar to *evaluate* (never assimilated), e.g. AMOC strength.

        The driver records this per member per cycle so skill against a known
        truth can be scored offline. Default ``NaN`` (not tracked).
        """
        return float("nan")

    def finalize(self) -> None:
        """Run optional end-of-run cleanup (default: no-op)."""
        return
