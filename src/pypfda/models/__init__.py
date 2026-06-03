"""Forward-model adapters for online particle-filter reconstruction.

The particle filter in :mod:`pypfda.filter` is model-agnostic. To run a full
*online* reconstruction you wrap your forward model in a
:class:`~pypfda.models.base.ForwardModel` adapter and hand it, together with a
:class:`~pypfda.ParticleFilter`, to :class:`pypfda.driver.CycleDriver`.

Reference adapters (a coupled GCM, an EMIC) are added under this subpackage; the
abstract contract lives in :mod:`pypfda.models.base`.
"""

from __future__ import annotations

from pypfda.models.base import ForwardModel, StateRef

__all__ = ["ForwardModel", "StateRef"]
