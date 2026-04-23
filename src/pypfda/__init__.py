"""Particle filter data assimilation in pure Python.

The public API is intentionally small. Most users only need
:class:`ParticleFilter` and :class:`AssimilationInfo` from the top-level
package; advanced users can reach into :mod:`pypfda.weights`,
:mod:`pypfda.resampling`, and :mod:`pypfda.localization` for the
underlying primitives.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pypfda.filter import AssimilationInfo, ParticleFilter

try:
    __version__ = version("pypfda")
except PackageNotFoundError:  # pragma: no cover - editable install before build
    __version__ = "0.0.0+unknown"

__all__ = [
    "AssimilationInfo",
    "ParticleFilter",
    "__version__",
]
