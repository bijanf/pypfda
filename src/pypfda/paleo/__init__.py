"""Paleoclimate utilities.

Contains forward proxy system models (e.g. coral δ¹⁸O), loaders for common proxy
databases, and OSSE pseudo-proxy network builders. Available with
``pip install 'pypfda[paleo]'``.
"""

from __future__ import annotations

from pypfda.paleo.proxy_network import (
    DEFAULT_MARINE_SST_SITES,
    ProxyNetwork,
    build_proxy_index,
)

__all__ = [
    "DEFAULT_MARINE_SST_SITES",
    "ProxyNetwork",
    "build_proxy_index",
]
