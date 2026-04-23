"""Spatial localization for ensemble data assimilation.

Localization mitigates the spurious long-range correlations that arise
when ensemble size is small relative to the state-space dimension. The
canonical choice in geoscience DA is the fifth-order piecewise rational
function of Gaspari & Cohn (1999), which is compactly supported,
smooth, and a valid covariance kernel.

References
----------
Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation
functions in two and three dimensions*. Quarterly Journal of the Royal
Meteorological Society, 125, 723–757.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_KM: float = 6371.0
"""Mean radius of Earth used by :func:`haversine_distance`, in kilometres."""


def haversine_distance(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
) -> NDArray[np.floating]:
    """Great-circle distance in kilometres between points on the sphere.

    Inputs are in degrees and may be scalars or broadcastable arrays.
    Result is in the same shape as the broadcast of the inputs.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array_like
        Latitudes and longitudes in degrees.

    Returns
    -------
    distance : ndarray
        Great-circle distance in kilometres.
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.asarray(EARTH_RADIUS_KM * c, dtype=float)


def gaspari_cohn(
    distance: ArrayLike,
    localization_radius: float,
) -> NDArray[np.floating]:
    r"""Gaspari–Cohn fifth-order piecewise rational localization function.

    The function is :math:`1` at zero distance, smoothly decays to
    :math:`0` at twice the localization radius, and is exactly zero
    beyond that. The half-width at half-maximum is approximately equal
    to the localization radius.

    Parameters
    ----------
    distance : array_like
        Distance from the observation. Same units as
        ``localization_radius``.
    localization_radius : float
        Localization radius. Must be positive.

    Returns
    -------
    weight : ndarray
        Localization weight in :math:`[0, 1]`, same shape as ``distance``.
    """
    if localization_radius <= 0:
        raise ValueError(
            f"localization_radius must be positive; got {localization_radius}"
        )

    r = np.abs(np.asarray(distance, dtype=float)) / localization_radius
    weight = np.zeros_like(r)

    # Region 1: 0 <= r < 1
    m1 = r < 1.0
    r1 = r[m1]
    weight[m1] = (
        1.0 - (5.0 / 3.0) * r1**2 + (5.0 / 8.0) * r1**3 + 0.5 * r1**4 - 0.25 * r1**5
    )

    # Region 2: 1 <= r < 2
    m2 = (r >= 1.0) & (r < 2.0)
    r2 = r[m2]
    weight[m2] = (
        4.0
        - 5.0 * r2
        + (5.0 / 3.0) * r2**2
        + (5.0 / 8.0) * r2**3
        - 0.5 * r2**4
        + (1.0 / 12.0) * r2**5
        - 2.0 / (3.0 * r2)
    )

    # Region 3: r >= 2 stays at zero.
    return weight


def pairwise_distance_matrix(
    lats: ArrayLike,
    lons: ArrayLike,
) -> NDArray[np.floating]:
    """Symmetric pairwise great-circle distance matrix.

    Parameters
    ----------
    lats, lons : array_like, shape (n,)
        Latitudes and longitudes in degrees.

    Returns
    -------
    D : ndarray, shape (n, n)
        ``D[i, j]`` is the distance in kilometres between point ``i``
        and point ``j``. Diagonal is zero.
    """
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    if lats_arr.shape != lons_arr.shape or lats_arr.ndim != 1:
        raise ValueError("lats and lons must be 1-D arrays of the same length")
    return haversine_distance(
        lats_arr[:, None], lons_arr[:, None], lats_arr[None, :], lons_arr[None, :]
    )
