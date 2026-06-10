"""Readers for PlaSim-LSG ocean output (srv "service format") and AMOC diagnostics.

The LSG ocean of the PlaSim-LSG coupled model writes its gridded fields to a
Fortran sequential-unformatted file (``lsg_output``, opened on unit 93 in
``lsgmod.f90`` subroutine ``outpostsrv``). The on-disk layout is the classic
"srv" / "service" format: each 2-D field is a pair of records,

1. an **8-int32 header** ``[code, level, idate, itime, nlon, nlat, dummy, dummy]``
2. a **data record** of ``nlon * nlat`` 4-byte reals,

with every record bracketed by the usual 4-byte Fortran length markers. The
binary is produced by ifort on x86, so everything is little-endian.

The LSG grid is ``ien=72`` (zonal) by ``jen=76`` (meridional) with ``ken=22``
levels. ``outpostsrv`` writes, in order: land/sea mask (code -40, all levels),
vector-point mask (-41), **potential temperature** (-2, all levels, in Kelvin
= ``t + tkelvin``), salinity (-5), velocities, and many surface diagnostics.
*Sea-surface temperature is the level-1 record of the potential-temperature
field* (code -2, level 1). ``lsg_output`` accumulates one such block per ocean
output step within a run segment, so the **last** matching record is the most
recent ocean state; that is what :func:`read_lsg_surface_temp` returns.

The AMOC index is *not* in the srv file: it is parsed from the ASCII
``plasim_diag`` (LSG diagnostics). Each ``ATL max (NADW)`` row holds the
Atlantic overturning maximum below 700 m at three latitude bands, in the column
order ``66-46N  16-44N  30S`` (see the header line written just above it in
``outdiaglsg``). The reconstruction target is the **16-44N band = the middle
(second) value**; :func:`read_lsg_amoc_from_diag` returns that value from the
last such row in the file. Units: Sv.

This module is dependency-light (numpy only) so it is testable without the model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

#: LSG grid dimensions (lsgmod.f90: ien, jen, ken).
IEN = 72  # zonal points (longitude)
JEN = 76  # meridional points (latitude)
KEN = 22  # ocean levels

#: srv field codes (lsgmod.f90 subroutine outpostsrv).
CODE_LSMASK = -40  # land/sea mask at scalar points (wet=1, land=0)
CODE_POTTEMP = -2  # potential temperature (Kelvin = t + tkelvin)
SURFACE_LEVEL = 1  # level index of the surface record

#: Temperature offset applied by LSG (lsgmod.f90: "Values changed according to
#: PlaSim" -> tkelvin = 273.16). Subtract to get degrees Celsius.
TKELVIN = 273.16

#: Marker text and column index of the reconstruction-target AMOC band in
#: plasim_diag. The data line is e.g.
#:     ATL max (NADW)   :      15.78822      16.72152      13.04996
#: with the three numbers being the 66-46N, 16-44N and 30S bands respectively.
#: The 16-44N target is the SECOND number (whitespace token index 5, 0-based).
AMOC_MARKER = "ATL max (NADW)"
AMOC_BAND_TOKEN = 5  # 0-based index into the line's whitespace tokens (16-44N)


# ---------------------------------------------------------------------------
# srv (service-format) record reader
# ---------------------------------------------------------------------------
def _iter_srv_fields(
    path: Path, nlon: int = IEN, nlat: int = JEN
) -> Iterator[tuple[int, int, int, NDArray[np.floating[Any]]]]:
    """Yield ``(code, level, idate, field)`` for every 2-D record in an srv file.

    ``field`` is a ``(nlat, nlon)`` float64 array (the Fortran data record is
    written row-major as ``yhelp(ien, jen)`` -> ``nlon`` fastest, so reshaping
    the flat buffer to ``(nlat, nlon)`` puts latitude on axis 0). Endianness is
    little-endian (ifort/x86). Records whose payload size does not match either
    an 8-int32 header or an ``nlon*nlat`` real data block are skipped.
    """
    raw = Path(path).read_bytes()
    header_bytes = 8 * 4
    data_bytes = nlon * nlat * 4
    off = 0
    n = len(raw)
    pending_hdr: tuple[int, int, int] | None = None
    while off + 4 <= n:
        m1 = int(np.frombuffer(raw, "<i4", 1, off)[0])
        if m1 <= 0 or off + 8 + m1 > n:
            break
        body_off = off + 4
        m2 = int(np.frombuffer(raw, "<i4", 1, body_off + m1)[0])
        if m1 != m2:  # corrupt / not a sequential-unformatted boundary
            break
        if m1 == header_bytes:
            h = np.frombuffer(raw, "<i4", 8, body_off)
            pending_hdr = (int(h[0]), int(h[1]), int(h[2]))
        elif m1 == data_bytes and pending_hdr is not None:
            data = np.frombuffer(raw, "<f4", nlon * nlat, body_off)
            field = data.astype(np.float64).reshape(nlat, nlon)
            code, level, idate = pending_hdr
            pending_hdr = None
            yield code, level, idate, field
        off += 8 + m1


def _last_field(path: Path, code: int, level: int) -> NDArray[np.floating[Any]] | None:
    """Return the last ``(nlat, nlon)`` field matching ``code`` and ``level``."""
    out: NDArray[np.floating[Any]] | None = None
    for c, lev, _idate, field in _iter_srv_fields(path):
        if c == code and lev == level:
            out = field
    return out


# ---------------------------------------------------------------------------
# Grid coordinates
# ---------------------------------------------------------------------------
def lsg_grid(grad: float = 5.0) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return nominal ``(lon, lat)`` 1-D cell-centre arrays of the LSG grid.

    LSG runs on a rotated, staggered E-grid: latitudes follow
    ``lat(j) = (grad/2) * (jen/2 - j + 0.5)`` (lsgmod.f90 subroutine ``inigr``),
    giving 76 rows from +93.75 to -93.75 in 2.5 deg steps for the default
    ``grad = 5.0`` (the rows beyond +/-90 are the grid's polar-overlap rows).
    The exact longitudes are shifted row-by-row by half a box (the E-grid
    rotation); for proxy snapping we use the un-rotated nominal cell-centre
    longitudes ``lon(i) = (i-1) * grad``, which agree with the true cell centres
    to within half a grid box (1.25 deg) -- far inside the proxy snap radius.
    Use :func:`lsg_grid_from_restart` for the exact per-row coordinates.
    """
    half = grad / 2.0
    lat = half * (JEN / 2.0 - np.arange(1, JEN + 1) + 0.5)
    lon = grad * np.arange(IEN)
    return lon.astype(float), lat.astype(float)


def lsg_grid_from_restart(
    restart_path: Path,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return exact ``(lon, lat)`` 1-D arrays read from an LSG restart header.

    An LSG restart (``kleiauf`` / ``kleiin1``) is Fortran sequential-unformatted;
    its second record is the 512-element real(kind=8) ``oddr`` header. Per
    ``lsgmod.f90``: ``oddr(50)`` is the zonal grid step (deg), ``oddr(52-1+j)``
    is the latitude of row ``j`` (1-based), and ``oddr(52-1+jen+j)`` is the
    per-row longitude origin. We return the latitude row vector and the
    longitude of the first interior row's columns (``origin + (i-1)*oddr(50)``).
    """
    raw = Path(restart_path).read_bytes()
    # walk to the 2nd record (record 0 = nddr int header, record 1 = oddr real*8)
    off = 0
    recs: list[bytes] = []
    while off + 4 <= len(raw) and len(recs) < 2:
        m1 = int(np.frombuffer(raw, "<i4", 1, off)[0])
        recs.append(raw[off + 4 : off + 4 + m1])
        off += 8 + m1
    oddr = np.frombuffer(recs[1], "<f8")
    dlam = float(oddr[49])  # oddr(50)
    lat = oddr[51 : 51 + JEN].astype(float)  # oddr(52..52+jen-1)
    # representative interior-row longitude origin (row 3, 1-based, is regular)
    origin = float(oddr[51 + JEN + 2])
    lon = (origin + dlam * np.arange(IEN)) % 360.0
    return lon, lat


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------
def read_lsg_surface_temp(
    path: Path,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return the last surface SST field and grid from an LSG ``lsg_output`` file.

    Reads the most recent surface potential-temperature record (code -2, level
    1), converts Kelvin to degrees Celsius, and masks land using the most recent
    land/sea mask record (code -40, level 1). Land cells are returned as
    ``NaN``.

    Returns
    -------
    sst : ndarray, shape ``(nlat, nlon) = (76, 72)``
        Sea-surface temperature in degrees Celsius, land masked to ``NaN``.
    lon : ndarray, shape ``(nlon,)``
        Nominal cell-centre longitudes (deg, in ``[0, 360)``).
    lat : ndarray, shape ``(nlat,)``
        Cell-centre latitudes (deg, +north).
    """
    path = Path(path)
    pottemp = _last_field(path, CODE_POTTEMP, SURFACE_LEVEL)
    if pottemp is None:
        raise ValueError(
            f"no surface potential-temperature record (code {CODE_POTTEMP}, "
            f"level {SURFACE_LEVEL}) found in {path}"
        )
    sst = pottemp - TKELVIN
    mask = _last_field(path, CODE_LSMASK, SURFACE_LEVEL)
    if mask is not None:
        sst = np.where(mask > 0.5, sst, np.nan)
    else:  # fall back to a physical-range mask if no land/sea record present
        sst = np.where((sst > -3.0) & (sst < 40.0), sst, np.nan)
    lon, lat = lsg_grid()
    return sst, lon, lat


def read_lsg_amoc_from_diag(plasim_diag_path: Path) -> float:
    """Return the 16-44N Atlantic overturning (Sv) from the last diag record.

    Parses the ASCII ``plasim_diag``: the last ``ATL max (NADW)`` row holds the
    overturning maxima at ``66-46N  16-44N  30S``; the 16-44N target is the
    middle (second) number. Returns ``nan`` if no such row is present.
    """
    last_value = float("nan")
    with Path(plasim_diag_path).open("r", errors="replace") as fh:
        for line in fh:
            if AMOC_MARKER in line:
                tokens = line.split()
                if len(tokens) > AMOC_BAND_TOKEN:
                    try:
                        last_value = float(tokens[AMOC_BAND_TOKEN])
                    except ValueError:
                        continue
    return last_value


def wet_mask_lsg(path: Path) -> NDArray[np.bool_]:
    """Return the ``(nlat, nlon)`` ocean mask (``True`` = wet) from ``lsg_output``."""
    sst, _lon, _lat = read_lsg_surface_temp(path)
    return cast("NDArray[np.bool_]", ~np.isnan(sst))
