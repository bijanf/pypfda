"""Regression tests for the CLIMBER-X stochastic-hosing forcing file.

The adapter advances the model calendar across segments (``year_ini = year_ini_start
+ elapsed``). The hosing file is rewritten every cycle and must therefore be written
on the *segment's own* calendar: a 0-based time axis would leave every segment after
the first entirely outside the file, and the model would not find its forcing.
"""

from __future__ import annotations

import numpy as np
import pytest

# netCDF4 must be imported before the numpy-using package below: some builds of its
# C extension emit a numpy ABI RuntimeWarning if numpy is already fully initialised,
# and this suite runs with filterwarnings=error.
nc4 = pytest.importorskip("netCDF4")

from pypfda.models.climberx import write_hosing_file  # noqa: E402


def _read(path):
    with nc4.Dataset(path) as ds:
        return np.asarray(ds.variables["time"][:]), np.asarray(ds.variables["fwf"][:])


def test_time_axis_starts_at_year0(tmp_path):
    """The axis must cover year0 .. year0 + years - 1, not 0 .. years - 1."""
    years, year0 = 7, 1305
    path = tmp_path / "hosing.nc"
    write_hosing_file(path, years, 0.1, 10.0, seed=1, year0=year0)
    t, _ = _read(path)
    assert t[0] == pytest.approx(year0)
    assert t[-1] == pytest.approx(year0 + years - 1)
    np.testing.assert_allclose(np.diff(t), 1.0)


def test_default_year0_is_backward_compatible(tmp_path):
    """Omitting year0 reproduces the pre-calendar (v1.0) 0-based axis exactly."""
    p = tmp_path / "h.nc"
    write_hosing_file(p, 5, 0.1, 10.0, seed=2)
    t, _ = _read(p)
    np.testing.assert_allclose(t, np.arange(5, dtype=float))


def test_year0_does_not_perturb_the_series(tmp_path):
    """Shifting the calendar must move the axis only, never the forcing values."""
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    write_hosing_file(a, 6, 0.2, 8.0, seed=3, year0=0)
    write_hosing_file(b, 6, 0.2, 8.0, seed=3, year0=-1000)
    _, fa = _read(a)
    _, fb = _read(b)
    np.testing.assert_allclose(fa, fb)


def test_series_is_zero_mean(tmp_path):
    """Zero-mean is what keeps the hosing from imposing a net freshwater trend."""
    p = tmp_path / "h.nc"
    write_hosing_file(p, 64, 0.3, 12.0, seed=4, year0=250)
    _, fwf = _read(p)
    assert abs(float(fwf.mean())) < 1e-12
