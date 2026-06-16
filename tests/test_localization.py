"""Tests for pypfda.localization."""

from __future__ import annotations

import numpy as np
import pytest

from pypfda.localization import (
    EARTH_RADIUS_KM,
    gaspari_cohn,
    haversine_distance,
    pairwise_distance_matrix,
)


class TestHaversine:
    def test_zero_distance(self) -> None:
        d = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert d.item() == pytest.approx(0.0, abs=1e-6)

    def test_antipode_is_pi_r(self) -> None:
        d = haversine_distance(0.0, 0.0, 0.0, 180.0)
        assert d.item() == pytest.approx(np.pi * EARTH_RADIUS_KM, rel=1e-4)

    def test_known_route_ny_to_london(self) -> None:
        d = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert d.item() == pytest.approx(5570.0, abs=20.0)

    def test_broadcasts(self) -> None:
        lats1 = np.array([0.0, 0.0])
        lons1 = np.array([0.0, 0.0])
        lats2 = np.array([0.0, 90.0])
        lons2 = np.array([90.0, 0.0])
        d = haversine_distance(lats1, lons1, lats2, lons2)
        assert d.shape == (2,)
        # Both points should be 1/4 of Earth's circumference away.
        np.testing.assert_allclose(d, np.full(2, np.pi / 2 * EARTH_RADIUS_KM), rtol=1e-4)


class TestGaspariCohn:
    def test_one_at_zero(self) -> None:
        assert gaspari_cohn(np.array([0.0]), 1000.0).item() == pytest.approx(1.0)

    def test_zero_beyond_two_radii(self) -> None:
        d = np.array([2000.0, 5000.0, 1e6])
        assert np.allclose(gaspari_cohn(d, 1000.0), 0.0)

    def test_monotone_decreasing_inside_support(self) -> None:
        d = np.linspace(0.0, 2000.0, 200)
        w = gaspari_cohn(d, 1000.0)
        diffs = np.diff(w)
        assert np.all(diffs <= 1e-12)

    def test_in_unit_interval(self) -> None:
        d = np.linspace(0.0, 5000.0, 500)
        w = gaspari_cohn(d, 1000.0)
        assert np.all(w >= 0)
        assert np.all(w <= 1.0 + 1e-12)

    def test_continuous_at_radius_boundary(self) -> None:
        eps = 1e-9
        below = gaspari_cohn(np.array([1000.0 - eps]), 1000.0)
        above = gaspari_cohn(np.array([1000.0 + eps]), 1000.0)
        assert np.abs(below - above).item() < 1e-7

    def test_continuous_at_two_radii_boundary(self) -> None:
        eps = 1e-9
        below = gaspari_cohn(np.array([2000.0 - eps]), 1000.0)
        above = gaspari_cohn(np.array([2000.0 + eps]), 1000.0)
        assert np.abs(below - above).item() < 1e-7

    def test_negative_radius_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            gaspari_cohn(np.array([100.0]), -1.0)


class TestPairwiseDistance:
    def test_diagonal_is_zero(self) -> None:
        lats = np.array([0.0, 30.0, -30.0])
        lons = np.array([0.0, 90.0, -90.0])
        dist_mat = pairwise_distance_matrix(lats, lons)
        assert np.all(np.diag(dist_mat) == 0.0)

    def test_symmetric(self) -> None:
        lats = np.array([10.0, -45.0, 75.0])
        lons = np.array([20.0, 100.0, -10.0])
        dist_mat = pairwise_distance_matrix(lats, lons)
        np.testing.assert_allclose(dist_mat, dist_mat.T, rtol=1e-12)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            pairwise_distance_matrix(np.zeros(3), np.zeros(4))
