"""Tests for physics/thermodynamics.py, focused on the geometric
offset-along-normal construction used for layer boundaries (the bow
shock is not spherically symmetric, so a layer thickness measured
along the local shock normal is not simply radial)."""

import numpy as np
import pytest

from bowshockmaps.physics.thermodynamics import offset_boundary_along_normal


def test_offset_at_apex_is_purely_radial():
    # At theta=0 (the apex), the local normal coincides with the
    # radial direction by symmetry, so offsetting by H should just add
    # H to the radius with no angular change.
    theta = np.array([1e-6])
    rr = np.array([1.0])
    R_base = np.array([1e17])
    H = np.array([1e16])

    theta_new, r_new = offset_boundary_along_normal(theta, rr, R_base, H, lam=0.0)

    assert theta_new[0] == pytest.approx(0.0, abs=1e-4)
    assert r_new[0] == pytest.approx(R_base[0] + H[0], rel=1e-3)


def test_zero_offset_returns_the_base_curve():
    theta = np.linspace(1e-3, np.deg2rad(90), 20)
    rr = np.linspace(1.0, 2.0, 20)
    R_base = rr * 1e17
    H = np.zeros_like(theta)

    theta_new, r_new = offset_boundary_along_normal(theta, rr, R_base, H, lam=0.0)

    # Zero offset: the curve is unchanged (just possibly re-sorted).
    order = np.argsort(theta)
    assert np.allclose(theta_new, theta[order])
    assert np.allclose(r_new, R_base[order])


def test_offset_moves_points_farther_from_origin_on_average():
    # A positive offset along the outward normal should, on average,
    # push the curve's points farther from the star than the base
    # curve (even though individual points can shift in theta too).
    theta = np.linspace(1e-3, np.deg2rad(60), 50)
    rr = 1.0 + 0.3 * theta  # a mildly growing, non-spherical shape
    R_base = rr * 1e17
    H = np.full_like(theta, 1e16)

    theta_new, r_new = offset_boundary_along_normal(theta, rr, R_base, H, lam=0.0)

    assert np.mean(r_new) > np.mean(R_base)


def test_offset_theta_grid_is_sorted():
    theta = np.linspace(1e-3, np.deg2rad(100), 80)
    rr = 1.0 + 0.4 * theta
    R_base = rr * 1e17
    H = np.full_like(theta, 3e16)

    theta_new, _ = offset_boundary_along_normal(theta, rr, R_base, H, lam=0.0)

    assert np.all(np.diff(theta_new) >= 0)
