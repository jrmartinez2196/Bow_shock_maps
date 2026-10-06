"""Tests for physics/thermodynamics.py, focused on the geometric
offset-along-normal construction used for layer boundaries (the bow
shock is not spherically symmetric, so a layer thickness measured
along the local shock normal is not simply radial)."""

import numpy as np
import pytest

from bowshockmaps.physics.thermodynamics import (
    offset_boundary_along_normal,
    post_shock_conditions,
)


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


def test_layer_thickness_is_frozen_below_theta_min_near_apex():
    # Regression test: the mass-flux formula behind H_hot/H_cold (in
    # whichever branch, radiative or adiabatic, uses it) has a sin(theta)
    # factor in its denominator -- a genuine geometric singularity at
    # the symmetry axis. Below a small theta_min cutoff, post_shock_conditions
    # should freeze H_hot/H_cold/H_total to their value at theta_min
    # instead of trusting the near-singular formula (same approach as
    # the reference Fortran implementation this model is based on).
    theta = np.linspace(1e-3, np.deg2rad(20), 200)
    rr = 1.0 + 0.1 * theta  # mildly growing, non-degenerate shape
    R0_phys = 1e17

    (
        n_post,
        T_post,
        n_rec,
        T_rec,
        P_post,
        regime,
        H_hot,
        H_cold,
        H_total,
        t_cool,
        t_adv,
    ) = post_shock_conditions(theta, rr, "FS", R0_phys, T_IL=8e3, Vstar=50e5, n_ism=0.2, lam=0.02)

    theta_min = 0.1  # rad, matches the cutoff used internally
    below = theta < theta_min
    above = ~below

    assert np.any(below) and np.any(above)

    i_min = np.argmax(above)  # first index at/above theta_min
    assert np.allclose(H_hot[below], H_hot[i_min])
    assert np.allclose(H_cold[below], H_cold[i_min])
    assert np.allclose(H_total[below], H_total[i_min])

    # And no blow-up: nothing below theta_min should be wildly larger
    # than the (well-behaved) value just above the cutoff.
    assert np.all(H_total[below] <= 10 * np.max(H_total[above]))
