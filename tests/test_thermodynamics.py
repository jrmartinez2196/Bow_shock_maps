"""Tests for physics/thermodynamics.py (post-shock conditions)."""

import numpy as np
import pytest

from bowshockmaps.physics.thermodynamics import (
    blend_adiabatic_thickness,
    post_shock_conditions,
)


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


# ---------------------------------------------------------------------
# blend_adiabatic_thickness: continuity of the thickness at the regime change
# ---------------------------------------------------------------------


def test_blend_equals_the_radiative_width_at_the_transition():
    # l_cool = W is the transition; there the thickness must be W, whatever H_ad.
    for H_ad in (0.3, 1.0, 7.5):
        assert blend_adiabatic_thickness(H_ad, 1.0, 1.0) == pytest.approx(1.0)


def test_blend_keeps_the_adiabatic_thickness_far_from_the_transition():
    # l_cool >> W: Bernoulli holds, nothing is blended.
    assert blend_adiabatic_thickness(2.0, 1.0, 1e6) == pytest.approx(2.0, rel=1e-9)
    assert blend_adiabatic_thickness(0.4, 1.0, 1e12) == pytest.approx(0.4, rel=1e-9)


def test_blend_is_a_geometric_interpolation_with_a_squared_weight():
    # l_cool = 2 W: w = (1/2)^2 = 0.25, H = H_ad^0.75 * W^0.25
    H_ad, W = 3.0, 1.5
    expected = H_ad**0.75 * W**0.25
    assert blend_adiabatic_thickness(H_ad, W, 2.0 * W) == pytest.approx(expected)


def test_blend_lies_between_the_adiabatic_thickness_and_the_width_and_is_monotonic():
    H_ad, W = 2.0, 1.0
    l_cool = np.logspace(0, 4, 50) * W  # from the transition outwards
    H = blend_adiabatic_thickness(H_ad, W, l_cool)
    assert np.all(H <= H_ad + 1e-12) and np.all(H >= W - 1e-12)
    assert np.all(np.diff(H) >= -1e-12)  # moves monotonically from W to H_ad


def test_blend_is_local_the_correction_dies_off_quickly():
    # At l_cool = 10 W the weight is 1e-2: the thickness is within ~1% of H_ad.
    H_ad, W = 2.0, 1.0
    assert blend_adiabatic_thickness(H_ad, W, 10.0 * W) == pytest.approx(H_ad, rel=1e-2)


@pytest.mark.parametrize(
    "H_ad, W, l_cool",
    [
        (1.0, np.nan, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, np.nan),
        (1.0, np.inf, 1.0),
    ],
)
def test_blend_returns_the_adiabatic_thickness_when_there_is_nothing_to_blend_with(H_ad, W, l_cool):
    assert blend_adiabatic_thickness(H_ad, W, l_cool) == H_ad


def test_blend_works_on_arrays():
    out = blend_adiabatic_thickness(
        np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([1.0, 1e9])
    )
    assert out == pytest.approx([1.0, 2.0])
