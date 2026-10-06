"""Tests for physics/thermodynamics.py (post-shock conditions)."""

import numpy as np

from bowshockmaps.physics.thermodynamics import post_shock_conditions


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
