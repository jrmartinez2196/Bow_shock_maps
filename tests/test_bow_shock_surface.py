"""Basic physical sanity checks for the bow-shock surface geometry."""

from bowshockmaps.constants import Msun_yr, pc
from bowshockmaps.physics.bow_shock_surface import standoff_distance


def test_standoff_distance_is_positive():
    R0 = standoff_distance(Mdot=1e-8 * Msun_yr, Vw=1000e5, Vstar=30e5, n_ism=1.0)
    assert R0 > 0


def test_standoff_distance_scales_with_wind_momentum():
    # Doubling the wind mass-loss rate should increase the standoff
    # distance (more wind momentum pushes the shock further out).
    common = dict(Vw=1000e5, Vstar=30e5, n_ism=1.0)
    R0_low = standoff_distance(Mdot=1e-8 * Msun_yr, **common)
    R0_high = standoff_distance(Mdot=2e-8 * Msun_yr, **common)
    assert R0_high > R0_low


def test_standoff_distance_shrinks_with_higher_stellar_velocity():
    # A faster-moving star compresses the bow shock closer to the star.
    common = dict(Mdot=1e-8 * Msun_yr, Vw=1000e5, n_ism=1.0)
    R0_slow = standoff_distance(Vstar=10e5, **common)
    R0_fast = standoff_distance(Vstar=50e5, **common)
    assert R0_fast < R0_slow


def test_standoff_distance_is_of_plausible_astrophysical_order():
    # For typical runaway-star parameters, R0 should be of order
    # 0.01-1 pc, not off by many orders of magnitude.
    R0 = standoff_distance(Mdot=1e-8 * Msun_yr, Vw=1000e5, Vstar=30e5, n_ism=1.0)
    assert 1e-3 * pc < R0 < 10 * pc
