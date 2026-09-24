"""Sanity checks on physical constants (cgs units)."""

import pytest

from bowshockmaps import constants


def test_constants_are_positive():
    positive_names = [
        "mu",
        "mu_sh",
        "mp",
        "kB",
        "me",
        "qe",
        "c",
        "G",
        "eV",
        "mec2",
        "sigma_T",
        "Ry",
        "h",
        "sr_per_arcsec2",
        "Rayleigh",
        "Msun",
        "Rsun",
        "Lsun",
        "AU",
        "pc",
        "year",
        "Msun_yr",
        "gamma_ad",
        "z_w",
        "z_ism",
    ]
    for name in positive_names:
        value = getattr(constants, name)
        assert value > 0, f"{name} should be positive, got {value}"


def test_msun_yr_is_consistent_with_msun_and_year():
    assert constants.Msun_yr == constants.Msun / constants.year


def test_speed_of_light_matches_known_value():
    assert constants.c == pytest.approx(2.99792458e10)  # cm/s
