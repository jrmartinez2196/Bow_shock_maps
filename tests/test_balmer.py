"""Balmer-dominated H-alpha of the forward shock (neutral ISM crossing it)."""

import numpy as np
import pytest

from bowshockmaps.physics.radiation import balmer_ionization_length, balmer_photon_flux
from bowshockmaps.visualization.app import BowShock


def test_flux_scales_with_neutral_flux():
    base = balmer_photon_flux(0.2, 0.0, 1e7, eta=0.3)
    assert base == pytest.approx(0.3 * 0.2 * 1e7)
    assert balmer_photon_flux(0.2, 0.0, 2e7, 0.3) == pytest.approx(2 * base)
    assert balmer_photon_flux(0.2, 0.0, 1e7, 0.6) == pytest.approx(2 * base)
    assert balmer_photon_flux(0.2, 0.5, 1e7, 0.3) == pytest.approx(0.5 * base)


def test_flux_zero_for_fully_ionized_ism_or_no_inflow():
    assert balmer_photon_flux(0.2, 1.0, 1e7) == 0.0
    assert balmer_photon_flux(0.2, 0.0, -1e6) == 0.0


def test_ionization_length_decreases_with_density_and_is_positive():
    n = np.array([0.1, 1.0, 10.0])
    ell = balmer_ionization_length(n, 2.4e5, 1.3e7)
    assert np.all(ell > 0) and np.all(np.diff(ell) < 0)
    # inversely proportional to density
    assert ell[0] / ell[1] == pytest.approx(10.0)


def _halpha(eta, r0_str):
    app = BowShock("RXJ0528+2838", convolve=False)
    app.nx, app.ny = 40, 40
    app.balmer_eta = eta
    app.r0_str = r0_str
    app.thermo_data = app.compute_thermo()
    return app.compute_maps()["I_Halpha"]


def test_balmer_adds_halpha_only_outside_stromgren_sphere():
    off = _halpha(0.0, 0.9)
    on = _halpha(0.3, 0.9)
    assert on.sum() > off.sum()
    assert np.all(on >= off - 1e-12)
    # A Stromgren sphere that holds the whole shell: ISM fully ionized, no neutrals.
    inside = _halpha(0.3, 10.0)
    assert inside.sum() == pytest.approx(_halpha(0.0, 10.0).sum())


def test_balmer_is_linear_in_eta():
    h0 = _halpha(0.0, 0.9).sum()
    h1 = _halpha(0.3, 0.9).sum() - h0
    h2 = _halpha(0.6, 0.9).sum() - h0
    assert h2 == pytest.approx(2 * h1, rel=1e-6)
