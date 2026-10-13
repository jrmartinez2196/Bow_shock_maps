"""Thermal dust emission of the forward-shock layers."""

import numpy as np
import pytest
from scipy.integrate import quad

from bowshockmaps.constants import c, h, kB
from bowshockmaps.physics import dust
from bowshockmaps.visualization.app import BowShock

AU = 1.495978707e13


def test_cooling_constant_matches_numerical_integral():
    """P_em = 4 pi a^2 * pi * int Q_em(nu) B_nu dnu = K T^6 (Q_em ~ nu^2, uncapped)."""
    a_um, T = 0.1, 30.0
    a = a_um * 1e-4
    q = dust.Q100 * (a_um / dust.A_REF_UM)

    def integrand(nu):
        return q * (nu / dust.NU_100UM) ** 2 * 2 * h * nu**3 / c**2 / np.expm1(h * nu / (kB * T))

    nu_peak = 2.8 * kB * T / h
    numeric, _ = quad(integrand, 1e-3 * nu_peak, 60 * nu_peak, limit=200)
    numeric *= 4 * np.pi**2 * a**2
    assert numeric == pytest.approx(dust._cooling_constant(a_um) * T**6, rel=1e-3)


def test_temperature_balances_heating_and_emission():
    n_e, T_gas, r, L = 1.0, 2.4e5, 8000 * AU, 1e33
    T_d = dust.dust_temperature(n_e, T_gas, r, L, 0.1)
    heating = dust.collisional_heating_per_grain(n_e, T_gas, 0.1) + dust.stellar_heating_per_grain(
        r, L, 0.1
    )
    assert dust._cooling_constant(0.1) * T_d**6 == pytest.approx(heating, rel=1e-10)


def test_heating_scalings():
    base = dust.collisional_heating_per_grain(1.0, 2e5, 0.1)
    assert dust.collisional_heating_per_grain(2.0, 2e5, 0.1) == pytest.approx(2 * base)
    assert dust.collisional_heating_per_grain(1.0, 2e5, 0.2) == pytest.approx(4 * base)
    s = dust.stellar_heating_per_grain(1e17, 1e33, 0.1)
    assert dust.stellar_heating_per_grain(2e17, 1e33, 0.1) == pytest.approx(s / 4)
    assert dust.stellar_heating_per_grain(1e17, 2e33, 0.1) == pytest.approx(2 * s)


def test_hotter_and_denser_gas_heats_the_dust_more():
    r = 8000 * AU
    cold = dust.dust_temperature(1.0, 1e4, r, 0.0, 0.1)
    hot = dust.dust_temperature(1.0, 1e6, r, 0.0, 0.1)
    dense = dust.dust_temperature(100.0, 1e6, r, 0.0, 0.1)
    assert cold < hot < dense


def test_efficiency_capped_at_one():
    assert dust.emission_efficiency(c / 1e-4, 0.1) == 1.0
    assert dust.emission_efficiency(dust.NU_100UM, 0.1) == pytest.approx(dust.Q100)


def test_emissivity_linear_in_dust_to_gas_ratio_and_zero_without_dust():
    args = (2.0, 1.0, 2.4e5, 8000 * AU, c / 22e-4, 1e32)
    j1 = dust.dust_emissivity(*args, a_um=0.1, f_dust_gas=0.01)
    j2 = dust.dust_emissivity(*args, a_um=0.1, f_dust_gas=0.02)
    assert j1 > 0
    assert j2 == pytest.approx(2 * j1)
    assert dust.dust_emissivity(*args, a_um=0.1, f_dust_gas=0.0) == 0.0


def test_no_overflow_for_cold_grains_at_short_wavelength():
    j = dust.dust_emissivity(2.0, 1.0, 1e4, 8000 * AU, c / 1e-4, 0.0)
    assert np.isfinite(j) and j >= 0.0


def _maps(use_dust):
    app = BowShock("RXJ0528+2838", convolve=False)
    app.nx, app.ny = 30, 30
    app.thermo_data = app.compute_thermo()
    app.dust_lambda_um = 22.0  # where the equilibrium dust is not exactly zero
    result = app.compute_maps() if use_dust else None
    return app, result


def test_map_has_finite_nonnegative_dust_maps():
    _, m = _maps(True)
    for key in ("I_dust_MJy_sr", "I_dust_mJy"):
        assert key in m
        assert np.all(np.isfinite(m[key])) and np.all(m[key] >= 0.0)
    assert m["I_dust_MJy_sr"].max() > 0.0
    # Both maps are the same quantity in different units
    ratio = m["I_dust_mJy"][m["I_dust_MJy_sr"] > 0] / m["I_dust_MJy_sr"][m["I_dust_MJy_sr"] > 0]
    assert np.allclose(ratio, ratio[0])


def test_shock_radiated_luminosity_is_positive_and_small_for_adiabatic_rs():
    from bowshockmaps.physics.dust import shock_radiated_luminosity
    from bowshockmaps.physics.shell_geometry import ShellGeometry
    from bowshockmaps.maps import precompute_shock_properties

    app = BowShock("RXJ0528+2838", convolve=False)
    R0 = app.get_R0_corrected()
    theta = np.linspace(1e-3, np.radians(120), 400)
    rr = app.R_RS_func(theta)
    kw_rs = dict(Mdot=app.Mdot, Vw=app.Vw, lam=app.lam, wind_regime=app.wind_regime, wind_T_fixed=None)
    kw_fs = dict(Vstar=app.Vstar, n_ism=app.n_ism, lam=app.lam)
    rs = precompute_shock_properties(theta, rr, R0, "RS", T_IL=app.T_IL, **kw_rs)
    fs = precompute_shock_properties(theta, rr, R0, "FS", T_IL=app.T_IL, **kw_fs)
    shell = ShellGeometry(
        theta, rr, R0, rs["H_hot"](theta), rs["H_cold"](theta), fs["H_cold"](theta), fs["H_hot"](theta)
    )
    L = shock_radiated_luminosity(shell, rs, fs)
    assert L["total"] > 0.0
    assert L["total"] == pytest.approx(sum(L["RS"]) + sum(L["FS"]))
    # The file's wind is fast and hot (adiabatic reverse shock): it radiates far less than the FS
    assert sum(L["RS"]) < sum(L["FS"])


def test_shock_heating_never_lowers_the_dust_emission():
    def peak(f):
        app = BowShock("RXJ0528+2838", convolve=False)
        app.nx, app.ny = 30, 30
        app.dust_lambda_um = 70.0
        app.f_uv_shock = f
        app.thermo_data = app.compute_thermo()
        return app.compute_maps()["I_dust_MJy_sr"].max()

    assert peak(1.0) >= peak(0.0) > 0.0
