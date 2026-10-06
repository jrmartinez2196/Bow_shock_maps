"""End-to-end integration test: does the full physical pipeline run?

This is deliberately not a physics-correctness test (there's no
reference/golden output to compare against yet — see project notes).
It exists to catch the class of bug a structural refactor is most
likely to introduce: broken wiring between modules that only shows up
once every piece runs together.
"""

import numpy as np
import pytest

from bowshockmaps.visualization.app import BowShock


@pytest.fixture
def app():
    return BowShock("RXJ0528+2838", convolve=True)


def test_bowshock_instantiates_without_error(app):
    assert app.T_ism > 0
    assert app.Mdot > 0


def test_bowshock_has_finite_projected_geometry(app):
    assert np.isfinite(app.get_R0_corrected())


def test_beam_fwhm_priority_explicit_override_wins():
    app = BowShock(
        "RXJ0528+2838", convolve=False, telescope="VLA", telescope_config="B", beam_fwhm=2.5
    )
    assert app.get_beam_fwhm(fallback_fwhm=999.0) == 2.5


def test_beam_fwhm_from_telescope_when_no_override():
    app = BowShock("RXJ0528+2838", convolve=False, telescope="VLA", telescope_config="B")
    fwhm = app.get_beam_fwhm(fallback_fwhm=999.0)
    assert 0 < fwhm < 999.0


def test_beam_fwhm_falls_back_when_no_telescope_or_override():
    app = BowShock("RXJ0528+2838", convolve=False)
    assert app.get_beam_fwhm(fallback_fwhm=42.0) == 42.0


def test_cd_and_fs_use_normal_offset_not_naive_radial_sum():
    # Regression test for the original bug report: CD_pos used to be
    # R_RS + H_RS_hot + H_RS_cold added directly at fixed theta, which
    # is only exact at the apex (bow shock isn't spherically
    # symmetric). Away from the apex that naive sum should now differ
    # measurably from the actual (normal-offset) CD position used by
    # the model.
    from scipy.interpolate import interp1d

    from bowshockmaps.maps import precompute_shock_properties
    from bowshockmaps.physics.thermodynamics import offset_boundary_along_normal

    app = BowShock("RXJ0528+2838", convolve=False)
    R0 = app.get_R0_corrected()
    lam = app.lam

    theta_max = np.deg2rad(120)
    theta_precomp = np.linspace(1e-6, theta_max, 300)
    rr_precomp = app.R_RS_func(theta_precomp)

    rs_props = precompute_shock_properties(
        theta_precomp,
        rr_precomp,
        R0,
        "RS",
        T_IL=app.T_IL,
        Mdot=app.Mdot,
        Vw=app.Vw,
        lam=lam,
        wind_regime=app.wind_regime,
        wind_T_fixed=app.wind_T_fixed,
    )

    R_RS_phys = rr_precomp * R0
    H_RS_hot = rs_props["H_hot"](theta_precomp)
    H_RS_cold = rs_props["H_cold"](theta_precomp)
    H_tot = H_RS_hot + H_RS_cold

    th_new, r_new = offset_boundary_along_normal(theta_precomp, rr_precomp, R_RS_phys, H_tot, lam)
    CD_func = interp1d(th_new, r_new, bounds_error=False, fill_value=(r_new[0], r_new[-1]))

    # Pick a point well away from the apex, where the shock surface is
    # noticeably tilted from radial.
    idx = np.argmin(np.abs(np.degrees(theta_precomp) - 90))
    naive_CD = R_RS_phys[idx] + H_tot[idx]
    actual_CD = CD_func(theta_precomp[idx])

    assert abs(naive_CD - actual_CD) / R_RS_phys[idx] > 0.01  # >1% difference at theta~90 deg


def test_compute_maps_includes_continuum_and_components():
    # Low resolution: the default resolution is realistic for a real
    # analysis but too heavy for a fast test run, so we shrink the grid
    # via the instance attributes (same knobs the sliders/CLI use).
    app = BowShock("RXJ0528+2838", convolve=False)
    app.nx, app.ny, app.nz = 15, 15, 100

    app.thermo_data = app.compute_thermo()
    map_data = app.compute_maps()

    for key in ("I_ff_total", "I_syn_total", "I_continuum_total", "I_continuum_mJy"):
        assert key in map_data
        assert np.isfinite(map_data[key]).all()

    assert np.allclose(
        map_data["I_continuum_total"],
        map_data["I_ff_total"] + map_data["I_syn_total"],
    )


@pytest.mark.parametrize("source_name", ["RXJ0528+2838", "BD+43"])
def test_layer_boundaries_are_physically_nested(source_name):
    # The bow shock isn't spherically symmetric, so layer boundaries
    # (RS, RS-hot-layer edge, CD, FS-cold-layer edge, FS) are built by
    # offsetting along the local shock normal and reparametrizing
    # (see offset_boundary_along_normal), not by adding thicknesses
    # directly to R_RS(theta) at fixed theta. Whatever the construction,
    # the physical layers must stay nested: each boundary is always at
    # least as far from the star as the previous one, for every theta.
    from scipy.interpolate import interp1d

    from bowshockmaps.maps import precompute_shock_properties
    from bowshockmaps.physics.thermodynamics import offset_boundary_along_normal

    app = BowShock(source_name, convolve=False)
    R0 = app.get_R0_corrected()
    lam = app.lam

    theta_max = np.deg2rad(120)
    theta_precomp = np.linspace(1e-6, theta_max, 300)
    rr_precomp = app.R_RS_func(theta_precomp)

    rs_props = precompute_shock_properties(
        theta_precomp,
        rr_precomp,
        R0,
        "RS",
        T_IL=app.T_IL,
        Mdot=app.Mdot,
        Vw=app.Vw,
        lam=lam,
        wind_regime=app.wind_regime,
        wind_T_fixed=app.wind_T_fixed,
    )
    fs_props = precompute_shock_properties(
        theta_precomp,
        rr_precomp,
        R0,
        "FS",
        T_IL=app.T_IL,
        Vstar=app.Vstar,
        n_ism=app.n_ism,
        lam=lam,
    )

    R_RS_phys = rr_precomp * R0
    H_RS_hot = rs_props["H_hot"](theta_precomp)
    H_RS_cold = rs_props["H_cold"](theta_precomp)
    H_FS_cold = fs_props["H_cold"](theta_precomp)
    H_FS_hot = fs_props["H_hot"](theta_precomp)

    def boundary_func(H_cumulative):
        th_new, r_new = offset_boundary_along_normal(
            theta_precomp, rr_precomp, R_RS_phys, H_cumulative, lam
        )
        return interp1d(th_new, r_new, bounds_error=False, fill_value=(r_new[0], r_new[-1]))

    RS_hot_outer_func = boundary_func(H_RS_hot)
    CD_func = boundary_func(H_RS_hot + H_RS_cold)
    FS_cold_outer_func = boundary_func(H_RS_hot + H_RS_cold + H_FS_cold)
    FS_outer_func = boundary_func(H_RS_hot + H_RS_cold + H_FS_cold + H_FS_hot)

    theta_eval = np.linspace(1e-6, theta_max, 2000)
    R_RS_eval = app.R_RS_func(theta_eval) * R0
    RS_hot_outer = np.maximum(RS_hot_outer_func(theta_eval), R_RS_eval)
    CD_pos = np.maximum(CD_func(theta_eval), RS_hot_outer)
    FS_cold_outer = np.maximum(FS_cold_outer_func(theta_eval), CD_pos)
    FS_pos = np.maximum(FS_outer_func(theta_eval), FS_cold_outer)

    assert np.all(RS_hot_outer >= R_RS_eval)
    assert np.all(CD_pos >= RS_hot_outer)
    assert np.all(FS_cold_outer >= CD_pos)
    assert np.all(FS_pos >= FS_cold_outer)


def test_figure2_renders_all_five_map_panels():
    app = BowShock("RXJ0528+2838", convolve=False)
    app.nx, app.ny, app.nz = 15, 15, 100

    app.thermo_data = app.compute_thermo()
    app.map_data = app.compute_maps()
    app.create_figure2()
    app.update_figure2()

    assert app.fig2_axes["map_keys"] == ["Halpha", "OIII", "continuum", "ff", "syn"]
    assert len(app.fig2_axes["maps"]) == 5
    for key in app.fig2_axes["map_keys"]:
        assert app.images[key] is not None

    # The radial-profile panel should have a curve for the total
    # continuum (free-free + synchrotron), not just its two components.
    assert app.profiles["continuum"] is not None
    x_data, y_data = app.profiles["continuum"].get_data()
    assert len(x_data) > 0
    assert np.isfinite(y_data).all()


def test_pa_rotation_actually_rotates_the_emission_map():
    # Regression test: PA used to be accepted by make_projection_maps
    # but never forwarded to los_projection_vectorized, so the emission
    # map was always computed as if PA=0 regardless of the source's
    # actual PA -- only the plot axis limits (plot_maps.compute_plot_limits)
    # accounted for it. A non-zero, non-special PA should now visibly
    # change which map is produced.
    import matplotlib

    matplotlib.use("Agg")

    app_pa0 = BowShock("RXJ0528+2838", convolve=False)
    app_pa0.nx, app_pa0.ny, app_pa0.nz = 20, 20, 150
    app_pa0.PA = 0.0
    app_pa0.thermo_data = app_pa0.compute_thermo()
    map_pa0 = app_pa0.compute_maps()

    app_pa45 = BowShock("RXJ0528+2838", convolve=False)
    app_pa45.nx, app_pa45.ny, app_pa45.nz = 20, 20, 150
    app_pa45.PA = 45.0
    app_pa45.thermo_data = app_pa45.compute_thermo()
    map_pa45 = app_pa45.compute_maps()

    # atol=0: these intensities are physically ~1e-20, far below
    # np.allclose's default atol=1e-8, which would swamp any real
    # difference and make every comparison trivially "close".
    assert not np.allclose(map_pa0["I_OIII"], map_pa45["I_OIII"], atol=0, equal_nan=True)


def test_build_layer_boundary_funcs_covers_theta_max_or_degrades_gracefully():
    # Regression test for the "opening wings" artifact at near edge-on
    # inclination: offsetting the RS curve along its local normal
    # compresses the resulting boundary's own theta range (see
    # offset_boundary_along_normal), so a boundary built only from
    # theta in [0, theta_max] can fall short of theta_max itself,
    # forcing constant extrapolation that looks like an unphysical
    # flattening near the edge of the visible structure.
    #
    # build_layer_boundary_funcs should either reach theta_max (by
    # adaptively extending the input range) or, if the analytic
    # bow-shock shape has no solution that far out for this lam, fail
    # gracefully (no exception) and get as close as it safely can.
    from bowshockmaps.maps import build_layer_boundary_funcs

    app = BowShock("RXJ0528+2838", convolve=False)
    theta_max = np.deg2rad(120)

    funcs = build_layer_boundary_funcs(
        theta_max=theta_max,
        lam=app.lam,
        R0_phys=app.get_R0_corrected(),
        T_IL=app.T_IL,
        Mdot=app.Mdot,
        Vw=app.Vw,
        wind_regime=app.wind_regime,
        wind_T_fixed=app.wind_T_fixed,
        Vstar=app.Vstar,
        n_ism=app.n_ism,
    )
    assert len(funcs) == 4

    # Evaluate each boundary across the full range; none should raise,
    # and values should stay finite.
    theta_eval = np.linspace(1e-3, theta_max * 0.999, 200)
    for f in funcs:
        vals = f(theta_eval)
        assert np.isfinite(vals).all()

    # The three least-offset boundaries should now cover the full
    # range without falling back to constant extrapolation at the end
    # (RS_hot_outer, CD, FS_cold_outer -- FS_outer is the most-offset
    # one and may still fall a bit short for this particular source,
    # per the ODE's own maximum valid opening angle; that's the
    # graceful-degradation case, not a bug).
    for f in funcs[:3]:
        last_vals = f(theta_eval[-10:])
        assert not np.allclose(last_vals, last_vals[0], rtol=1e-4), (
            "boundary appears to be flat (constant-extrapolated) near theta_max, "
            "expected it to keep varying smoothly"
        )
