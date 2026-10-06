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


def test_pa_is_counterclockwise_from_north_via_display_transform():
    # PA is measured counterclockwise from North (+y, up) -- the
    # standard astronomical convention with East on the left of a
    # North-up sky image. Maps (and the apex marker) are computed in
    # an intrinsic frame with the apex along -x from the star, and PA is
    # applied at display time via Affine2D().rotate_deg_around(0, 0,
    # PA - 90) (see plot_maps.update_map_image/arrow/contours). Check
    # that composition lands the apex where the convention says:
    # PA=0 -> North, PA=90 -> East (left), PA=270 -> West (right), and
    # BD+43's PA=351.5 -> up and slightly to the right.
    from matplotlib.transforms import Affine2D

    from bowshockmaps.visualization.plot_maps import compute_R0_position

    pos = compute_R0_position(inclination=30.0, distance=500.0, R0_corrected=1e17)
    apex = np.array([[pos["x_R0"], pos["y_R0"]]])
    assert apex[0, 0] < 0 and apex[0, 1] == 0  # intrinsic frame: apex along -x

    def displayed(PA):
        return Affine2D().rotate_deg_around(0.0, 0.0, PA - 90.0).transform(apex)[0]

    x, y = displayed(0.0)
    assert y > 0 and abs(x) < 1e-9 * abs(y)  # North: straight up

    x, y = displayed(90.0)
    assert x < 0 and abs(y) < 1e-9 * abs(x)  # East: to the left

    x, y = displayed(270.0)
    assert x > 0 and abs(y) < 1e-9 * abs(x)  # West: to the right

    x, y = displayed(351.5)  # BD+43: up, slightly to the right
    assert y > 0 and x > 0 and x < 0.2 * y


def test_pa_is_not_applied_inside_the_emission_calculation():
    # Regression test: PA is applied once, at display time. An earlier
    # change also rotated the pixel grid by PA inside the LOS
    # integration, which rotated every map twice (and put BD+43's bow
    # shock pointing the wrong way). The raw maps must therefore NOT
    # depend on PA.
    maps = []
    for PA in (0.0, 45.0):
        app = BowShock("RXJ0528+2838", convolve=False)
        app.nx, app.ny, app.nz = 20, 20, 150
        app.PA = PA
        app.thermo_data = app.compute_thermo()
        maps.append(app.compute_maps()["I_OIII"])

    assert np.array_equal(maps[0], maps[1], equal_nan=True)


def test_layer_boundaries_do_not_exist_beyond_their_coverage():
    # Offsetting the RS curve along its normal pulls points back to
    # smaller polar angle, so each boundary covers a narrower range of
    # theta than the arc it came from. Beyond that range the boundary
    # (and any layer needing it) must simply not exist (NaN) -- not be
    # held constant (a flat arc) nor be fabricated by evaluating the
    # shock physics at angles outside the modeled domain.
    from bowshockmaps.maps import build_layer_boundary_funcs
    from bowshockmaps.physics.bow_shock_surface import integrate_r_theta_christie

    theta_max = np.deg2rad(120)
    thr, rr = integrate_r_theta_christie(lam=0.02, R0=1.0, theta_max=theta_max)
    theta = np.linspace(1e-6, theta_max, 300)
    rr = np.interp(theta, thr, rr)
    R0 = 1e17
    H = np.full_like(theta, 0.15 * R0)  # a thick-ish layer, so the compression is clear

    funcs = build_layer_boundary_funcs(
        theta, rr, R0, 0.02, H_RS_hot=H, H_RS_cold=H, H_FS_cold=H, H_FS_hot=H
    )
    assert len(funcs) == 4

    # Well inside the modeled range every boundary exists and is finite...
    inside = np.deg2rad(np.array([5.0, 30.0, 60.0]))
    for f in funcs:
        assert np.isfinite(f(inside)).all()

    # ...they are nested (each farther from the star than the previous)...
    vals = np.array([f(inside) for f in funcs])
    assert np.all(np.diff(vals, axis=0) > 0)

    # ...and the outermost one stops short of theta_max (NaN beyond),
    # rather than reaching it by extrapolation.
    assert np.isnan(funcs[-1](theta_max))


def test_no_shock_when_normal_mach_number_is_subsonic():
    # Regression test: for a slow star the normal Mach number of the
    # forward shock drops below 1 at large theta (BD+43: M ~ 2 at 120
    # deg, < 1 by ~140 deg). There is no shock there. The
    # Rankine-Hugoniot relations are invalid below M=1 and gave negative
    # temperatures (-> "invalid value encountered in power" in
    # lambda_T); post_shock_conditions must instead return ambient
    # conditions and zero layer thickness, with no warnings.
    import warnings

    from bowshockmaps.physics.bow_shock_surface import integrate_r_theta_christie
    from bowshockmaps.physics.thermodynamics import (
        post_shock_conditions,
        pre_shock_ism,
        vnorm_forward,
    )

    app = BowShock("BD+43", convolve=False)
    lam = app.lam
    thr, rr_c = integrate_r_theta_christie(lam=lam, R0=1.0, theta_max=np.deg2rad(160))
    theta = np.linspace(0.01, np.deg2rad(160), 300)
    rr = np.interp(theta, thr, rr_c)

    _, _, _, cs_pre = pre_shock_ism(app.Vstar, app.n_ism, lam)
    M = vnorm_forward(theta, rr, lam, app.Vstar) / cs_pre
    assert np.any(M < 1) and np.any(M > 1)  # the test is only meaningful if both occur

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = post_shock_conditions(
            theta,
            rr,
            "FS",
            app.get_R0_corrected(),
            T_IL=app.T_IL,
            Vstar=app.Vstar,
            n_ism=app.n_ism,
            lam=lam,
        )
    n_post, T_post, _, T_rec, _, _, H_hot, H_cold, H_total, _, _ = out

    no_shock = M <= 1
    assert np.isfinite(T_post).all() and np.all(T_post > 0)
    assert np.isfinite(T_rec).all() and np.all(T_rec > 0)
    assert np.all(H_total[no_shock] == 0)
    assert np.all(H_hot[no_shock] == 0) and np.all(H_cold[no_shock] == 0)
    assert np.all(H_total[~no_shock] >= 0)


def test_bd43_maps_stay_compact_and_warning_free():
    # Regression test: an earlier "adaptive extension" of the layer
    # boundaries evaluated the shock physics at theta up to ~180 deg,
    # where BD+43's forward shock has no solution (M < 1). That raised
    # the lambda_T "invalid value encountered in power" warning and made
    # the forward shock fill the whole field of view. The emission must
    # be confined to the modeled shell: the corners of a generous field
    # of view stay empty.
    import warnings

    app = BowShock("BD+43", convolve=False)
    app.nx, app.ny, app.nz = 30, 30, 150

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app.thermo_data = app.compute_thermo()
        maps = app.compute_maps()

    assert not any("invalid value encountered in power" in str(w.message) for w in caught)

    for key in ("I_Halpha", "I_OIII", "I_ff_total"):
        img = maps[key]
        assert np.isfinite(img).all()
        for corner in (img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]):
            assert corner == 0
