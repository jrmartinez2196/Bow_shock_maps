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
    app = BowShock(
        "RXJ0528+2838", convolve=False, telescope="VLA", telescope_config="B", band="radio"
    )
    fwhm = app.get_beam_fwhm(fallback_fwhm=999.0)
    assert 0 < fwhm < 999.0


def test_beam_fwhm_falls_back_when_no_telescope_or_override():
    app = BowShock("RXJ0528+2838", convolve=False)
    assert app.get_beam_fwhm(fallback_fwhm=42.0) == 42.0


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


def test_band_given_at_construction_is_in_effect():
    from bowshockmaps.spectral_bands import get_frequency

    app = BowShock("RXJ0528+2838", convolve=False, band="radio")
    assert app.band_name == "radio"
    assert app.nu_ff == get_frequency("radio")


def test_telescope_that_cannot_observe_in_the_band_fails_early_and_clearly():
    # Regression test: `bowshockmaps -s BD+43 --telescope VLA --telescope-config A`
    # (no --band, so the default FUV) gave a beam of ~1e-6 arcsec, which made
    # the Nyquist refinement ask for ~3.6e9 pixels per axis and die with a
    # 26 GiB MemoryError. A VLA does not observe at UV frequencies: say so,
    # immediately, and say which bands would work.
    with pytest.raises(ValueError, match="does not observe") as excinfo:
        BowShock("RXJ0528+2838", convolve=True, telescope="VLA", telescope_config="A")
    assert "radio" in str(excinfo.value)


def test_explicit_beam_fwhm_skips_the_telescope_band_check():
    # With an explicit beam the telescope plays no role, so no check.
    app = BowShock(
        "RXJ0528+2838", convolve=True, telescope="VLA", telescope_config="A", beam_fwhm=2.0
    )
    assert app.get_beam_fwhm(fallback_fwhm=999.0) == 2.0


def test_set_continuum_band_refuses_a_band_the_telescope_cannot_observe():
    app = BowShock(
        "RXJ0528+2838", convolve=False, telescope="VLA", telescope_config="A", band="radio"
    )
    with pytest.raises(ValueError, match="does not observe"):
        app.set_continuum_band("FUV")
    assert app.band_name == "radio"  # unchanged
    app.set_continuum_band("low_radio")  # a valid one still works
    assert app.band_name == "low_radio"


def test_very_fine_beam_does_not_blow_up_the_grid():
    # A beam far finer than anything the grid can afford must be capped at
    # max_pixels and treated as unresolved, not refined without bound.
    app = BowShock("RXJ0528+2838", convolve=True, beam_fwhm=1e-4, max_pixels=40)
    app.nx, app.ny, app.nz = 20, 20, 60
    app.thermo_data = app.compute_thermo()

    with pytest.warns(UserWarning, match="Beam too fine"):
        maps = app.compute_maps()

    assert maps["I_Halpha"].shape == (40, 40)
    assert np.isfinite(maps["I_Halpha"]).all()


# ---------------------------------------------------------------------
# Automatic grid: field of view, line-of-sight range and step
# ---------------------------------------------------------------------


def _quick_app(**kwargs):
    app = BowShock("RXJ0528+2838", convolve=False, **kwargs)
    app.nx = app.ny = 40
    app.thermo_data = app.compute_thermo()
    return app


@pytest.mark.parametrize("inclination", [0.0, 30.0, 75.0, 90.0])
def test_automatic_field_of_view_contains_the_whole_shell(inclination):
    # Regression test for a fixed field of view (+-(6+2 sin^2 i) R0) that
    # clips the shell at large theta_max or low inclination. Emission on the
    # outermost ring of pixels would mean the structure is cut off.
    app = _quick_app(accuracy="fast")
    app.inclination = inclination
    maps = app.compute_maps()
    for key in ("I_Halpha", "I_OIII", "I_ff_total", "I_syn_total"):
        image = maps[key]
        edge = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
        assert np.all(edge == 0.0), (key, inclination)


def test_automatic_field_of_view_gives_square_pixels():
    app = _quick_app(accuracy="fast")
    app.inclination = 0.0  # edge-on: a much wider than tall field
    maps = app.compute_maps()
    dx = np.diff(maps["x"]).mean()
    dy = np.diff(maps["y"]).mean()
    assert dx == pytest.approx(dy, rel=1e-6)  # exactly square
    assert maps["I_ff_total"].shape[1] < maps["I_ff_total"].shape[0]  # fewer columns than rows


def test_fov_option_sets_a_square_field_of_that_half_width():
    app = _quick_app(fov=3.0, accuracy="fast")
    maps = app.compute_maps()
    from bowshockmaps.maps import arcsecond

    half_arcsec = arcsecond(3.0 * app.get_R0_corrected(), app.distance)
    assert maps["x"].min() == pytest.approx(-half_arcsec, rel=1e-6)
    assert maps["x"].max() == pytest.approx(half_arcsec, rel=1e-6)
    assert maps["y"].min() == pytest.approx(-half_arcsec, rel=1e-6)
    assert maps["y"].max() == pytest.approx(half_arcsec, rel=1e-6)


def _record_los_arguments(monkeypatch):
    import bowshockmaps.maps as maps_module

    seen = {}
    real = maps_module.los_projection_vectorized

    def wrapper(*args, **kwargs):
        seen.update(zmax=kwargs["zmax"], nz=kwargs["nz"])
        return real(*args, **kwargs)

    monkeypatch.setattr(maps_module, "los_projection_vectorized", wrapper)
    return seen


def test_line_of_sight_range_and_steps_are_derived_by_default(monkeypatch):
    seen = _record_los_arguments(monkeypatch)
    app = _quick_app(accuracy="fast")
    assert app.zmax is None and app.nz is None  # config.auto_los is on
    app.compute_maps()
    assert seen["nz"] % 2 == 1  # symmetric about z = 0, includes the star's plane
    assert seen["zmax"] > 0


def test_explicit_zmax_and_nz_are_used_exactly(monkeypatch):
    seen = _record_los_arguments(monkeypatch)
    app = _quick_app()
    app.zmax = 7.0 * app.get_R0_corrected()
    app.nz = 123
    app.compute_maps()
    assert seen["nz"] == 123
    assert seen["zmax"] == pytest.approx(7.0 * app.get_R0_corrected())


def test_higher_accuracy_uses_a_finer_line_of_sight_step(monkeypatch):
    # BD+43 has a thin relevant layer, so the step is not pinned by the
    # floor/ceiling and the three accuracies must be ordered.
    seen = _record_los_arguments(monkeypatch)
    steps = {}
    for accuracy in ("fast", "normal", "fine"):
        app = BowShock("BD+43", convolve=False, accuracy=accuracy)
        app.nx = app.ny = 20
        app.thermo_data = app.compute_thermo()
        app.compute_maps()
        steps[accuracy] = 2.0 * seen["zmax"] / (seen["nz"] - 1)
    assert steps["fast"] > steps["normal"] > steps["fine"]
    assert steps["fast"] / steps["normal"] == pytest.approx(2.0, rel=0.1)
    assert steps["normal"] / steps["fine"] == pytest.approx(2.0, rel=0.1)


def test_unknown_accuracy_is_rejected():
    app = _quick_app(accuracy="ultra")
    with pytest.raises(ValueError, match="accuracy"):
        app.compute_maps()


# ---------------------------------------------------------------------
# Layer thicknesses must be physical (regression: RS "muy grueso")
# ---------------------------------------------------------------------


def _all_nonnegative(values):
    """True if every non-NaN value is >= 0. (A shock that is adiabatic at every
    angle has no cold layer, so its cold-layer arrays are entirely NaN; that is
    fine, but np.nanmin of an all-NaN array is NaN and fails any comparison.)"""
    values = np.asarray(values, dtype=float)
    return bool(np.all(values[~np.isnan(values)] >= 0.0))


def _assert_regime_is_cooling_vs_advection(d, shock):
    """The regime is exactly t_cool < t_adv (cooling length shorter than the layer),
    or radiative where no adiabatic flow exists (t_adv is NaN)."""
    ratio = np.asarray(d[f"ratio_{shock}"], dtype=float)
    radiative = np.asarray(d[f"regime_{shock}"]) == "radiative"
    assert np.array_equal(radiative, (ratio < 1.0) | np.isnan(ratio)), shock


def _thermo(source, Vw=None, Vstar=None, n_ism=None, log_mdot=None):
    app = BowShock(source, convolve=False)
    if Vw is not None:
        app.Vw = Vw * 1e5  # km/s -> cm/s
    if Vstar is not None:
        app.Vstar = Vstar * 1e5
    if n_ism is not None:
        app.n_ism = n_ism
    if log_mdot is not None:
        app.Mdot = 10**log_mdot * 1.989e33 / 3.156e7  # Msun/yr -> g/s
    return app.compute_thermo()


def test_radiative_reverse_shock_near_the_axis_with_a_slow_wind_is_not_absurdly_thick():
    # Regression test. With V_wind = 100 km/s (RXJ0528+2838 parameters) the
    # time criterion t_cool < t_adv = R/v_tan is satisfied at every angle
    # near the axis, because v_tan -> 0 there. But the cooling length is then
    # ~1.07 R, more than the accumulated mass can fill: the "radiative"
    # solution had H_hot/R = 1.07 and a cold layer of NEGATIVE thickness. A
    # gas that needs more than a radius to cool does not cool in its layer:
    # it is adiabatic.
    d = _thermo("RXJ0528+2838", Vw=100.0)
    assert d["regime_RS"][0] == "adiabatic"
    assert _all_nonnegative(d["H_RS_cold"])
    assert np.max(d["H_RS_total"]) < 0.5  # thickest it gets is ~0.4 R, at large theta


@pytest.mark.parametrize("source", ["RXJ0528+2838", "BD+43"])
@pytest.mark.parametrize("Vw", [50.0, 100.0, 500.0, 3000.0])
@pytest.mark.parametrize("Vstar", [50.0, 128.5, 300.0])
@pytest.mark.parametrize("n_ism", [0.2, 6.0])
@pytest.mark.parametrize("log_mdot", [-8.0, -6.0])
def test_layers_are_physical_across_the_parameter_range(source, Vw, Vstar, n_ism, log_mdot):
    # No negative thickness, none larger than the radius, no NaN in the hot
    # layer, whichever regime each angle ends up in. (Outside this range --
    # a wind of <= 30 km/s on a star at 300 km/s, or a star at ~2x the sound
    # speed of the ISM -- the model's shock does not really exist and H/R > 1
    # can still occur.)
    d = _thermo(source, Vw=Vw, Vstar=Vstar, n_ism=n_ism, log_mdot=log_mdot)
    for shock in ("RS", "FS"):
        assert _all_nonnegative(d[f"H_{shock}_cold"]), shock
        assert _all_nonnegative(d[f"H_{shock}_hot"]), shock
        assert not np.isnan(d[f"H_{shock}_hot"]).any(), shock
        assert not np.isnan(d[f"H_{shock}_total"]).any(), shock
        assert np.max(d[f"H_{shock}_total"]) <= 1.0, shock
        _assert_regime_is_cooling_vs_advection(d, shock)


def test_regime_has_no_singularity_at_the_symmetry_axis():
    # The regime used to be t_cool < R/v_tan, which diverges where v_tan -> 0
    # (the axis): every angle there was "radiative" whatever the shock, with a
    # cooling layer of ~1 R. Now it compares with the thickness of the layer
    # the gas would form if adiabatic, which is finite at the axis, so the apex
    # is radiative or adiabatic according to the physics.
    # A fast, tenuous wind (cooling length >> layer): adiabatic at the apex.
    hot = _thermo("RXJ0528+2838", Vw=3000.0, Vstar=128.5, n_ism=0.2, log_mdot=-8.0)
    assert hot["regime_RS"][0] == "adiabatic"
    assert hot["ratio_RS"][0] > 1.0
    # Dense gas that cools quickly (BD+43's forward shock): radiative at the apex.
    cool = _thermo("BD+43")
    assert cool["regime_FS"][0] == "radiative"
    assert cool["ratio_FS"][0] < 1.0
    # and in both the thickness at the apex is finite and well below a radius
    assert 0.0 < hot["H_RS_total"][0] < 0.3
    assert 0.0 < cool["H_FS_total"][0] < 0.3


def test_forward_shock_cold_layer_appears_where_the_cooling_length_drops_below_the_layer():
    # Wind of 500 km/s, v_star = 128.5 km/s, n_ISM = 0.2, Mdot = 1e-9: the
    # reverse shock is adiabatic everywhere (its cooling length is ~4e4 times the
    # layer) and the forward shock turns radiative at ~96 deg, where
    # l_cool/H_ad crosses 1.
    d = _thermo("RXJ0528+2838", Vw=500.0, Vstar=128.5, n_ism=0.2, log_mdot=-9.0)
    theta = np.degrees(BowShock("RXJ0528+2838", convolve=False).theta_grid)
    assert np.all(np.asarray(d["regime_RS"]) == "adiabatic")
    radiative = np.asarray(d["regime_FS"]) == "radiative"
    assert radiative.any() and not radiative[0]
    onset = theta[np.argmax(radiative)]
    assert onset == pytest.approx(96.0, abs=2.0)
    assert np.all(radiative[np.argmax(radiative) :])  # and it stays radiative beyond
    _assert_regime_is_cooling_vs_advection(d, "FS")


def test_transition_to_radiative_leaves_a_nonnegative_cold_layer():
    # Near the transition the cold layer from mass conservation (which uses v_tan)
    # can come out marginally negative (1-5% of the hot layer) because the regime
    # is decided with the adiabatic layer (v_adv); it is floored at zero. The hot
    # layer is then the whole shocked layer.
    d = _thermo("RXJ0528+2838", Vw=500.0, Vstar=128.5, n_ism=0.2, log_mdot=-9.0)
    radiative = np.asarray(d["regime_FS"]) == "radiative"
    assert _all_nonnegative(d["H_FS_cold"])
    assert np.all(d["H_FS_hot"][radiative] > 0.0)
    assert np.allclose(d["H_FS_total"], d["H_FS_hot"] + np.nan_to_num(d["H_FS_cold"]))
