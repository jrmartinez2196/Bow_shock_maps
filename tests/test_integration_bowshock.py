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
