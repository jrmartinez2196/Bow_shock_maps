"""Tests for the normal-coordinate shell geometry and the LOS integration
built on it, against cases with known analytic answers (a spherical shell,
a flat slab)."""

import numpy as np
import pytest

import bowshockmaps.maps as maps
from bowshockmaps.physics.shell_geometry import ShellGeometry


def sphere_shell(thicknesses=(0.1, 0.2, 0.1, 0.3), theta_max_deg=120.0, n=3000):
    """A sphere of radius R0=1 as the 'reverse shock', with constant-thickness layers."""
    theta = np.linspace(1e-6, np.deg2rad(theta_max_deg), n)
    one = np.ones_like(theta)
    h1, h2, h3, h4 = (h * one for h in thicknesses)
    return ShellGeometry(theta, one, 1.0, h1, h2, h3, h4)


def test_normal_of_a_sphere_is_radial():
    shell = sphere_shell()
    assert np.allclose(shell.n_rho, np.sin(shell.theta), atol=1e-3)
    assert np.allclose(shell.n_z, np.cos(shell.theta), atol=1e-3)


def test_layers_of_a_spherical_shell_are_classified_exactly():
    shell = sphere_shell()  # layers end at r = 1.1, 1.3, 1.4, 1.7
    rng = np.random.default_rng(0)
    theta = np.deg2rad(rng.uniform(5, 110, 20000))
    r = rng.uniform(0.3, 2.2, 20000)
    idx, d, valid = shell.locate(r * np.sin(theta), r * np.cos(theta))

    # the signed distance along the normal is just r - R0
    assert np.allclose(d[valid], (r - 1.0)[valid], atol=1e-5)

    rs_hot, rs_cold, fs_cold, fs_hot = shell.layers(idx, d, valid)
    expected = np.select([r < 1.0, r <= 1.1, r <= 1.3, r <= 1.4, r <= 1.7], [0, 1, 2, 3, 4], 0)
    got = np.select([rs_hot, rs_cold, fs_cold, fs_hot], [1, 2, 3, 4], 0)
    assert np.array_equal(expected, got)


def test_shell_ends_at_the_end_of_the_modeled_arc():
    # Points past the last sample of the arc (polar angle > theta_max)
    # are not in the shell, even at radii that would be inside a layer.
    shell = sphere_shell(theta_max_deg=120.0)
    rng = np.random.default_rng(1)
    theta = np.deg2rad(rng.uniform(125, 170, 5000))
    r = rng.uniform(1.0, 1.7, 5000)
    idx, d, valid = shell.locate(r * np.sin(theta), r * np.cos(theta))
    assert sum(m.sum() for m in shell.layers(idx, d, valid)) == 0


def test_points_far_from_the_shell_are_skipped():
    shell = sphere_shell()
    idx, d, valid = shell.locate(np.array([50.0, 0.0]), np.array([0.0, 80.0]))
    assert not valid.any()
    assert np.all(np.isneginf(d))


def test_points_inside_the_reverse_shock_are_in_no_layer():
    shell = sphere_shell()
    idx, d, valid = shell.locate(np.array([0.2, 0.0]), np.array([0.3, 0.0]))
    assert not any(m.any() for m in shell.layers(idx, d, valid))


# ---------------------------------------------------------------------
# Line-of-sight integration, through los_projection_vectorized
# ---------------------------------------------------------------------


class _Props(dict):
    """Shock-property tables as callables of theta, with simple known values."""

    def __init__(self, n_post):
        T = 1e4
        super().__init__(
            n_post=n_post,
            T_post=lambda th: np.full_like(th, T),
            P_post=lambda th: np.full_like(th, 1e-12),
            n_IL=lambda th: np.ones_like(th),
            T_IL_arr=lambda th: np.full_like(th, T),
            regime=lambda th: np.zeros_like(th),
        )


def _run(shell, x, y, rs_n_post, monkeypatch, nz=2000, zmax=4.0):
    """Integrate with H-alpha emissivity replaced by the density itself."""
    monkeypatch.setattr(maps, "emissivity_Halpha", lambda n, T, ion_H=None: n)
    fs_props = _Props(lambda th: np.ones_like(th))
    return maps.los_projection_vectorized(
        x,
        y,
        shell,
        _Props(rs_n_post),
        fs_props,
        inclination=0.0,
        zmax=zmax,
        nz=nz,
        R_stromgren=1e30,
    )["I_Halpha"]


def test_los_path_length_through_a_spherical_shell(monkeypatch):
    # With unit emissivity in every layer, the integral along a line of
    # sight is the chord length through the whole shell, r in [1, 1.7]:
    #   2 * (sqrt(1.7^2 - b^2) - sqrt(max(1 - b^2, 0)))   with b = impact parameter.
    shell = sphere_shell(theta_max_deg=179.0)
    b = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5])
    x, y = b[None, :], np.zeros((1, b.size))
    I = _run(shell, x, y, lambda th: np.ones_like(th), monkeypatch).ravel()

    exact = 2 * (np.sqrt(1.7**2 - b**2) - np.sqrt(np.maximum(1.0 - b**2, 0.0)))
    assert np.allclose(I, exact, rtol=0.03)


def test_line_of_sight_along_the_axis_only_sees_the_modeled_half(monkeypatch):
    # b = 0 runs along the symmetry axis. The front half of the chord
    # (apex side, z in [1, 1.7]) is inside the modeled arc; the back half
    # crosses the shell at polar angle 180 deg, past the end of the arc
    # (theta_max = 179 deg), so it is not part of the shell.
    shell = sphere_shell(theta_max_deg=179.0)
    I = _run(shell, np.array([[0.0]]), np.zeros((1, 1)), lambda th: np.ones_like(th), monkeypatch)
    assert I.item() == pytest.approx(0.7, rel=0.03)


def test_properties_come_from_the_foot_point_not_the_polar_angle(monkeypatch):
    # A flat slab perpendicular to the symmetry axis: the surface is
    # z = z0, its normal is +z everywhere, and the single layer is
    # z0 <= z <= z0 + H. Take the density of the shock at angle theta
    # to be theta itself. A line of sight at cylindrical radius rho
    # crosses the slab at its foot point theta_foot = atan(rho / z0), so
    # the integral must be theta_foot * H. The polar angle of the points
    # inside the slab (atan(rho / z), z in [z0, z0 + H]) is smaller, so
    # looking the density up by polar angle would give a visibly
    # different answer.
    z0, H, rho = 1.0, 0.5, 1.0
    theta = np.linspace(1e-6, np.deg2rad(75.0), 3000)
    rr = z0 / np.cos(theta)  # z = r cos(theta) = z0
    zero = np.zeros_like(theta)
    shell = ShellGeometry(theta, rr, 1.0, H + zero, zero, zero, zero)

    I = _run(shell, np.array([[rho]]), np.zeros((1, 1)), lambda th: th, monkeypatch).item()

    theta_foot = np.arctan(rho / z0)
    assert I == pytest.approx(theta_foot * H, rel=0.02)

    mean_polar = np.mean(np.arctan(rho / np.linspace(z0, z0 + H, 1000)))
    assert (
        abs(mean_polar * H - theta_foot * H) > 0.05 * theta_foot * H
    )  # the test can tell them apart
