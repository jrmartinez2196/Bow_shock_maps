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


# ---------------------------------------------------------------------
# Grid refinement to sample the instrumental beam
# ---------------------------------------------------------------------


def _pix(n, fwhm, max_pixels, f_ny=0.5):
    # 1 cm == 1 arcsec at distance 206265 / pi / ... keep it simple: use
    # arcsecond() itself on a 100-unit-wide field.
    return maps._pixels_to_sample_beam("x", n, 0.0, 100.0, 1.0, fwhm, f_ny, max_pixels)


def test_grid_is_left_alone_when_it_already_samples_the_beam():
    d = maps.arcsecond(100.0 / 99, 1.0)  # pixel size [arcsec] for n=100
    assert _pix(100, fwhm=10 * d, max_pixels=1000) == 100


def test_grid_is_refined_to_sample_a_finer_beam():
    d = maps.arcsecond(100.0 / 99, 1.0)
    with pytest.warns(UserWarning, match="Increasing"):
        n_new = _pix(100, fwhm=d / 2, max_pixels=1000)  # needs pixel <= d/4
    assert 100 < n_new <= 1000
    pixel = maps.arcsecond(100.0 / (n_new - 1), 1.0)
    assert pixel <= 0.5 * (d / 2) * 1.01  # samples the beam


def test_refinement_is_capped_and_says_the_beam_is_unresolved():
    d = maps.arcsecond(100.0 / 99, 1.0)
    with pytest.warns(UserWarning, match="treated as unresolved"):
        n_new = _pix(100, fwhm=d * 1e-6, max_pixels=250)
    assert n_new == 250


def test_cap_never_reduces_the_requested_resolution():
    d = maps.arcsecond(100.0 / 599, 1.0)
    with pytest.warns(UserWarning, match="treated as unresolved"):
        n_new = _pix(600, fwhm=d * 1e-6, max_pixels=250)
    assert n_new == 600


def test_a_beam_much_smaller_than_a_pixel_only_changes_the_units():
    # This is what justifies treating a capped (unresolved) beam as "no
    # smoothing": a Gaussian far narrower than a pixel is the identity on
    # the grid. The only effect left is the conversion of the radio maps
    # to per-beam units, with the true beam area.
    rng = np.random.default_rng(0)
    img = rng.uniform(1.0, 2.0, (30, 30))
    axis = np.arange(30.0)  # 1 arcsec pixels
    fwhm = 1e-4  # arcsec

    out = maps.convolution({"I_Halpha": img, "I_ff_mJy": img}, axis, axis, fwhm, fwhm)

    assert np.allclose(out["I_Halpha"], img, rtol=1e-12)
    sigma = maps.fwhm_to_sigma(fwhm)
    assert np.allclose(out["I_ff_mJy"], img * 2 * np.pi * sigma * sigma, rtol=1e-12)


# ---------------------------------------------------------------------
# The speed-ups in los_projection_vectorized must not change the result
# ---------------------------------------------------------------------


def _full_run(shell, x, y, monkeypatch, zmax=4.0, nz=801):
    monkeypatch.setattr(maps, "emissivity_Halpha", lambda n, T, ion_H=None: n)
    props = _Props(lambda th: np.ones_like(th))
    return maps.los_projection_vectorized(
        x, y, shell, props, _Props(lambda th: np.ones_like(th)),
        inclination=0.6, zmax=zmax, nz=nz, R_stromgren=1e30,
    )["I_Halpha"]  # fmt: skip


# The integral is a Riemann sum with step dz, so two evaluations that are
# the same up to rounding can still differ by one step if a sample lands
# exactly on a layer surface. Compare to within that.
DZ = 8.0 / 800


def test_y_mirror_symmetry_shortcut_gives_the_same_map(monkeypatch):
    # In the intrinsic frame the map is exactly symmetric under y -> -y.
    # With a symmetric grid only half is computed and mirrored; shifting
    # the grid by a hair turns that off, and both must agree. (Grid values
    # avoid sitting exactly on a layer radius: 1, 1.1, 1.3, 1.4, 1.7.)
    shell = sphere_shell(theta_max_deg=170.0)
    X, Y = np.meshgrid(np.linspace(-1.55, 1.55, 9), np.linspace(-1.55, 1.55, 8))

    folded = _full_run(shell, X, Y, monkeypatch)
    unfolded = _full_run(shell, X, Y + 1e-7, monkeypatch)  # no longer exactly symmetric

    assert np.allclose(folded, unfolded, rtol=1e-3, atol=2 * DZ)
    assert np.array_equal(folded, folded[::-1])  # exactly mirrored


def test_y_mirror_symmetry_shortcut_with_an_odd_number_of_rows(monkeypatch):
    shell = sphere_shell(theta_max_deg=170.0)
    X, Y = np.meshgrid(np.linspace(-1.55, 1.55, 7), np.linspace(-1.55, 1.55, 7))  # includes y = 0

    folded = _full_run(shell, X, Y, monkeypatch)
    unfolded = _full_run(shell, X, Y + 1e-7, monkeypatch)

    assert np.allclose(folded, unfolded, rtol=1e-3, atol=2 * DZ)
    assert np.array_equal(folded, folded[::-1])


def test_skipping_samples_outside_the_shell_box_changes_nothing(monkeypatch):
    # Only the part of each line of sight inside the bounding box of the
    # shell is integrated. Doubling zmax with the same step dz adds only
    # samples outside it, so the result must not change.
    shell = sphere_shell(theta_max_deg=170.0)
    X, Y = np.meshgrid(np.linspace(-1.55, 1.55, 7), np.linspace(-1.55, 1.55, 6))

    near = _full_run(shell, X, Y, monkeypatch, zmax=4.0, nz=801)  # dz = 0.01
    far = _full_run(shell, X, Y, monkeypatch, zmax=8.0, nz=1601)  # same dz

    assert np.allclose(near, far, rtol=1e-9, atol=0.0)


# ---------------------------------------------------------------------
# Grid derived from the shell: sky extent, line-of-sight reach, layer scale
# ---------------------------------------------------------------------


def _brute_force_extent(shell, inclination, n_phi=720):
    """Sky extent and line-of-sight reach from points sampled all over the
    surface of revolution (every boundary of every layer, every azimuth)."""
    ci, si = np.cos(inclination), np.sin(inclination)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    xs, ys, zs = [], [], []
    for d in (0.0 * shell.d1, shell.d1, shell.d2, shell.d3, shell.d4):
        rho = shell.C_rho + d * shell.n_rho
        z = shell.C_z + d * shell.n_z
        X = rho[:, None] * np.cos(phi)[None, :]
        Y = rho[:, None] * np.sin(phi)[None, :]
        Z = np.broadcast_to(z[:, None], X.shape)
        xs.append(ci * X - si * Z)
        ys.append(Y)
        zs.append(si * X + ci * Z)
    x, y, z_los = (np.concatenate([a.ravel() for a in lst]) for lst in (xs, ys, zs))
    return min(x.min(), 0.0), max(x.max(), 0.0), np.abs(y).max(), np.abs(z_los).max()


@pytest.mark.parametrize("inclination_deg", [0.0, 15.0, 45.0, 75.0, 90.0])
@pytest.mark.parametrize("theta_max_deg", [90.0, 135.0])
def test_sky_extent_and_los_reach_match_brute_force(inclination_deg, theta_max_deg):
    shell = sphere_shell(theta_max_deg=theta_max_deg)
    inc = np.deg2rad(inclination_deg)
    x_min, x_max, y_half = shell.sky_extent(inc)
    reach = shell.los_reach(inc)
    bx_min, bx_max, by, bz = _brute_force_extent(shell, inc)
    assert x_min == pytest.approx(bx_min, abs=2e-3)
    assert x_max == pytest.approx(bx_max, abs=2e-3)
    assert y_half == pytest.approx(by, abs=2e-3)
    assert reach == pytest.approx(bz, abs=2e-3)


def test_sky_extent_of_a_full_sphere_is_its_radius_for_every_inclination():
    # A full sphere of outer radius 1.7 looks like a disk of radius 1.7
    # from every direction, and the line of sight needs |z| <= 1.7.
    shell = sphere_shell(theta_max_deg=179.99)
    for inclination in np.deg2rad([0.0, 30.0, 60.0, 90.0]):
        x_min, x_max, y_half = shell.sky_extent(inclination)
        assert (x_min, x_max, y_half) == pytest.approx((-1.7, 1.7, 1.7), abs=2e-3)
        assert shell.los_reach(inclination) == pytest.approx(1.7, abs=2e-3)


def test_thinnest_layer_ignores_a_layer_that_carries_almost_no_emission():
    # Layers 0.5, 0.05, 0.2, 0.001 thick (in R0). The 0.001 one is far
    # thinner, but with the same density its emission measure is a tiny
    # fraction of the total, so it must not set the scale (this is
    # BD+43's forward-shock hot layer).
    shell = sphere_shell(thicknesses=(0.5, 0.05, 0.2, 0.001))
    ones = np.ones_like(shell.theta)
    assert shell.thinnest_layer((ones, ones, ones, ones)) == pytest.approx(0.05)


def test_thinnest_layer_counts_a_thin_layer_that_does_carry_emission():
    # Same layers, but the thin one is 100x denser: n^2 makes it matter.
    shell = sphere_shell(thicknesses=(0.5, 0.05, 0.2, 0.001))
    ones = np.ones_like(shell.theta)
    assert shell.thinnest_layer((ones, ones, ones, 100.0 * ones)) == pytest.approx(0.001)


def test_thinnest_layer_is_none_for_a_shell_without_layers():
    shell = sphere_shell(thicknesses=(0.0, 0.0, 0.0, 0.0))
    ones = np.ones_like(shell.theta)
    assert shell.thinnest_layer((ones, ones, ones, ones)) is None
