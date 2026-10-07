# maps.py
# Functions for projection, radial profiles, and LOS integration

import logging
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter

from bowshockmaps.constants import AU, eV, gamma_ad
from bowshockmaps.paths import IONIZATION_TABLE_FILE
from bowshockmaps.physics.bow_shock_surface import integrate_r_theta_christie
from bowshockmaps.physics.ionization import IonizationTable
from bowshockmaps.physics.normalization import k0_e, k0_p
from bowshockmaps.physics.radiation import (
    emissivity_Halpha,
    emissivity_OIII,
    nu_emissivity_freefree,
    nu_emissivity_sync,
    precompute_gaunt_for_temperatures,
)
from bowshockmaps.physics.shell_geometry import ShellGeometry
from bowshockmaps.physics.thermodynamics import (
    magnetic_field,
    post_shock_conditions,
    vnorm_forward,
    vnorm_wind,
    vtan,
)

logger = logging.getLogger(__name__)

# Samples of the reverse-shock curve used to locate points in normal coordinates
# (see ShellGeometry); neighbouring samples must be much closer than a layer.
N_ARC_SAMPLES = 3000

ion_table = IonizationTable(IONIZATION_TABLE_FILE)


def arcsecond(R, d):
    """
    Convert from physical units to arcseconds.

    Parameters
    ----------
    R : float or array
        Quantity to convert [cm]
    d : float
        Source distance [pc]

    Returns
    -------
    R_arcsec : float or array
        Value in arcseconds
    """
    return (R / AU) / d


def setup_interpolators(theta_grid, R_vals, dR_vals, n_vals):
    """
    Create interpolating functions from tabulated theta grid.

    Builds 1D linear interpolators (via np.interp)

    Parameters
    ----------
    theta_grid : array
        Angle values [rad] at which R_vals, dR_vals, n_vals are tabulated.
    R_vals : array
        Shock distance
    dR_vals : array
    n_vals : array
        Numerical density [cm^-3].

    Returns
    -------
    R_func : callable
        Function R_func(theta)
    dR_func : callable
        Function dR_func(theta)
    n_func : callable
        Function n_func(theta)
    """
    R_func = lambda theta: np.interp(theta, theta_grid, R_vals)
    dR_func = lambda theta: np.interp(theta, theta_grid, dR_vals)
    n_func = lambda theta: np.interp(theta, theta_grid, n_vals)
    return R_func, dR_func, n_func


def get_r_theta_from_christie(lam, n_theta=800):
    """
    Get normalized r(theta) from Christie ODE integration.

    Returns
    -------
    theta_vals : array
        Angle values [rad] (includes theta=0)
    r_vals : array
        Normalized radius values (R/R0)
    r_func : callable
        Interpolating function for r(theta)
    """

    theta_vals, r_vals = integrate_r_theta_christie(
        lam=lam, R0=1.0, theta_max=np.pi - 1e-5, n_theta=n_theta, eps_start=1e-6
    )

    # Apex
    theta_vals = np.insert(theta_vals, 0, 0.0)
    r_vals = np.insert(r_vals, 0, 1.0)

    r_func = interp1d(
        theta_vals,
        r_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(r_vals[0], r_vals[-1]),
    )

    return theta_vals, r_vals, r_func


def precompute_shock_properties(theta_grid, rr_grid, R0_phys, shock, T_IL=8e3, **kwargs):
    """
    Precompute all shock properties on a regular grid for fast interpolation.
    Uses RegularGridInterpolator for much faster evaluation than interp1d.

    Parameters
    ----------
    theta_grid : array
        Angle grid [rad]
    rr_grid : array
        Normalized radius grid (R/R0)
    R0_phys : float
        Physical standoff radius [cm]
    shock : str
        'FS' or 'RS'
    T_IL : float
        Recombination temperature [K]
    **kwargs : dict
        Physical parameters (Mdot, Vw, Vstar, n_ism, lam, etc.)

    Returns
    -------
    props : dict
        Dictionary with RegularGridInterpolator objects for:
        - n_post: Post-shock density [cm^-3]
        - T_post: Post-shock temperature [K]
        - n_IL: Ionization layer density (NaN if adiabatic) [cm^-3]
        - T_IL_arr: Ionization layer temperature (NaN if adiabatic) [K]
        - regime: 1.0 for radiative, 0.0 for adiabatic
        - H_hot: Hot layer thickness [cm]
        - H_cold: Cold layer thickness [cm] (0 for adiabatic)
        - H_total: Total thickness [cm]
        - v_perp: Perpendicular velocity [cm/s]
        - v_tan: Tangential velocity [cm/s]
    """
    # Compute post-shock conditions for all theta
    (
        n_post,
        T_post,
        n_IL,
        T_IL_arr,
        P_post,
        regime,
        H_hot,
        H_cold,
        H_total,
        t_cool,
        t_adv,
    ) = post_shock_conditions(theta_grid, rr_grid, shock, R0_phys, T_IL, **kwargs)

    # Get velocities
    lam = kwargs.get("lam", 0.0)
    if shock == "RS":
        Vw = kwargs.get("Vw")
        v_perp = vnorm_wind(theta_grid, rr_grid, lam, Vw)
        v_tan = vtan(theta_grid, rr_grid, lam, shock="RS", Vw=Vw)
    else:  # FS
        Vstar = kwargs.get("Vstar")
        v_perp = vnorm_forward(theta_grid, rr_grid, lam, Vstar)
        v_tan = vtan(theta_grid, rr_grid, lam, shock="FS", Vstar=Vstar)

    # Convert regime to float for interpolation (1 = radiative, 0 = adiabatic)
    regime_float = np.where(np.array(regime) == "radiative", 1.0, 0.0)

    # Create RegularGridInterpolator
    theta_unique = theta_grid

    props = {
        "n_post": RegularGridInterpolator(
            (theta_unique,), n_post, bounds_error=False, fill_value=None
        ),
        "T_post": RegularGridInterpolator(
            (theta_unique,), T_post, bounds_error=False, fill_value=None
        ),
        "n_IL": RegularGridInterpolator((theta_unique,), n_IL, bounds_error=False, fill_value=None),
        "T_IL_arr": RegularGridInterpolator(
            (theta_unique,), T_IL_arr, bounds_error=False, fill_value=None
        ),
        "P_post": RegularGridInterpolator(
            (theta_unique,), P_post, bounds_error=False, fill_value=None
        ),
        "regime": RegularGridInterpolator(
            (theta_unique,), regime_float, bounds_error=False, fill_value=None
        ),
        "H_hot": RegularGridInterpolator(
            (theta_unique,), H_hot, bounds_error=False, fill_value=None
        ),
        "H_cold": RegularGridInterpolator(
            (theta_unique,), H_cold, bounds_error=False, fill_value=None
        ),
        "H_total": RegularGridInterpolator(
            (theta_unique,), H_total, bounds_error=False, fill_value=None
        ),
        "v_perp": RegularGridInterpolator(
            (theta_unique,), v_perp, bounds_error=False, fill_value=None
        ),
        "v_tan": RegularGridInterpolator(
            (theta_unique,), v_tan, bounds_error=False, fill_value=None
        ),
    }

    return props


def _pixels_to_sample_beam(axis, n, vmin, vmax, distance, fwhm, f_ny, max_pixels):
    """
    Number of pixels along one axis needed to sample the instrumental beam.

    The pixel must be no larger than ``f_ny * fwhm`` for the discrete
    convolution to represent the beam. For a beam much finer than the map
    that can require an unaffordable number of pixels, so the result is
    capped at ``max_pixels`` (never below the requested ``n``). When the
    cap binds, the beam is smaller than a pixel: convolving with it would
    not change the map (a Gaussian much narrower than a pixel is the
    identity on the grid), so it is effectively unresolved, and only the
    conversion to per-beam units still applies. A warning says so.
    """
    d_arcsec = arcsecond((vmax - vmin) / (n - 1), distance)
    d_required = f_ny * fwhm
    if d_arcsec <= d_required:
        return n

    n_new = int(n * d_arcsec / d_required) + 1
    if n_new <= max_pixels:
        warnings.warn(
            f"Map resolution in {axis} is too low for beam size "
            f"(d{axis} = {d_arcsec:.2f} arcsec, required d{axis} <= {d_required:.3g} arcsec). "
            f"Increasing n{axis}: {n} -> {n_new}."
        )
        return n_new

    n_capped = max(n, max_pixels)
    d_capped = d_arcsec * (n - 1) / (n_capped - 1)
    warnings.warn(
        f"Beam too fine for the map in {axis}: FWHM = {fwhm:.3g} arcsec would need "
        f"d{axis} <= {d_required:.3g} arcsec, i.e. n{axis} = {n_new} pixels, more than "
        f"max_pixels = {max_pixels}. Using n{axis} = {n_capped} "
        f"(d{axis} = {d_capped:.3g} arcsec): the beam is smaller than a pixel, so it is "
        f"treated as unresolved -- the map is not smoothed, only converted to per-beam "
        f"units. To resolve it, raise max_pixels (cost grows as nx*ny*nz) or reduce the "
        f"field of view."
    )
    return n_capped


def make_projection_maps(
    xmin,
    xmax,
    ymin,
    ymax,
    nx,
    ny,
    theta_max,
    R_RS_func,
    inclination=0.0,
    PA=0.0,
    zmax=5e15,
    nz=75,
    fwhm_x=3.0,
    fwhm_y=3.0,
    f_ny=0.5,
    lmb=0.0,
    R0_phys=1.0,
    rs_radiative=None,
    fs_radiative=None,
    T_IL=8e3,
    Vstar=None,
    n_ism=None,
    Mdot=None,
    Vw=None,
    wind_regime="hot",
    wind_T_fixed=None,
    f_NTp=0.1,
    f_NTe=0.01,
    p_inj=2.5,
    f_B=0.1,
    R_stromgren=3.086e17,
    nu_ff=2e6 * 1e9,
    distance=224.0,
    convolve=True,
    max_pixels=2000,
):
    """
    Vectorized 2D projected emission maps with pre-computed properties.

    Precomputes shock properties once for all theta, then reuses them
    during LOS integration

    Parameters
    ----------
    xlim, ylim : float
        Map limits [cm]
    nx, ny : int
        Number of pixels in x and y
    R_RS_func : callable
        Function R_RS(theta) giving reverse shock radius [cm]
    inclination : float
        Inclination angle [rad]
    PA : float
        Accepted for API compatibility but intentionally unused: the
        maps are computed in the intrinsic frame (apex along -x) and
        the position angle is applied at display time, by
        `visualization.plot_maps` (Affine2D rotation by PA - 90 deg).
    zmax : float
        Maximum LOS extent [cm]
    nz : int
        Number of LOS integration steps
    fwhm_x, fwhm_y : float
        beam size [arcsec]
    f_ny : float
        Nyquist frequency
    lmb : float
        log10(lambda) parameter
    R0_phys : float
        Physical standoff radius [cm]
    T_IL : float
        Recombination temperature [K]
    Vstar : float
        Stellar velocity [cm/s]
    n_ism : float
        ISM density [cm^-3]
    Mdot : float
        Mass loss rate [g/s]
    Vw : float
        Wind velocity [cm/s]
    wind_regime : str
        'cold', 'hot', or 'fixed'
    wind_T_fixed : float or None
        Fixed wind temperature for 'fixed' regime [K]
    R_stromgren : float
        Stromgren radius [cm]
    nu_ff : float
        Frequency for free-free emission [Hz]
    distance : float
        Source distance [pc]

    max_pixels : int
        Ceiling on the pixels per axis when the grid is refined to sample
        the beam. If a beam is too fine to be sampled within it, the beam
        is treated as unresolved (the map is not smoothed, only converted
        to per-beam units) and a warning says so.

    Returns
    -------
    x_vals, y_vals : arrays
        Coordinate grids [arcsec]
    result : dict
        Emission maps: I_Halpha, I_OIII, I_ff_total, I_ff_mJy, I_syn_total,
        I_syn_mJy, I_continuum_total (I_ff_total+I_syn_total), I_continuum_mJy
    """

    logger.info(f"Beam size: {fwhm_x:.3g} arcsec")

    lam = 10**lmb

    # ==========================
    # Resolution check
    # =========================
    if convolve:
        nx = _pixels_to_sample_beam("x", nx, xmin, xmax, distance, fwhm_x, f_ny, max_pixels)
        ny = _pixels_to_sample_beam("y", ny, ymin, ymax, distance, fwhm_y, f_ny, max_pixels)

    # Measured cost of the line-of-sight integration: ~8 ns per nominal sample
    # (nx * ny * nz) on one core, e.g. 1000 x 1000 x 1000 in ~7 s and
    # 2000 x 2000 x 1000 in ~35 s.
    n_samples = nx * ny * nz
    est_seconds = 8e-9 * n_samples
    if est_seconds > 120.0:
        logger.warning(
            "Large map: %d x %d pixels x %d line-of-sight steps = %.1e samples, roughly "
            "%.0f minutes on one core.",
            nx,
            ny,
            nz,
            n_samples,
            est_seconds / 60.0,
        )

    theta_precomp = np.linspace(1e-6, theta_max, 300)
    rr_precomp = R_RS_func(theta_precomp)  # already normalized

    rs_props = precompute_shock_properties(
        theta_precomp,
        rr_precomp,
        R0_phys,
        "RS",
        T_IL=T_IL,
        Mdot=Mdot,
        Vw=Vw,
        lam=lam,
        wind_regime=wind_regime,
        wind_T_fixed=wind_T_fixed,
    )

    fs_props = precompute_shock_properties(
        theta_precomp,
        rr_precomp,
        R0_phys,
        "FS",
        T_IL=T_IL,
        Vstar=Vstar,
        n_ism=n_ism,
        lam=lam,
    )

    # ==========================================================
    # Shell geometry in normal coordinates.
    #
    # The layer thicknesses are measured along the local normal to the
    # shock surface, so whether a point is inside a layer -- and which
    # part of the shock it belongs to, hence its density and temperature
    # -- is decided from its foot point on the reverse-shock curve and its
    # distance along the normal there, not from its polar angle (see
    # `ShellGeometry`). The curve is sampled densely: neighbouring samples
    # must be much closer than a layer thickness.
    # ==========================================================
    theta_arc = np.linspace(theta_precomp[0], theta_precomp[-1], N_ARC_SAMPLES)
    shell = ShellGeometry(
        theta_arc,
        R_RS_func(theta_arc),
        R0_phys,
        H_RS_hot=rs_props["H_hot"](theta_arc),
        H_RS_cold=rs_props["H_cold"](theta_arc),
        H_FS_cold=fs_props["H_cold"](theta_arc),
        H_FS_hot=fs_props["H_hot"](theta_arc),
    )

    # Generate 2D coordinate grid
    x_vals = np.linspace(xmin, xmax, nx)
    y_vals = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Convert to arcseconds
    x_vals_arcsec = arcsecond(x_vals, distance)
    y_vals_arcsec = arcsecond(y_vals, distance)

    # LOS projection
    result = los_projection_vectorized(
        X,
        Y,
        shell,
        rs_props,
        fs_props,
        inclination=inclination,
        zmax=zmax,
        nz=nz,
        R_stromgren=R_stromgren,
        f_NTp=f_NTp,
        f_NTe=f_NTe,
        p_inj=p_inj,
        f_B=f_B,
        nu_ff=nu_ff,
    )

    if convolve:
        # Instrumental convolution
        result = convolution(
            result,
            x_vals_arcsec,
            y_vals_arcsec,
            fwhm_x=fwhm_x,
            fwhm_y=fwhm_y,
            f_ny=f_ny,
        )

    return x_vals_arcsec, y_vals_arcsec, result


def los_projection_vectorized(
    x,
    y,
    shell,
    rs_props,
    fs_props,
    inclination=0.0,
    zmax=5e15,
    nz=500,
    R_stromgren=3.086e17,
    f_NTp=0.1,
    f_NTe=0.01,
    p_inj=2.5,
    f_B=0.1,
    nu_ff=2e6 * 1e9,
):
    """
    Line-of-sight projection of the emission of an inclined bow-shock shell.

    Correctly handles both adiabatic and radiative regimes:
    - Adiabatic: only hot layer (H_cold = 0, but n_rec/T_rec = n_post/T_post)
    - Radiative: hot layer + cold recombination layer

    Layers, from the star outward, measured along the local normal to the
    shock surface:
    Star -> Wind -> [RS] -> Hot_RS -> Cold_RS -> [CD] -> Cold_FS -> Hot_FS -> [FS] -> ISM
                                (if rad)    (if rad)    (if rad)    (if rad)

    The layers are located in normal coordinates (see `ShellGeometry`): the
    foot point on the reverse-shock curve decides which layer a point is in
    (from its signed distance along the normal) and supplies that layer's
    density, temperature and magnetic field, i.e. those of the part of the
    shock that generated it.

    How the integral is evaluated. The shell is axisymmetric, so the
    emissivity at a point depends only on its (rho, z) -- cylindrical radius
    and height along the symmetry axis. It is therefore evaluated once per
    cell of a 2D (rho, z) table, and each line-of-sight sample just looks its
    cell up. (Locating every sample individually costs ~96% of the run time.)
    Two more savings that change nothing: in the intrinsic frame the map is
    exactly symmetric under y -> -y, so only half of it is computed when the
    grid allows it; and samples farther from the star than the bounding box
    of the shell are skipped, since they cannot be inside it.

    Parameters
    ----------
    x, y : 2D arrays
        Sky-plane coordinate grids [cm], in the intrinsic frame (the
        position angle is applied at display time, not here).
    shell : ShellGeometry
        Reverse-shock curve with the cumulative thickness of the layers.
    rs_props, fs_props : dict
        Shock properties from precompute_shock_properties, as
        interpolators of the angle along the shock (``shell.theta``).
    inclination : float
        Inclination angle [rad]
    zmax : float
        Maximum LOS extent [cm]
    nz : int
        Number of LOS integration steps (over [-zmax, zmax]; only those
        that can reach the shell are used)
    R_stromgren : float
        Stromgren radius [cm]
    nu_ff : float
        Frequency for free-free emission [Hz]

    Returns
    -------
    result : dict
        Integrated intensities: I_Halpha, I_OIII, I_ff_total, I_syn_total,
        I_continuum_total (= I_ff_total + I_syn_total)
    """

    # Precompute gaunt factors
    Z_q = 1.0
    gaunt_lookup = precompute_gaunt_for_temperatures(nu_ff, Z=Z_q)

    ci, si = np.cos(inclination), np.sin(inclination)

    # NOTE: the position angle (PA) is deliberately NOT applied here. The
    # maps are computed in the intrinsic frame (apex along -x) and PA is
    # applied at display time -- see plot_maps.update_map_image /
    # update_map_arrow / update_map_contours (Affine2D rotation by PA-90)
    # and compute_plot_limits. Applying it here as well would rotate
    # everything twice.

    shape_2d = y.shape
    ny_, nx_ = shape_2d

    # LOS grid
    z_all = np.linspace(-zmax, zmax, nz)
    dz = z_all[1] - z_all[0]

    # =========================
    # Shock properties, tabulated along the shell. Entry j belongs to the
    # part of the shock at shell.theta[j]; a line-of-sight sample takes
    # the entry of its foot point.
    theta_tab = shell.theta

    n_post_RS = rs_props["n_post"](theta_tab)
    T_post_RS = rs_props["T_post"](theta_tab)
    P_post_RS = rs_props["P_post"](theta_tab)
    n_IL_RS = rs_props["n_IL"](theta_tab)
    T_IL_RS = rs_props["T_IL_arr"](theta_tab)
    regime_RS = rs_props["regime"](theta_tab)

    n_post_FS = fs_props["n_post"](theta_tab)
    T_post_FS = fs_props["T_post"](theta_tab)
    P_post_FS = fs_props["P_post"](theta_tab)
    n_IL_FS = fs_props["n_IL"](theta_tab)
    T_IL_FS = fs_props["T_IL_arr"](theta_tab)
    regime_FS = fs_props["regime"](theta_tab)

    # =========================
    # Non-thermal distributions normalizatoins
    U_Th_RS = P_post_RS / (gamma_ad - 1.0)
    U_Th_FS = P_post_FS / (gamma_ad - 1.0)

    U_NTp_RS = np.where(regime_RS == 0.0, f_NTp * U_Th_RS, 0.0)
    U_NTp_FS = np.where(regime_FS == 0.0, f_NTp * U_Th_FS, 0.0)

    U_NTe_RS = np.where(regime_RS == 0.0, f_NTe * U_Th_RS, 0.0)
    U_NTe_FS = np.where(regime_FS == 0.0, f_NTe * U_Th_FS, 0.0)

    # NT distribution normalization used for NT emission
    # Proton normalization (k0p_RS/k0p_FS) is computed alongside the
    # electron one but not yet consumed downstream: protons don't
    # contribute meaningfully to synchrotron emission (radiative losses
    # scale as 1/mass^2), so nothing here uses it today. Kept
    # intentionally for a future hadronic-emission channel (e.g.
    # pion-decay gamma-rays) rather than removed.
    k0p_RS = k0_p(U_NTp_RS, p_inj=p_inj, Eminp=1e9 * eV)  # noqa: F841
    k0e_RS = k0_e(U_NTe_RS, p_inj=p_inj, Emine=1e6 * eV)

    k0p_FS = k0_p(U_NTp_FS, p_inj=p_inj, Eminp=1e9 * eV)  # noqa: F841
    k0e_FS = k0_e(U_NTe_FS, p_inj=p_inj, Emine=1e6 * eV)

    # =========================
    # Magnetic field
    U_B_RS = f_B * U_Th_RS
    U_B_FS = f_B * U_Th_FS

    B_RS, B_RS_avg = magnetic_field(U_B_RS)
    B_FS, B_FS_avg = magnetic_field(U_B_FS)

    # Field at the apex (first sample of the curve)
    B_RS_apex, _ = magnetic_field(U_B_RS[:1])
    B_FS_apex, _ = magnetic_field(U_B_FS[:1])

    logger.info(f"Apex magnetic field: B_RS = {B_RS_apex[0]*1e6:.1f} muG")
    logger.info(f"Apex magnetic field: B_FS = {B_FS_apex[0]*1e6:.1f} muG")

    def layer_emissivity(n, T, r_pt, k0e=None, B_avg=None):
        """Emissivities (per unit length) of gas in a layer with density
        `n` and temperature `T`, at distance `r_pt` from the star (all
        arrays of the same length). Synchrotron only where a non-thermal
        electron population `k0e`, `B_avg` is given (the hot layers).
        Returns (H-alpha, [OIII], free-free, free-free [mJy], synchrotron,
        synchrotron [mJy])."""
        # Ionization fractions: full ionization inside the Stromgren
        # sphere, CIE at the layer temperature outside it.
        ion_H = np.ones_like(r_pt)
        ion_O = np.ones_like(r_pt)
        outside_stromgren = r_pt > R_stromgren
        if np.any(outside_stromgren):
            ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(
                T[outside_stromgren]
            )

        e_Ha = emissivity_Halpha(n, T, ion_H=ion_H)
        e_OIII = emissivity_OIII(n, T, ion_H=ion_H, ion_O=ion_O)
        j_ff, j_ff_mJy = nu_emissivity_freefree(
            n,
            T,
            ion_H=ion_H,
            Z_q=Z_q,
            nu=nu_ff,
            gaunt_lookup=gaunt_lookup,
        )
        if k0e is not None:
            j_syn, j_syn_mJy = nu_emissivity_sync(k0e, B_avg, p_inj, nu_ff)
        else:
            j_syn = j_syn_mJy = np.zeros_like(r_pt)
        return e_Ha, e_OIII, j_ff, j_ff_mJy, j_syn, j_syn_mJy

    # =========================
    # 2D (rho, z) EMISSIVITY TABLE
    #
    # Cell size: the line-of-sight sampling (dz, pixel size) already limits
    # how well a layer edge is resolved, so a cell of half the finer of the
    # two adds no error of its own; it is kept within [extent/4000,
    # extent/200] so the table cost stays bounded.
    rho_hi, z_lo, z_hi = shell.bbox
    extent = max(rho_hi, z_hi - z_lo)
    pixel_sizes = []
    if nx_ > 1:
        pixel_sizes.append(abs(x[0, 1] - x[0, 0]))
    if ny_ > 1:
        pixel_sizes.append(abs(y[1, 0] - y[0, 0]))
    cell = 0.5 * min([dz] + pixel_sizes)
    cell = float(np.clip(cell, extent / 4000.0, extent / 200.0))

    n_rho = int(np.ceil(rho_hi / cell))
    n_z = int(np.ceil((z_hi - z_lo) / cell))
    rho_centers = (np.arange(n_rho) + 0.5) * cell

    cell_to_row = np.full(n_rho * n_z, -1, dtype=np.int32)
    stored_cells = []
    stored_values = []
    rows_per_chunk = max(1, int(2e6 // n_rho))
    for k0 in range(0, n_z, rows_per_chunk):
        k1 = min(n_z, k0 + rows_per_chunk)
        z_g = np.repeat(z_lo + (np.arange(k0, k1) + 0.5) * cell, n_rho)
        rho_g = np.tile(rho_centers, k1 - k0)

        idx, d, valid = shell.locate(rho_g, z_g)
        rs_hot, rs_cold, fs_cold, fs_hot = shell.layers(idx, d, valid)

        for mask, which in (
            (rs_hot, "rs_hot"),
            (rs_cold, "rs_cold"),
            (fs_cold, "fs_cold"),
            (fs_hot, "fs_hot"),
        ):
            cells = np.flatnonzero(mask)
            if cells.size == 0:
                continue
            j = idx[cells]
            r_pt = np.hypot(rho_g[cells], z_g[cells])
            if which == "rs_hot":
                em = layer_emissivity(n_post_RS[j], T_post_RS[j], r_pt, k0e_RS[j], B_RS_avg[j])
            elif which == "rs_cold":
                em = layer_emissivity(n_IL_RS[j], T_IL_RS[j], r_pt)
            elif which == "fs_cold":
                em = layer_emissivity(n_IL_FS[j], T_IL_FS[j], r_pt)
            else:
                em = layer_emissivity(n_post_FS[j], T_post_FS[j], r_pt, k0e_FS[j], B_FS_avg[j])
            stored_cells.append(k0 * n_rho + cells)
            stored_values.append(em)

    if stored_cells:
        occupied = np.concatenate(stored_cells)
        cell_to_row[occupied] = np.arange(occupied.size, dtype=np.int32)
        emis = [np.concatenate([v[c] for v in stored_values]) for c in range(6)]
    else:
        emis = [np.zeros(0)] * 6
    e_Ha, e_OIII, e_ff, e_ff_mJy, e_syn, e_syn_mJy = emis

    # =========================
    # LOS INTEGRATION

    # Only the part of each line of sight inside the bounding box of the
    # shell can contribute: there |z| <= hypot(rho_hi, max(|z_lo|, |z_hi|)).
    z_reach = np.hypot(rho_hi, max(abs(z_lo), abs(z_hi)))
    z = z_all[np.abs(z_all) <= z_reach + dz]

    # In the intrinsic frame the map is symmetric under y -> -y (X, Z and
    # rho = hypot(X, Y) do not change). If the grid is symmetric too,
    # compute y >= 0 and mirror.
    y_col = y[:, 0]
    mirror = ny_ > 1 and np.allclose(y_col, -y_col[::-1], rtol=0.0, atol=1e-9 * np.abs(y_col).max())
    j0 = ny_ // 2 if mirror else 0
    x_half = x[j0:, :].ravel()
    y_half = y[j0:, :].ravel()
    n_half = x_half.size

    out = {k: np.zeros(n_half) for k in ("Ha", "OIII", "ff", "ff_mJy", "syn", "syn_mJy")}
    tables = {
        "Ha": e_Ha,
        "OIII": e_OIII,
        "ff": e_ff,
        "ff_mJy": e_ff_mJy,
        "syn": e_syn,
        "syn_mJy": e_syn_mJy,
    }
    inv_cell = 1.0 / cell

    # Pixels are processed in blocks to bound memory at very large maps.
    block = 2_000_000
    for start in range(0, n_half, block):
        xs = x_half[start : start + block]
        ys = y_half[start : start + block]

        for z_i in z:
            # Rotate coordinates: cylindrical radius and height along the
            # symmetry axis of every pixel's sample at this depth
            X_i = ci * xs + si * z_i
            Z_i = -si * xs + ci * z_i
            rho_i = np.hypot(X_i, ys)

            cand = np.flatnonzero((rho_i < rho_hi) & (Z_i >= z_lo) & (Z_i < z_hi))
            if cand.size == 0:
                continue

            jr = np.minimum((rho_i[cand] * inv_cell).astype(np.intp), n_rho - 1)
            jz = np.minimum(((Z_i[cand] - z_lo) * inv_cell).astype(np.intp), n_z - 1)
            row = cell_to_row[jz * n_rho + jr]
            inside = row >= 0
            if not inside.any():
                continue

            pix = start + cand[inside]
            row = row[inside]
            for key, table in tables.items():
                out[key][pix] += table[row] * dz

    def unfold(a_half):
        """Back to the full (ny, nx) map, mirroring y -> -y if it was folded."""
        half = a_half.reshape(ny_ - j0, nx_)
        if not mirror:
            return half
        full = np.empty(shape_2d)
        full[j0:] = half
        full[:j0] = half[ny_ - 1 - np.arange(j0) - j0]
        return full

    I_ff_total = unfold(out["ff"])
    I_ff_mJy = unfold(out["ff_mJy"])
    I_syn_total = unfold(out["syn"])
    I_syn_mJy = unfold(out["syn_mJy"])

    result = {
        "I_Halpha": unfold(out["Ha"]),
        "I_OIII": unfold(out["OIII"]),
        "I_ff_total": I_ff_total,
        "I_ff_mJy": I_ff_mJy,
        "I_syn_total": I_syn_total,
        "I_syn_mJy": I_syn_mJy,
        "I_continuum_total": I_ff_total + I_syn_total,
        "I_continuum_mJy": I_ff_mJy + I_syn_mJy,
    }

    return result


def convolution(result, x_vals_arcsec, y_vals_arcsec, fwhm_x, fwhm_y, f_ny=0.7):
    """
    Convolve maps with a Gaussian instrumental beam.

    Parameters
    ----------
    result : dict
        Dictionary with 2D emission maps pre convolution
    x_vals_arcsec, y_vals_arcsec : arrays
        Coordinate axes [arcsec]
    fwhm_x, fwhm_y : float
        Beam size [arcsec]
    f_ny : float
        Nyquist frequency

    Returns
    -------
    result_conv : dict
        Convolved maps.
        Keys: I_process
        Values: convolved maps; 2D matrices
    """

    # Pixel size [arcsec/pixel]
    dx = np.abs(x_vals_arcsec[1] - x_vals_arcsec[0])
    dy = np.abs(y_vals_arcsec[1] - y_vals_arcsec[0])

    # Convert FWHM to sigma
    sigma_x = fwhm_to_sigma(fwhm_x)
    sigma_y = fwhm_to_sigma(fwhm_y)

    # Convert beam size to pixels
    sigma_x_pix = sigma_x / dx
    sigma_y_pix = sigma_y / dy

    result_conv = {}

    # loop through dicts
    # image -> 2D matrix
    for key, image in result.items():

        conv = gaussian_filter(image, sigma=(sigma_y_pix, sigma_x_pix), mode="constant", cval=0.0)

        # Convert radio map to mJy/beam
        if key in ("I_ff_mJy", "I_syn_mJy", "I_continuum_mJy"):
            A_beam = 2.0 * np.pi * sigma_x * sigma_y  # [arcsec^2]
            conv *= A_beam

        result_conv[key] = conv

    return result_conv


def fwhm_to_sigma(fwhm):
    """
    Converts FWHM to sigma.

    Parameter:
    ----------
    fwhm : float

    Returns:
    --------
    sigma : float
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma


def ionization_fraction(T):
    """
    Ionization fractions from ionization_table.dat

    Parameters
    ----------
    T : float or array
        Gas temperature [K].

    Returns
    -------
    ion_H : float or array
        Ionization fraction of hydrogen
    ion_O : float or array
        Ionization fraction of oxygen

    Both with the same shape as T
    """

    return ion_table.fractions(T)


def radial_profile(x_vals, y_vals, image, dX=0.0, nbins=33, r_min=0.0, r_max=200.0):
    """
    Compute radial profile from a 2D image

    Parameters
    ----------
    x_vals, y_vals : arrays
        Coordinate grids [arcsec]
    image : 2D array
        Image to profile
    dX : float
        Offset in x direction [arcsec]
    nbins : int
        Number of radial bins
    r_min, r_max : float
        Radial range [arcsec]

    Returns
    -------
    r_centers : array
        Bin centers [arcsec]
    profile : array
        Radial profile values
    """
    X, Y = np.meshgrid(x_vals, y_vals)
    R = np.sqrt((X - dX) ** 2 + Y**2)

    radii = np.linspace(r_min, r_max, nbins + 1)
    r_centers = 0.5 * (radii[1:] + radii[:-1])
    profile = np.zeros(nbins)

    for i in range(nbins):
        mask = (R >= radii[i]) & (R < radii[i + 1])
        if np.any(mask):
            profile[i] = np.mean(image[mask])

    return r_centers, profile
