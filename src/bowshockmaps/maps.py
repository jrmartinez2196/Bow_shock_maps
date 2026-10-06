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
from bowshockmaps.physics.thermodynamics import (
    magnetic_field,
    offset_boundary_along_normal,
    post_shock_conditions,
    vnorm_forward,
    vnorm_wind,
    vtan,
)

logger = logging.getLogger(__name__)

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


def build_layer_boundary_funcs(
    theta_max,
    lam,
    R0_phys,
    T_IL,
    Mdot,
    Vw,
    wind_regime,
    wind_T_fixed,
    Vstar,
    n_ism,
    n_points=300,
    initial_pad=1.2,
    pad_growth=1.5,
    max_pad=4.0,
):
    """
    Build the four layer-boundary functions (RS hot/cold interface, CD,
    FS cold/hot interface, FS), each as boundary(theta) -> physical
    radius [cm], via `offset_boundary_along_normal`.

    Offsetting the RS curve outward along its local normal also shifts
    points to a *smaller* theta the more the surface tilts away from
    radial (see `offset_boundary_along_normal`'s docstring) -- so a
    boundary built from theta in [0, theta_max] generally does not
    itself reach all the way to theta_max; interp1d then has to hold
    it constant (extrapolate) beyond whatever it does reach, which can
    look like an unphysical "flattening" or "flaring" near the edge of
    the model's angular range, especially visible near edge-on
    inclinations where that whole range projects into view.

    To avoid that, build the curve from an *extended* input theta range
    (theta_max * pad, pad > 1) -- genuinely re-integrating the shock
    shape out there, not just evaluating R_RS_func beyond its own
    domain (which would hit the same kind of plateau) -- and grow pad
    adaptively until the resulting boundary actually covers theta_max,
    or until max_pad is reached.

    Parameters
    ----------
    theta_max : float
        The model's nominal angular range [rad]; the boundary functions
        must cover at least this range without falling back to
        constant extrapolation.
    lam, R0_phys, T_IL, Mdot, Vw, wind_regime, wind_T_fixed, Vstar, n_ism :
        Same physical parameters as elsewhere in this module.
    n_points : int
        Number of theta samples per attempt.
    initial_pad, pad_growth, max_pad : float
        Start by integrating out to theta_max*initial_pad; if the
        resulting boundaries don't yet cover theta_max, multiply the
        pad by pad_growth and retry, up to max_pad.

    Returns
    -------
    RS_hot_outer_func, CD_func, FS_cold_outer_func, FS_outer_func : callable
    """

    def _curves_at(theta_max_ext):
        """Build the four (theta_new, r_new) boundary curves from an
        integration out to theta_max_ext. Raises RuntimeError (propagated
        from the ODE solver) if theta_max_ext exceeds the analytic
        bow-shock shape's maximum valid opening angle for this lam."""
        thr_ext_curve, rr_ext_curve = integrate_r_theta_christie(
            lam=lam, R0=1.0, theta_max=theta_max_ext
        )
        r_interp_ext = interp1d(
            thr_ext_curve,
            rr_ext_curve,
            bounds_error=False,
            fill_value=(rr_ext_curve[0], rr_ext_curve[-1]),
        )
        theta_grid_ext = np.linspace(1e-6, theta_max_ext, n_points)
        rr_grid_ext = r_interp_ext(theta_grid_ext)
        R_RS_phys_ext = rr_grid_ext * R0_phys

        rs_props_ext = precompute_shock_properties(
            theta_grid_ext,
            rr_grid_ext,
            R0_phys,
            "RS",
            T_IL=T_IL,
            Mdot=Mdot,
            Vw=Vw,
            lam=lam,
            wind_regime=wind_regime,
            wind_T_fixed=wind_T_fixed,
        )
        fs_props_ext = precompute_shock_properties(
            theta_grid_ext,
            rr_grid_ext,
            R0_phys,
            "FS",
            T_IL=T_IL,
            Vstar=Vstar,
            n_ism=n_ism,
            lam=lam,
        )

        H_RS_hot_ext = rs_props_ext["H_hot"](theta_grid_ext)
        H_RS_cold_ext = rs_props_ext["H_cold"](theta_grid_ext)
        H_FS_cold_ext = fs_props_ext["H_cold"](theta_grid_ext)
        H_FS_hot_ext = fs_props_ext["H_hot"](theta_grid_ext)

        cumulative_offsets = [
            H_RS_hot_ext,
            H_RS_hot_ext + H_RS_cold_ext,
            H_RS_hot_ext + H_RS_cold_ext + H_FS_cold_ext,
            H_RS_hot_ext + H_RS_cold_ext + H_FS_cold_ext + H_FS_hot_ext,
        ]
        return [
            offset_boundary_along_normal(theta_grid_ext, rr_grid_ext, R_RS_phys_ext, H_cum, lam)
            for H_cum in cumulative_offsets
        ]

    # theta is a polar angle measured from the apex: it cannot exceed
    # pi (180 deg) no matter how much padding is requested -- that's a
    # hard geometric ceiling, separate from (and tighter than, in
    # general) whatever opening angle the analytic bow-shock shape's
    # own asymptotic limit allows for a given lam.
    theta_hard_limit = np.pi - 1e-3

    pad = initial_pad
    last_good_curves = None
    curves = None

    while True:
        theta_max_ext = min(theta_max * pad, theta_hard_limit)
        try:
            curves = _curves_at(theta_max_ext)
        except RuntimeError:
            # The analytic bow-shock shape has a maximum valid opening
            # angle for this lam (an asymptotic "Mach cone" angle)
            # beyond which no solution exists -- this pad pushed past
            # it. Fall back to the largest extension that *did* work,
            # or, if even the first (smallest) padding already failed,
            # to the unextended theta_max itself.
            curves = last_good_curves if last_good_curves is not None else _curves_at(theta_max)
            coverage = min(th_new[-1] for th_new, _ in curves)
            logger.warning(
                "Could not extend theta range to %.1f deg for this lam (bow-shock "
                "shape has no solution that far out); layer-boundary curves only "
                "reach theta=%.1f deg (target was theta_max=%.1f deg), using "
                "constant extrapolation beyond that.",
                np.degrees(theta_max_ext),
                np.degrees(coverage),
                np.degrees(theta_max),
            )
            break

        coverage = min(th_new[-1] for th_new, _ in curves)
        at_hard_limit = theta_max_ext >= theta_hard_limit
        if coverage >= theta_max or pad >= max_pad or at_hard_limit:
            if coverage < theta_max:
                logger.warning(
                    "Layer-boundary curves only reach theta=%.1f deg (< theta_max=%.1f "
                    "deg) even after padding input theta to %.1f deg%s; using constant "
                    "extrapolation beyond that.",
                    np.degrees(coverage),
                    np.degrees(theta_max),
                    np.degrees(theta_max_ext),
                    (
                        " (the geometric ceiling: theta cannot exceed 180 deg)"
                        if at_hard_limit
                        else ""
                    ),
                )
            break
        last_good_curves = curves
        pad *= pad_growth

    return tuple(
        interp1d(th_new, r_new, bounds_error=False, fill_value=(r_new[0], r_new[-1]))
        for th_new, r_new in curves
    )


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

    Returns
    -------
    x_vals, y_vals : arrays
        Coordinate grids [arcsec]
    result : dict
        Emission maps: I_Halpha, I_OIII, I_ff_total, I_ff_mJy, I_syn_total,
        I_syn_mJy, I_continuum_total (I_ff_total+I_syn_total), I_continuum_mJy
    """

    logger.info(f"Beam size: {fwhm_x:.1f} arcsec")

    lam = 10**lmb

    # ==========================
    # Resolution check
    # =========================
    if convolve:
        dx_phys = (xmax - xmin) / (nx - 1)
        dy_phys = (ymax - ymin) / (ny - 1)

        dx_arcsec = arcsecond(dx_phys, distance)
        dy_arcsec = arcsecond(dy_phys, distance)

        if dx_arcsec > f_ny * fwhm_x:
            nx_new = int(nx * dx_arcsec / (f_ny * fwhm_x)) + 1
            warnings.warn(
                f"Map resolution in x is too low for beam size "
                f"(dx = {dx_arcsec:.2f} arcsec, required dx <= {f_ny*fwhm_x:.2f} arcsec). "
                f"Increasing nx: {nx} -> {nx_new}."
            )
            nx = nx_new

        if dy_arcsec > f_ny * fwhm_y:
            ny_new = int(ny * dy_arcsec / (f_ny * fwhm_y)) + 1
            warnings.warn(
                f"Map resolution in y is too low for beam size "
                f"(dy = {dy_arcsec:.2f} arcsec, required dy <= {f_ny*fwhm_y:.2f} arcsec). "
                f"Increasing ny: {ny} -> {ny_new}."
            )
            ny = ny_new

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
    # Layer-boundary curves (RS_hot_outer, CD, FS_cold_outer, FS),
    # offset along the local shock normal instead of added directly
    # to the radial coordinate R_RS(theta) at fixed theta -- the bow
    # shock isn't spherically symmetric, so away from the apex those
    # are not the same thing (see `offset_boundary_along_normal`).
    # Built from an adaptively-extended theta range so each boundary
    # actually covers [0, theta_max] (see `build_layer_boundary_funcs`),
    # then reused for every pixel/LOS-step via interpolation, same as
    # R_RS_func.
    # ==========================================================
    RS_hot_outer_func, CD_func, FS_cold_outer_func, FS_outer_func = build_layer_boundary_funcs(
        theta_max=theta_max,
        lam=lam,
        R0_phys=R0_phys,
        T_IL=T_IL,
        Mdot=Mdot,
        Vw=Vw,
        wind_regime=wind_regime,
        wind_T_fixed=wind_T_fixed,
        Vstar=Vstar,
        n_ism=n_ism,
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
        R_RS_func,
        inclination=inclination,
        zmax=zmax,
        nz=nz,
        lmb=lmb,
        R0_phys=R0_phys,
        R_stromgren=R_stromgren,
        rs_props=rs_props,
        fs_props=fs_props,
        RS_hot_outer_func=RS_hot_outer_func,
        CD_func=CD_func,
        FS_cold_outer_func=FS_cold_outer_func,
        FS_outer_func=FS_outer_func,
        theta_bounds=(theta_precomp[0], theta_precomp[-1]),
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
    R_RS_func,
    inclination=0.0,
    zmax=5e15,
    nz=500,
    lmb=0.0,
    R0_phys=1.0,
    R_stromgren=3.086e17,
    rs_props=None,
    fs_props=None,
    RS_hot_outer_func=None,
    CD_func=None,
    FS_cold_outer_func=None,
    FS_outer_func=None,
    theta_bounds=(1e-6, np.deg2rad(120.0)),
    f_NTp=0.1,
    f_NTe=0.01,
    p_inj=2.5,
    f_B=0.1,
    nu_ff=2e6 * 1e9,
):
    """
    Vectorized LOS projection along z-axis for an inclined shell.

    Correctly handles both adiabatic and radiative regimes:
    - Adiabatic: only hot layer (H_cold = 0, but n_rec/T_rec = n_post/T_post)
    - Radiative: hot layer + cold recombination layer

    Radial structure from star outward:
    Star -> Wind -> [RS] -> Hot_RS -> Cold_RS -> [CD] -> Cold_FS -> Hot_FS -> [FS] -> ISM
                                (if rad)    (if rad)    (if rad)    (if rad)

    Parameters
    ----------
    x, y : 2D arrays
        Coordinate grids [cm]
    R_RS_func : callable
        Function R_RS(theta) giving reverse shock radius [cm]
    inclination : float
        Inclination angle [rad]
    zmax : float
        Maximum LOS extent [cm]
    nz : int
        Number of LOS integration steps
    lmb : float
        log10(lambda) parameter
    R0_phys : float
        Physical standoff radius [cm]
    R_stromgren : float
        Stromgren radius [cm]
    rs_props, fs_props : dict
        Precomputed shock properties from precompute_shock_properties
    RS_hot_outer_func, CD_func, FS_cold_outer_func, FS_outer_func : callable
        boundary(theta) -> physical radius [cm] for each layer boundary
        (RS hot/cold interface, contact discontinuity, FS cold/hot
        interface, forward shock), built by offsetting the RS curve
        along its local normal (see `offset_boundary_along_normal`)
        rather than added directly to R_RS(theta) at fixed theta.
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

    # LOS grid
    z = np.linspace(-zmax, zmax, nz)
    dz = z[1] - z[0]

    x_flat = x.ravel()[None, :]
    y_flat = y.ravel()[None, :]
    z_grid = z[:, None]

    # Rotate coordinates
    X = ci * x_flat + si * z_grid
    Y = y_flat
    Z = -si * x_flat + ci * z_grid

    # Spherical coordinates
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(np.clip(Z / np.where(r == 0, 1, r), -1, 1))
    theta_flat = theta.ravel()
    theta_flat = np.clip(theta_flat, theta_bounds[0], theta_bounds[1])

    # Reverse shock radius
    R_RS = R_RS_func(theta) * R0_phys

    shape_2d = y.shape
    n_pixels = shape_2d[0] * shape_2d[1]

    # Output maps
    I_Halpha = np.zeros(n_pixels)
    I_OIII = np.zeros(n_pixels)
    I_ff_total = np.zeros(n_pixels)
    I_ff_mJy = np.zeros(n_pixels)
    I_syn_total = np.zeros(n_pixels)
    I_syn_mJy = np.zeros(n_pixels)

    # =========================
    # RS PROPERTIES

    H_RS_cold = rs_props["H_cold"](theta_flat).reshape(theta.shape)

    n_post_RS = rs_props["n_post"](theta_flat).reshape(theta.shape)
    T_post_RS = rs_props["T_post"](theta_flat).reshape(theta.shape)
    P_post_RS = rs_props["P_post"](theta_flat).reshape(theta.shape)

    n_IL_RS = rs_props["n_IL"](theta_flat).reshape(theta.shape)
    T_IL_RS = rs_props["T_IL_arr"](theta_flat).reshape(theta.shape)

    # =========================
    # FS PROPERTIES

    H_FS_cold = fs_props["H_cold"](theta_flat).reshape(theta.shape)

    n_post_FS = fs_props["n_post"](theta_flat).reshape(theta.shape)
    T_post_FS = fs_props["T_post"](theta_flat).reshape(theta.shape)
    P_post_FS = fs_props["P_post"](theta_flat).reshape(theta.shape)

    n_IL_FS = fs_props["n_IL"](theta_flat).reshape(theta.shape)
    T_IL_FS = fs_props["T_IL_arr"](theta_flat).reshape(theta.shape)

    # =========================
    # Non-thermal distributions normalizatoins
    U_Th_RS = P_post_RS / (gamma_ad - 1.0)
    U_Th_FS = P_post_FS / (gamma_ad - 1.0)

    regime_RS = rs_props["regime"](theta_flat).reshape(theta.shape)
    regime_FS = fs_props["regime"](theta_flat).reshape(theta.shape)

    U_NTp_RS = np.where(regime_RS == 0.0, f_NTp * U_Th_RS, 0.0)
    U_NTp_FS = np.where(regime_FS == 0.0, f_NTp * U_Th_FS, 0.0)

    U_NTe_RS = np.where(regime_RS == 0.0, f_NTe * U_Th_RS, 0.0)
    U_NTe_FS = np.where(regime_FS == 0.0, f_NTe * U_Th_FS, 0.0)

    # NT distribution normalization used for NT emission
    k0p_RS = k0_p(U_NTp_RS, p_inj=p_inj, Eminp=1e9 * eV)
    k0e_RS = k0_e(U_NTe_RS, p_inj=p_inj, Emine=1e6 * eV)

    k0p_FS = k0_p(U_NTp_FS, p_inj=p_inj, Eminp=1e9 * eV)
    k0e_FS = k0_e(U_NTe_FS, p_inj=p_inj, Emine=1e6 * eV)

    # =========================
    # Magnetic field
    U_B_RS = f_B * U_Th_RS
    U_B_FS = f_B * U_Th_FS

    B_RS, B_RS_avg = magnetic_field(U_B_RS)
    B_FS, B_FS_avg = magnetic_field(U_B_FS)

    theta_apex = np.array([theta_bounds[0]])
    U_B_RS_apex = f_B * rs_props["P_post"](theta_apex) / (gamma_ad - 1.0)
    U_B_FS_apex = f_B * fs_props["P_post"](theta_apex) / (gamma_ad - 1.0)

    B_RS_apex, _ = magnetic_field(U_B_RS_apex)
    B_FS_apex, _ = magnetic_field(U_B_FS_apex)

    logger.info(f"Apex magnetic field: B_RS = {B_RS_apex[0]*1e6:.1f} muG")
    logger.info(f"Apex magnetic field: B_FS = {B_FS_apex[0]*1e6:.1f} muG")

    # =========================
    # Layer boundary positions.
    #
    # These are NOT simply R_RS + (cumulative thickness) at fixed
    # theta: the bow shock isn't spherically symmetric, so a thickness
    # measured along the local shock normal only maps onto a radial
    # distance that way at the apex. RS_hot_outer_func/CD_func/
    # FS_cold_outer_func/FS_outer_func already account for this (see
    # offset_boundary_along_normal / make_projection_maps).
    RS_hot_outer = RS_hot_outer_func(theta_flat).reshape(theta.shape)
    CD_pos = CD_func(theta_flat).reshape(theta.shape)
    FS_cold_outer = FS_cold_outer_func(theta_flat).reshape(theta.shape)
    FS_pos = FS_outer_func(theta_flat).reshape(theta.shape)

    # Each boundary is built from an independent offset-and-reparametrize
    # pass (see offset_boundary_along_normal), so near-degenerate regions
    # (e.g. theta ~ 0, where the local normal direction itself is
    # ill-defined, or where a layer's thickness ~ 0 and two boundaries
    # should nearly coincide) can leave tiny numerical crossings. The
    # physical layers are strictly nested by construction (each boundary
    # is always farther from the star than the previous one), so enforce
    # that explicitly rather than let interpolation noise violate it.
    RS_hot_outer = np.maximum(RS_hot_outer, R_RS)
    CD_pos = np.maximum(CD_pos, RS_hot_outer)
    FS_cold_outer = np.maximum(FS_cold_outer, CD_pos)
    FS_pos = np.maximum(FS_pos, FS_cold_outer)

    # =========================
    # LOS INTEGRATION

    for i in range(nz):

        r_i = r[i, :]

        outside_stromgren = r_i > R_stromgren  # Partial ionization

        R_RS_i = R_RS[i, :]
        RS_hot_outer_i = RS_hot_outer[i, :]
        CD_pos_i = CD_pos[i, :]
        FS_cold_outer_i = FS_cold_outer[i, :]
        FS_pos_i = FS_pos[i, :]

        # ==========================================================
        # RS - Hot post shock layer
        # ==========================================================

        inside_hot_rs = (r_i >= R_RS_i) & (r_i <= RS_hot_outer_i) & (theta[i, :] <= theta_bounds[1])

        ion_H = np.ones_like(
            r_i
        )  # Ionization fractions; initialize assuming full ionization as inside the Stromgren sphere
        ion_O = np.ones_like(r_i)

        # Compute only for positions r_i > R_str
        ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(
            T_post_RS[i, outside_stromgren]
        )  # In terms of the temperature considering CIE if outside R_str

        I_Halpha += (
            emissivity_Halpha(n_post_RS[i, :], T_post_RS[i, :], ion_H=ion_H) * inside_hot_rs * dz
        )

        I_OIII += (
            emissivity_OIII(n_post_RS[i, :], T_post_RS[i, :], ion_H=ion_H, ion_O=ion_O)
            * inside_hot_rs
            * dz
        )

        j_ff, j_ff_mJy = nu_emissivity_freefree(
            n_post_RS[i, :],
            T_post_RS[i, :],
            ion_H=ion_H,
            Z_q=Z_q,
            nu=nu_ff,
            gaunt_lookup=gaunt_lookup,
        )

        j_syn, j_syn_mJy = nu_emissivity_sync(k0e_RS[i, :], B_RS_avg[i, :], p_inj, nu_ff)

        I_ff_total += j_ff * inside_hot_rs * dz
        I_ff_mJy += j_ff_mJy * inside_hot_rs * dz

        I_syn_total += j_syn * inside_hot_rs * dz
        I_syn_mJy += j_syn_mJy * inside_hot_rs * dz

        # ==========================================================
        # RS - cold post cooling layer (T_IL might be different????)
        # ==========================================================

        if np.any(H_RS_cold[i, :] > 0):

            inside_cold_rs = (
                (r_i >= RS_hot_outer_i)
                & (r_i <= CD_pos_i)
                & (H_RS_cold[i, :] > 0)
                & (theta[i, :] <= theta_bounds[1])
            )

            ion_H = np.ones_like(r_i)
            ion_O = np.ones_like(r_i)

            ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(
                T_IL_RS[i, outside_stromgren]
            )
            I_Halpha += (
                emissivity_Halpha(n_IL_RS[i, :], T_IL_RS[i, :], ion_H=ion_H) * inside_cold_rs * dz
            )
            I_OIII += (
                emissivity_OIII(n_IL_RS[i, :], T_IL_RS[i, :], ion_H=ion_H, ion_O=ion_O)
                * inside_cold_rs
                * dz
            )

            j_ff, j_ff_mJy = nu_emissivity_freefree(
                n_IL_RS[i, :],
                T_IL_RS[i, :],
                ion_H=ion_H,
                Z_q=Z_q,
                nu=nu_ff,
                gaunt_lookup=gaunt_lookup,
            )

            I_ff_total += j_ff * inside_cold_rs * dz
            I_ff_mJy += j_ff_mJy * inside_cold_rs * dz

        # ==========================================================
        # FORWARD SHOCK - cold post cooling layer (T_IL might be T_ISM if outside R_str????)
        # ==========================================================

        if np.any(H_FS_cold[i, :] > 0):

            inside_cold_fs = (
                (r_i >= CD_pos_i)
                & (r_i <= FS_cold_outer_i)
                & (H_FS_cold[i, :] > 0)
                & (theta[i, :] <= theta_bounds[1])
            )

            ion_H = np.ones_like(r_i)
            ion_O = np.ones_like(r_i)

            ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(
                T_IL_FS[i, outside_stromgren]
            )

            I_Halpha += (
                emissivity_Halpha(n_IL_FS[i, :], T_IL_FS[i, :], ion_H=ion_H) * inside_cold_fs * dz
            )

            I_OIII += (
                emissivity_OIII(n_IL_FS[i, :], T_IL_FS[i, :], ion_H=ion_H, ion_O=ion_O)
                * inside_cold_fs
                * dz
            )

            j_ff, j_ff_mJy = nu_emissivity_freefree(
                n_IL_FS[i, :],
                T_IL_FS[i, :],
                ion_H=ion_H,
                Z_q=Z_q,
                nu=nu_ff,
                gaunt_lookup=gaunt_lookup,
            )

            I_ff_total += j_ff * inside_cold_fs * dz
            I_ff_mJy += j_ff_mJy * inside_cold_fs * dz

        # ==========================================================
        # FS - Hot post shock layer
        # ==========================================================

        hot_start = FS_cold_outer_i

        inside_hot_fs = (r_i >= hot_start) & (r_i <= FS_pos_i) & (theta[i, :] <= theta_bounds[1])

        ion_H = np.ones_like(r_i)
        ion_O = np.ones_like(r_i)

        ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(
            T_post_FS[i, outside_stromgren]
        )

        I_Halpha += (
            emissivity_Halpha(n_post_FS[i, :], T_post_FS[i, :], ion_H=ion_H) * inside_hot_fs * dz
        )

        I_OIII += (
            emissivity_OIII(n_post_FS[i, :], T_post_FS[i, :], ion_H=ion_H, ion_O=ion_O)
            * inside_hot_fs
            * dz
        )

        j_ff, j_ff_mJy = nu_emissivity_freefree(
            n_post_FS[i, :],
            T_post_FS[i, :],
            ion_H=ion_H,
            Z_q=Z_q,
            nu=nu_ff,
            gaunt_lookup=gaunt_lookup,
        )

        I_ff_total += j_ff * inside_hot_fs * dz
        I_ff_mJy += j_ff_mJy * inside_hot_fs * dz

        j_syn, j_syn_mJy = nu_emissivity_sync(k0e_FS[i, :], B_FS_avg[i, :], p_inj, nu_ff)

        I_syn_total += j_syn * inside_hot_fs * dz
        I_syn_mJy += j_syn_mJy * inside_hot_fs * dz

    # =========================
    # RESHAPE
    # =========================

    result = {
        "I_Halpha": I_Halpha.reshape(shape_2d),
        "I_OIII": I_OIII.reshape(shape_2d),
        "I_ff_total": I_ff_total.reshape(shape_2d),
        "I_ff_mJy": I_ff_mJy.reshape(shape_2d),
        "I_syn_total": I_syn_total.reshape(shape_2d),
        "I_syn_mJy": I_syn_mJy.reshape(shape_2d),
        "I_continuum_total": (I_ff_total + I_syn_total).reshape(shape_2d),
        "I_continuum_mJy": (I_ff_mJy + I_syn_mJy).reshape(shape_2d),
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
