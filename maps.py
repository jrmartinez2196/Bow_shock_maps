# maps.py
# Functions for projection, radial profiles, and LOS integration

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from constants import AU, mu, mp, mu_sh, kB, eV
from bow_shock_surface import integrate_r_theta_christie
from thermodynamics import (
    vnorm_wind, vnorm_forward, vtan, 
    post_shock_conditions
)
from ionization import IonizationTable
from radiation import emissivity_Halpha, emissivity_OIII, emissivity_freefree
from radiation import precompute_gaunt_for_temperatures

from matplotlib.colors import LogNorm

ion_table = IonizationTable("ionization_table.dat")

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
        lam=lam, 
        R0=1.0, 
        theta_max=np.pi - 1e-5,
        n_theta=n_theta,
        eps_start=1e-6
    )
    
    # Apex
    theta_vals = np.insert(theta_vals, 0, 0.0)
    r_vals = np.insert(r_vals, 0, 1.0)
    
    r_func = interp1d(theta_vals, r_vals, 
                      kind='linear',
                      bounds_error=False, 
                      fill_value=(r_vals[0], r_vals[-1]))
    
    return theta_vals, r_vals, r_func


def precompute_shock_properties(theta_grid, rr_grid, R0_phys, shock, T_IL=1e4, **kwargs):
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
        - n_RH: Rankine-Hugoniot density [cm^-3]
        - T_RH: Rankine-Hugoniot temperature [K]
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
    n_RH, T_RH, n_IL, T_IL_arr, regime, H_hot, H_cold, H_total = post_shock_conditions(
        theta_grid, rr_grid, shock, R0_phys, T_IL, **kwargs
    )
    
    # Get velocities
    lam = kwargs.get('lam', 0.)
    if shock == 'RS':
        Vw = kwargs.get('Vw')
        v_perp = vnorm_wind(theta_grid, rr_grid, lam, Vw)
        v_tan = vtan(theta_grid, rr_grid, lam, shock='RS', Vw=Vw)
    else:  # FS
        Vstar = kwargs.get('Vstar')
        v_perp = vnorm_forward(theta_grid, rr_grid, lam, Vstar)
        v_tan = vtan(theta_grid, rr_grid, lam, shock='FS', Vstar=Vstar)
    
    # Convert regime to float for interpolation (1 = radiative, 0 = adiabatic)
    regime_float = np.where(np.array(regime) == 'radiative', 1.0, 0.0)
    
    # Create RegularGridInterpolator
    theta_unique = theta_grid
    
    props = {
        'n_RH': RegularGridInterpolator((theta_unique,), n_RH, bounds_error=False, fill_value=None),
        'T_RH': RegularGridInterpolator((theta_unique,), T_RH, bounds_error=False, fill_value=None),
        'n_IL': RegularGridInterpolator((theta_unique,), n_IL, bounds_error=False, fill_value=None),
        'T_IL_arr': RegularGridInterpolator((theta_unique,), T_IL_arr, bounds_error=False, fill_value=None),
        'regime': RegularGridInterpolator((theta_unique,), regime_float, bounds_error=False, fill_value=None),
        'H_hot': RegularGridInterpolator((theta_unique,), H_hot, bounds_error=False, fill_value=None),
        'H_cold': RegularGridInterpolator((theta_unique,), H_cold, bounds_error=False, fill_value=None),
        'H_total': RegularGridInterpolator((theta_unique,), H_total, bounds_error=False, fill_value=None),
        'v_perp': RegularGridInterpolator((theta_unique,), v_perp, bounds_error=False, fill_value=None),
        'v_tan': RegularGridInterpolator((theta_unique,), v_tan, bounds_error=False, fill_value=None),
    }
    
    return props


def make_projection_maps(xmin, xmax, ymin, ymax, nx, ny, R_RS_func,
                              inclination=0.0, PA = 0.,
                              zmax=5e15, nz=75,
                              fwhm_x = 1.0, fwhm_y = 1.0, f_ny=0.7,
                              lmb=0.0, R0_phys=1.0,
                              rs_radiative=None, fs_radiative=None,
                              T_IL=1e4,
                              Vstar=None, n_ism=None, Mdot=None, Vw=None,
                              wind_regime='hot', wind_T_fixed=None,
                              R_stromgren=3.086e17,
                              nu_ff=2e6*1e9,
                              distance=224.0,
                              convolve=True):
    """
    Vectorized 2D projected emission maps with pre-computed properties.
    
    Precomputes shock properties once for all theta, then reuses them
    during LOS integration. This yields ~50-100x speedup.
    
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
        Emission maps: I_Halpha, I_OIII, I_ff_total
    """
    lam = 10**lmb
    
    theta_precomp = np.linspace(1e-6, np.pi - 1e-6, 300)
    rr_precomp = R_RS_func(theta_precomp) # already normalized
    
    rs_props = precompute_shock_properties(
        theta_precomp, rr_precomp, R0_phys, 'RS', T_IL=T_IL,
        Mdot=Mdot, Vw=Vw, lam=lam,
        wind_regime=wind_regime, wind_T_fixed=wind_T_fixed
    )
    
    fs_props = precompute_shock_properties(
        theta_precomp, rr_precomp, R0_phys, 'FS', T_IL=T_IL,
        Vstar=Vstar, n_ism=n_ism, lam=lam
    )
    
    # Generate 2D coordinate grid
    x_vals = np.linspace(xmin, xmax, nx)
    y_vals = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_vals, y_vals)


    # Rotate according to projected angle
    PA_rot = PA - 90.
    PA_rot_rad = np.deg2rad(PA_rot)
    X_rot = X*np.cos(PA_rot_rad) + Y*np.sin(PA_rot_rad)
    Y_rot = X*np.sin(PA_rot_rad) - Y*np.cos(PA_rot_rad)
    
    # Convert to arcseconds
    x_vals_arcsec = arcsecond(x_vals, distance)
    y_vals_arcsec = arcsecond(y_vals, distance)
    
    # LOS projection
    result = los_projection_vectorized(
        X_rot, Y_rot, R_RS_func, inclination=inclination,
        zmax=zmax, nz=nz,
        lmb=lmb, R0_phys=R0_phys,
        R_stromgren=R_stromgren,
        rs_props=rs_props, fs_props=fs_props,
        nu_ff=nu_ff
    )

    if convolve:
        # Instrumental convolution
        result = convolution(
            result,
            x_vals_arcsec,
            y_vals_arcsec,
            fwhm_x=fwhm_x,
            fwhm_y=fwhm_y,
            f_ny=f_ny
        )
    
    return x_vals_arcsec, y_vals_arcsec, result


def los_projection_vectorized(x, y, R_RS_func, inclination=0.0,
                               zmax=5e15, nz=500, 
                               lmb=0.0, R0_phys=1.0,
                               R_stromgren=3.086e17,
                               rs_props=None, fs_props=None,
                               nu_ff=2e6*1e9):
    """
    Vectorized LOS projection along z-axis for an inclined shell.
    
    Correctly handles both adiabatic and radiative regimes:
    - Adiabatic: only hot layer (H_cold = 0, but n_rec/T_rec = n_RH/T_RH)
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
    nu_ff : float
        Frequency for free-free emission [Hz]
    
    Returns
    -------
    result : dict
        Integrated intensities: I_Halpha, I_OIII, I_ff_total
    """

    # Precompute gaunt factors
    Z_q = 1.0
    gaunt_lookup = precompute_gaunt_for_temperatures(nu_ff, Z=Z_q)

    lam = 10**lmb
    ci, si = np.cos(inclination), np.sin(inclination)
    
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
    
    # Reverse shock radius
    R_RS = R_RS_func(theta) * R0_phys
    
    shape_2d = y.shape
    n_pixels = shape_2d[0] * shape_2d[1]
    
    # Output maps
    I_Halpha = np.zeros(n_pixels)
    I_OIII = np.zeros(n_pixels)
    I_ff_total = np.zeros(n_pixels)
    I_ff_mJy = np.zeros(n_pixels)

    theta_flat = theta.ravel()
    
    # =========================
    # RS PROPERTIES

    H_RS_hot = rs_props['H_hot'](theta_flat).reshape(theta.shape)
    H_RS_cold = rs_props['H_cold'](theta_flat).reshape(theta.shape)
    
    n_RH_RS = rs_props['n_RH'](theta_flat).reshape(theta.shape)
    T_RH_RS = rs_props['T_RH'](theta_flat).reshape(theta.shape)

    n_IL_RS = rs_props['n_IL'](theta_flat).reshape(theta.shape)
    T_IL_RS = rs_props['T_IL_arr'](theta_flat).reshape(theta.shape)

    # =========================
    # FS PROPERTIES

    H_FS_cold = fs_props['H_cold'](theta_flat).reshape(theta.shape)
    H_FS_hot = fs_props['H_hot'](theta_flat).reshape(theta.shape)
    
    n_RH_FS = fs_props['n_RH'](theta_flat).reshape(theta.shape)
    T_RH_FS = fs_props['T_RH'](theta_flat).reshape(theta.shape)

    n_IL_FS = fs_props['n_IL'](theta_flat).reshape(theta.shape)
    T_IL_FS = fs_props['T_IL_arr'](theta_flat).reshape(theta.shape)

    # CD and FS positions
    CD_pos = R_RS + H_RS_hot + H_RS_cold
    FS_pos = CD_pos + H_FS_cold + H_FS_hot

    # =========================
    # LOS INTEGRATION

    for i in range(nz):

        r_i = r[i, :]

        inside_stromgren = r_i <= R_stromgren    # Total ionization
        outside_stromgren = r_i > R_stromgren   # Partial ionization

        R_RS_i = R_RS[i, :]
        CD_pos_i = CD_pos[i, :]
        FS_pos_i = FS_pos[i, :]

        # ==========================================================
        # RS - Hot post shock layer
        # ==========================================================

        inside_hot_rs = ( (r_i >= R_RS_i) & (r_i <= R_RS_i + H_RS_hot[i, :]) )

        ion_H = np.ones_like(r_i)   # Ionization fractions; initialize assuming full ionization as inside the Stromgren sphere
        ion_O = np.ones_like(r_i)

        # Compute only for positions r_i > R_str
        ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(T_RH_RS[i, outside_stromgren]) # In terms of the temperature considering CIE if outside R_str

        I_Halpha += emissivity_Halpha(n_RH_RS[i, :], T_RH_RS[i, :] , ion_H=ion_H) * inside_hot_rs * dz

        I_OIII += emissivity_OIII(n_RH_RS[i, :], T_RH_RS[i, :], ion_H=ion_H, ion_O=ion_O) * inside_hot_rs * dz

        j_ff, j_ff_mJy = emissivity_freefree(n_RH_RS[i, :], T_RH_RS[i, :], ion_H=ion_H, Z_q=Z_q, nu=nu_ff, gaunt_lookup=gaunt_lookup)

        I_ff_total += j_ff * inside_hot_rs * dz
        I_ff_mJy += j_ff_mJy * inside_hot_rs * dz

        # ==========================================================
        # RS - cold post cooling layer (T_IL might be different????)
        # ==========================================================

        if np.any(H_RS_cold[i, :] > 0):

            inside_cold_rs = ( (r_i >= R_RS_i + H_RS_hot[i, :]) & (r_i <= CD_pos_i) & (H_RS_cold[i, :] > 0) )

            ion_H = np.ones_like(r_i)
            ion_O = np.ones_like(r_i)

            ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(T_IL_RS[i, outside_stromgren])
            I_Halpha += emissivity_Halpha(n_IL_RS[i, :], T_IL_RS[i, :], ion_H=ion_H) * inside_cold_rs * dz
            I_OIII += emissivity_OIII(n_IL_RS[i, :], T_IL_RS[i, :], ion_H=ion_H, ion_O=ion_O) * inside_cold_rs * dz

            j_ff, j_ff_mJy = emissivity_freefree(n_IL_RS[i, :], T_IL_RS[i, :], ion_H=ion_H, Z_q=Z_q, nu=nu_ff, gaunt_lookup=gaunt_lookup)

            I_ff_total += j_ff * inside_cold_rs * dz
            I_ff_mJy += j_ff_mJy * inside_cold_rs * dz

        # ==========================================================
        # FORWARD SHOCK - cold post cooling layer (T_IL might be T_ISM if outside R_str????)
        # ==========================================================

        if np.any(H_FS_cold[i, :] > 0):

            inside_cold_fs = ( (r_i >= CD_pos_i) & (r_i <= CD_pos_i + H_FS_cold[i, :]) & (H_FS_cold[i, :] > 0) )

            ion_H = np.ones_like(r_i)
            ion_O = np.ones_like(r_i)

            ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(T_IL_FS[i, outside_stromgren])

            I_Halpha += emissivity_Halpha(n_IL_FS[i, :],T_IL_FS[i, :], ion_H=ion_H) * inside_cold_fs * dz

            I_OIII += emissivity_OIII(n_IL_FS[i, :],T_IL_FS[i, :], ion_H=ion_H, ion_O=ion_O) * inside_cold_fs * dz

            j_ff, j_ff_mJy = emissivity_freefree(n_IL_FS[i, :], T_IL_FS[i, :], ion_H=ion_H, Z_q=Z_q, nu=nu_ff,gaunt_lookup=gaunt_lookup)

            I_ff_total += j_ff * inside_cold_fs * dz
            I_ff_mJy += j_ff_mJy * inside_cold_fs * dz

        # ==========================================================
        # FS - Hot post shock layer
        # ==========================================================

        hot_start = CD_pos_i + H_FS_cold[i, :]

        inside_hot_fs = ( (r_i >= hot_start) & (r_i <= FS_pos_i) )

        ion_H = np.ones_like(r_i)
        ion_O = np.ones_like(r_i)

        ion_H[outside_stromgren], ion_O[outside_stromgren] = ionization_fraction(T_RH_FS[i, outside_stromgren])

        I_Halpha += emissivity_Halpha(n_RH_FS[i, :], T_RH_FS[i, :], ion_H=ion_H) * inside_hot_fs * dz

        I_OIII += emissivity_OIII(n_RH_FS[i, :], T_RH_FS[i, :], ion_H=ion_H, ion_O=ion_O) * inside_hot_fs * dz

        j_ff, j_ff_mJy = emissivity_freefree(n_RH_FS[i, :],T_RH_FS[i, :], ion_H=ion_H, Z_q=Z_q, nu=nu_ff,
            gaunt_lookup=gaunt_lookup)

        I_ff_total += j_ff * inside_hot_fs * dz
        I_ff_mJy += j_ff_mJy * inside_hot_fs * dz

    # =========================
    # RESHAPE
    # =========================

    result = {
        'I_Halpha': I_Halpha.reshape(shape_2d),
        'I_OIII': I_OIII.reshape(shape_2d),
        'I_ff_total': I_ff_total.reshape(shape_2d),
        'I_ff_mJy': I_ff_mJy.reshape(shape_2d)
    }

    return result


def convolution(result, x_vals_arcsec, y_vals_arcsec,
                fwhm_x, fwhm_y,
                f_ny=0.7):
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

    if (dx > f_ny * fwhm_x):
        raise ValueError(f'Map resolution is to low! dx = {dx:.1f} arcsec, allowed dx <= {f_ny*fwhm_x:.1f} arcsec')
    elif (dy > f_ny * fwhm_y):
        raise ValueError(f'Map resolution is to low! dy = {dy:.1f} arcsec, allowed dy <= {f_ny*fwhm_y:.1f} arcsec')

    # Convert beam size to pixels
    sigma_x_pix = sigma_x / dx
    sigma_y_pix = sigma_y / dy

    result_conv = {}

    # loop through dicts
    # image -> 2D matrix
    for key, image in result.items():

        conv = gaussian_filter(image,sigma=(sigma_y_pix, sigma_x_pix), mode='constant', cval=0.)

        # Convert radio map to mJy/beam
        if key == 'I_ff_mJy':
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
    """

    return ion_table.fractions(T)
    

def radial_profile(x_vals, y_vals, image, dX=0.0, nbins=33, r_min=0., r_max=200.):
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
    R = np.sqrt((X - dX)**2 + Y**2)
    
    radii = np.linspace(r_min, r_max, nbins + 1)
    r_centers = 0.5 * (radii[1:] + radii[:-1])
    profile = np.zeros(nbins)
    
    for i in range(nbins):
        mask = (R >= radii[i]) & (R < radii[i+1])
        if np.any(mask):
            profile[i] = np.mean(image[mask])
    
    return r_centers, profile