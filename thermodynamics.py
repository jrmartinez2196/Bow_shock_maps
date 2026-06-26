# thermodynamics.py
import numpy as np
from constants import mp, kB, mu, mu_sh, gamma_ad, AU

# ============================================================
# Geometric function
# ============================================================

def AA(thr, rr, lam=0.):
    """
    Calculates the components of the unit perpendicular and tangential vectors
    to the termination shock.
    Eqs. 13 and 14 from Christie+ 2016.

    Parameters
    ----------
    thr : float or array
        Angle from the apex [rad].
    rr : float or array
        Normalized distance to the star r/R0.
    lam : float
        lambda = alpha/(1+alpha), alpha being P_th_med/P_kin_med

    Returns
    -------
    Aomega : float or array
    Az : float or array
    The unit vector tangential and perpendicular to the bow shock surface are:
    n_t = Aomega * e_omega + Az * e_z
    n_perp = -Az * e_omega + Aomega* e_z
    """
    s = np.sin(thr)
    c = np.cos(thr)
    fl = np.sqrt((thr - s * c)**2 * (1 - lam * rr * rr)**2 + (s**4 * (1 - rr * rr)**2))
    Aomega = (thr - s * c) * (1 - lam * rr * rr) / fl
    Az = s**2 * (1 - rr * rr) / fl
    return Aomega, Az


# ============================================================
# dL_dAperp function
# ============================================================

def dL_dAperp(R_phys, theta, sin_alpha):
    """
    Calculates the length of a segment along the surface while increasing theta
    and the area of the annulus perpendicular to the upstream flow.
    
    Parameters:
    -----------
    R_phys : array
        Distance to the bow shock [cm]
    theta : array
        Angle from the apex [rad]
    sin_alpha : array
        sin(alpha) where alpha is the angle between surface normal and upstream flow
    
    Returns:
    --------
    dL : array
        Segment length [cm]
    dA_perp : array
        Perpendicular surface area [cm^2]
    """
    R = R_phys
    n = len(R)
    dL = np.zeros(n)

    if n > 0:
        dL[0] = R[0] * theta[0]

    for i in range(1, n):
        dtheta = theta[i] - theta[i-1]
        dL[i] = np.sqrt(R[i]**2 + R[i-1]**2 - 2.0 * R[i] * R[i-1] * np.cos(dtheta))

    dA_perp = (R * np.sin(theta)) * (dL * sin_alpha) * 2.0 * np.pi

    return dL, dA_perp


# ============================================================
# Pre-shock perpendicular velocities
# ============================================================

def vnorm_forward(thr, rr, lam=0., Vstar=None):
    """
    Pre-forward shock perpendicular velocity.

    Parameters
    ----------
    thr : float or array
        Angle from the apex [rad].
    rr : float or array
        Normalized radial coordinate r/R0.
    lam : float
        Thermal pressure parameter
    Vstar : float
        Stellar velocity [cm/s].

    Returns
    -------
    v_perp : float or array
        Pre-shock velocity component perpendicular to the forward shock surface [cm/s].
    """
    s = np.sin(thr)
    c = np.cos(thr)
    Aomega, Az = AA(thr, rr, lam)
    return np.abs(Vstar * Aomega)


def vnorm_wind(thr, rr, lam=0., Vw=None):
    """
    Pre-forward shock perpendicular velocity.

    Parameters
    ----------
    thr : float or array
        Angle from the apex [rad].
    rr : float or array
        Normalized radial coordinate r/R0.
    lam : float
        Thermal pressure parameter
    Vw : float
        Stellar wind velocity [cm/s].

    Returns
    -------
    v_perp : float or array
        Pre-shock velocity component perpendicular to the reverse shock surface [cm/s].
    """
    s = np.sin(thr)
    c = np.cos(thr)
    Aomega, Az = AA(thr, rr, lam=lam)
    return np.abs(Vw * (-s * Az + c * Aomega))


def vtan(thr, rr, lam=0., shock='RS', Vw=None, Vstar=None):
    """
    Calculate the post-shock tangential velocity, assuming it is conserved
    across the shock (only the normal component is decelerated).

    Parameters
    ----------
    thr : float or array
        Angle from the apex [rad].
    rr : float or array
        Normalized radial coordinate r/R0.
    lam : float
        Lambda parameter
    shock : str
        'RS' for reverse shock or 'FS' for forward shock.
    Vw : float, optional
        Stellar wind velocity [cm/s]. Required if shock='RS'.
    Vstar : float, optional
        Stellar velocity [cm/s]. Required if shock='FS'.

    Returns
    -------
    v_tan : float or array
        Tangential velocity component [cm/s], clipped to a minimum of 1e-10 cm/s to avoid division by zero elsewhere.
    """
    s = np.sin(thr)
    c = np.cos(thr)
    Ao, Az = AA(thr, rr, lam)
    
    if shock == 'RS':
        if Vw is None:
            raise ValueError("Vw must be provided for reverse shock")
        V = Vw * (c*Az + s*Ao)
    elif shock == 'FS':
        if Vstar is None:
            raise ValueError("Vstar must be provided for forward shock")
        V = Vstar * np.maximum(-Az, 0.)
    
    return np.maximum(V, 1e-10)


# ============================================================
# Cooling function
# ============================================================

def lambda_T(T):
    """
    Cooling function from Myasnikov et al. (1998).
    
    Parameters:
    -----------
    T : float or array
        Temperature [K]
    
    Returns:
    --------
        lambda_T : [erg cm^3 s^-1]
    """
    T = np.asarray(T)
    result = np.zeros_like(T)

    mask_low = T < 1e4
    if np.any(mask_low):
        result[mask_low] = 7e-27 * 1e4
    
    mask1 = (T >= 1e4) & (T <= 1e5)
    if np.any(mask1):
        result[mask1] = 7e-27 * T[mask1]
    
    mask2 = (T > 1e5) & (T <= 4e7)
    if np.any(mask2):
        result[mask2] = 7e-19 * T[mask2]**(-0.6)
    
    mask_high = T > 4e7
    if np.any(mask_high):
        result[mask_high] = 3e-27 * np.sqrt(4e7)
        mask3 = T > 4e7
        if np.any(mask3):
            result[mask3] = 3e-27 * np.sqrt(T[mask3])
    
    invalid = ~(mask_low | mask1 | mask2 | mask_high)
    if np.any(invalid):
        result[invalid] = 1e-22
    
    return result


def cooling_time(n_post, T_post):
    """
    Post-shock cooling timescale [s].

    Parameters:
    -----------
    n_post : float or array
        Post shock numerical density
    T_post : float or array
        Post shock temperature

    Returns:
    --------
    t : float or array
        Post shock cooling time [s]
    """
    lambda_val = lambda_T(T_post)
    t = kB * T_post / (n_post * lambda_val)

    return t


# ============================================================
# Pre-shock conditions
# ============================================================

def pre_shock_ism(Vstar, n_ism, lam=0.):
    """
    ISM conditions ahead of the forward shock.

    Parameters
    ----------
    Vstar : float
        Stellar velocity [cm/s].
    n_ism : float
        ISM number density [cm^-3].
    lam : float
        alpha = lam / (1 + lam).

    Returns
    -------
    n_pre : float
        Pre-shock number density [cm^-3] (equal to n_ism).
    T_pre : float
        Pre-shock temperature [K].
    P_pre : float
        Pre-shock pressure thermal [erg/cm^3].
    cs : float
        Pre-shock sound speed [cm/s].
    """
    rho_pre = n_ism * mu * mp
    alpha = lam / (1 - lam) if lam < 1 else 0
    P_pre = alpha * rho_pre * Vstar**2
    cs = np.sqrt(gamma_ad * P_pre / rho_pre)
    T_pre = mp * cs**2 / (gamma_ad * kB)
    
    return n_ism, T_pre, P_pre, cs


def pre_shock_wind(Mdot, Vw, r_phys, wind_regime='hot', wind_T_fixed=None):
    """
    Wind conditions (reverse shock pre-shock).
    
    Parameters:
    -----------
    r_phys : float or array
        Physical radius [cm]
    wind_regime : str
        'cold', 'hot', or 'fixed'
    wind_T_fixed : float or None
        Fixed wind temperature [K] for 'fixed' regime
    
    Returns:
    --------
    n_pre : array
        Pre-shock numerical density [cm^-3]
    T_pre : array
        Pre-shock temperature [K]
    cs : array
        Speed of sound [cm/s]
    """
    rho_pre = Mdot / (4 * np.pi * r_phys**2 * Vw)
    n_pre = rho_pre / (mu_sh * mp)
        
    Vw_kms = Vw / 1e5
    if wind_regime == 'cold':
        T_pre = np.full_like(r_phys, 1e4, dtype=float)
    elif wind_regime == 'hot':
        T_pre = 1e5 * (Vw_kms / 2000.0)**2
        #T_pre = np.clip(T_pre, 3e4, 2e6)
    elif wind_regime == 'fixed':
        if wind_T_fixed is None:
            raise ValueError("wind_T_fixed must be provided for 'fixed' regime")
        T_pre = np.full_like(r_phys, wind_T_fixed, dtype=float)
    else:
        raise ValueError(f"Unknown regime: {wind_regime}")

    P_pre = n_pre*kB*T_pre
    
    cs = np.sqrt(gamma_ad * P_pre/rho_pre)
    
    return n_pre, T_pre, P_pre, cs


def vadv(thr, rr, R0_phys, v_t, v_perp, comp, t_cool):
    """
    Define a minimum advection velocity to avoid unphysically slow
    advection near the apex.

    The method finds the angle theta_L at which the surface distance
    traveled, s(theta_L), equals the cooling length,

        l_cool = (v_perp / comp) * t_cool

    and adopts the tangential velocity at that angle as the floor:

        v_adv_min = v_t(theta_L)

    The advection velocity is then

        v_adv = max(v_t, v_adv_min)

    Parameters
    ----------
    thr : array
        Angles from the apex [rad].
    rr : array
        Normalized radial coordinate r/R0.
    R0_phys : float
        Physical standoff radius [cm].
    v_t : array
        Tangential velocity [cm/s].
    v_perp : array
        Pre-shock perpendicular velocity [cm/s].
    comp : array
        Compression factor.
    t_cool : array
        Cooling timescale [s].

    Returns
    -------
    v_adv : array
        Advection velocity [cm/s], equal to max(v_t, v_adv_min).
    """

    R_phys = rr * R0_phys

    # Cooling length
    l_cool = (v_perp / comp) * t_cool

    # Length from the apex along the bow shock
    dL = np.zeros_like(thr)

    for i in range(1, len(thr)):
        dtheta = thr[i] - thr[i-1]

        dL[i] = np.sqrt( R_phys[i]**2 + R_phys[i-1]**2 - 2.0*R_phys[i]*R_phys[i-1]*np.cos(dtheta) )

    s = np.cumsum(dL)

    # We seek where f = 0
    f = s - l_cool

    idx = np.where(f >= 0)[0]

    if len(idx) == 0:
        v_min = v_t[-1]
    else:
        iL = idx[0]
        v_min = v_t[iL]

    return np.maximum(v_t, v_min)


# ============================================================
# Post-shock conditions
# ============================================================

def post_shock_conditions(thr, rr, shock, R0_phys, T_IL=1e4, **kwargs):
    """
    Calculate post-shock conditions for forward or reverse shock.
    
    When radiative: hot layer + cold recombination layer between hot layer and CD.
    When adiabatic: only hot layer (cold layer properties = hot layer properties, thickness=0)

    We employ Rankine-Hugoniot conditions if the shock is radiative
    And polytropic relation + specific enthalpy conservation if the shock is adiabatic
    
    Parameters:
    -----------
    thr : ndarray
        Angles from apex [rad]
    rr : ndarray
        Normalized radial coordinate r/R0
    shock : str
        'FS' for forward shock, 'RS' for reverse shock
    R0_phys : float
        Stagnation radius [cm]
    T_IL : float
        Recombination zone temperature [K]
    **kwargs : dict
        RS: Mdot, Vw, lam, wind_regime, wind_T_fixed
        FS: Vstar, n_ism, lam
    
    Returns:
    --------
    n_post : ndarray
        Post-shock numerical density [cm^-3] (hot layer)
    T_post : ndarray
        Post-shock temperature [K] (hot layer)
    n_rec : ndarray
        Recombination zone density [cm^-3] (cold layer, equals n_post if adiabatic)
    T_rec : ndarray
        Recombination zone temperature [K] (cold layer, equals T_post if adiabatic)
    P_post : ndarray
        Post-shock thermal pressure
    regime : ndarray of str
        'radiative' or 'adiabatic' for each theta
    H_hot : ndarray
        Hot layer thickness [cm] (post-shock layer)
    H_cold : ndarray
        Cold recombination layer thickness [cm] (0 for adiabatic)
    H_total : ndarray
        Total shocked layer thickness [cm] (H_hot + H_cold)
    t_cool : ndarray
        Post-shock thermal cooling timescale
    t_adv : ndarray
        Advection timescale
    """

    R_phys = rr * R0_phys
    n_points = len(thr)
    
    # Pre-shock conditions
    n_pre = np.zeros(n_points)
    T_pre = np.zeros(n_points)
    P_pre = np.zeros(n_points)
    cs_pre = np.zeros(n_points)
    cs_post = np.zeros(n_points)
    v_adv = np.zeros(n_points)

    rho_adi = np.zeros(n_points)
    
    if shock == 'RS':
        Mdot = kwargs.get('Mdot')
        Vw = kwargs.get('Vw')
        lam = kwargs.get('lam', 0.)
        wind_regime = kwargs.get('wind_regime', 'hot')
        wind_T_fixed = kwargs.get('wind_T_fixed', None)
        
        n_pre, T_pre, P_pre, cs_pre = pre_shock_wind(
            Mdot, Vw, R_phys, wind_regime, wind_T_fixed
        )
        v_perp = vnorm_wind(thr, rr, lam, Vw)
        v_pre = Vw
        mu_pre = mu_sh
        
    else:  # FS
        Vstar = kwargs.get('Vstar')
        n_ism = kwargs.get('n_ism')
        lam = kwargs.get('lam', 0.)
        
        n_pre[:] = n_ism
        _, T_pre[:], P_pre [:], cs_pre[:] = pre_shock_ism(Vstar, n_ism, lam)
        v_perp = vnorm_forward(thr, rr, lam, Vstar)
        v_pre = Vstar
        mu_pre = mu
    
    # Mach number
    M = v_perp / cs_pre

    # Compression factor (Rankine-Hugoniot)
    comp = (gamma_ad + 1.) * M**2 / ((gamma_ad - 1.) * M**2 + 2.)

    # Pre shock density
    rho_pre = n_pre * mu_pre * mp
    
    # Rankine-Hugoniot post-shock density (hot layer) used if radiative
    rho_RH = rho_pre * comp
    n_RH = rho_RH / (mu_sh * mp)
    cte_RH = (2. * gamma_ad * M**2 - (gamma_ad - 1.))
    P_RH = cte_RH / ( gamma_ad + 1. ) * P_pre

    T_ratio = ((gamma_ad - 1.) * M**2 + 2.) * cte_RH / ((gamma_ad + 1.)**2 * M**2)
    T_RH = T_pre * T_ratio

    # Adiabatic conditions
    P_adi = rho_pre * v_pre * v_perp
    rho_adi[0] = gamma_ad/(gamma_ad+1.) * 2. * P_adi[0] / v_pre**2.
    
    # Cooling and advection times
    t_cool = cooling_time(n_RH, T_RH)
    v_t = vtan(thr, rr, lam, shock, kwargs.get('Vw'), kwargs.get('Vstar'))
    v_adv = vadv(thr, rr, R0_phys, v_t, v_perp, comp, t_cool)
    t_adv = R_phys/v_adv
    
    # Radiative regime if cooling time < advection time
    is_radiative = t_cool < t_adv
    
    # Geometric factor for mass accumulation
    sin_alpha = v_perp / v_pre
    sin_alpha = np.clip(sin_alpha, 1e-10, 1.0)
    
    # Accumulated mass rate
    _, dA_perp = dL_dAperp(R_phys, thr, sin_alpha)
    dM = rho_pre * v_pre * dA_perp
    dot_M = np.cumsum(dM)
    
    # Initialize outputs
    n_post = np.zeros(n_points)
    T_post = np.zeros(n_points)
    n_rec = np.zeros(n_points)
    T_rec = np.zeros(n_points)
    P_post = np.zeros(n_points)
    cs_post = np.zeros(n_points)
    rho_post = np.zeros(n_points)
    H_hot = np.zeros(n_points)
    H_cold = np.zeros(n_points)
    H_total = np.zeros(n_points)
    regime = np.array(['adiabatic'] * n_points, dtype=object)

    if is_radiative[0]:
        regime[0] = 'radiative'
        rho_post[0] = rho_RH[0]
        P_post[0] = P_RH[0]
        T_post[0] = T_RH[0]
        n_post[0] = n_RH[0]
        n_rec[0] = n_RH[0] * (T_RH[0]/T_IL)
        T_rec[0] = T_IL
    else:
        rho_post[0] = rho_adi[0]
        P_post[0] = P_adi[0]
        T_post[0] = P_post[0]*mp*mu_sh/rho_post[0]/kB
        n_post[0] = rho_post[0]/(mu_sh*mp)
        n_rec[0] = n_post[0]
        T_rec[0] = T_post[0]

    cs_post[0] = np.sqrt(gamma_ad*P_post[0]/rho_post[0])

    supersonic = False
    v_perp_crit = None
    
    for i in range(1, n_points):
        
        if is_radiative[i]:
            regime[i] = 'radiative'

            rho_post[i] = rho_RH[i]
            P_post[i] = P_RH[i]
            cs_post[i] = np.sqrt(gamma_ad*P_post[i]/rho_post[i])

            if (v_adv[i] >= cs_post[i]):
                supersonic = True
                v_perp_crit = v_perp[i]

            n_post[i] = n_RH[i]
            T_post[i] = T_RH[i]
            
            # Cold layer (recombination zone) properties
            n_rec[i] = n_RH[i] * (T_RH[i] / T_IL)
            T_rec[i] = T_IL
            
            # Hot layer thickness: cooling layer (post-shock)
            H_hot[i] = (v_perp[i] / comp[i]) * t_cool[i]
            H_hot[i] = max(H_hot[i], 0.0)
            
            # Cold layer thickness from mass conservation
            rho_cold = n_rec[i] * mu_sh * mp
            denominator = 2.0 * np.pi * R_phys[i] * np.sin(thr[i]) * v_adv[i] * rho_cold
            
            if denominator > 0 and i > 0:
                H_cold[i] = dot_M[i] / denominator - H_hot[i] * (rho_RH[i] / rho_cold)
            H_cold[i] = max(H_cold[i], 0.0)
            
        else:
            # Adiabatic: only hot layer, cold layer = hot layer (no recombination)
            regime[i] = 'adiabatic'

            if not supersonic:

                P_post[i] = P_adi[i]
                rho_post[i] = rho_post[i-1] * (P_post[i]/P_post[i-1])**(1./gamma_ad)
                cs_post[i] = np.sqrt(gamma_ad*P_post[i]/rho_post[i])

                if v_adv[i] >= cs_post[i]:
                    supersonic = True
                    v_perp_crit = v_perp[i]

            else:
                P_post[i] = P_adi[i] * (v_perp[i]/v_perp_crit)
                rho_post[i] = rho_post[i-1] * (P_post[i]/P_post[i-1])**(1./gamma_ad)
                cs_post[i] = np.sqrt(gamma_ad*P_post[i]/rho_post[i])

            n_post[i] = rho_post[i]/(mu_sh*mp)
            T_post[i] = P_post[i] * mp * mu_sh / rho_post[i] / kB
            
            # Cold layer = hot layer
            n_rec[i] = n_post[i]
            T_rec[i] = T_post[i]
            
            denominator = 2.0 * np.pi * R_phys[i] * np.sin(thr[i]) * v_adv[i] * rho_post[i]
            
            H_hot[i] = dot_M[i] / denominator
            H_cold[i] = 0.0
    
    # Total thickness = hot layer + cold layer
    H_total = H_hot + H_cold

    return n_post, T_post, n_rec, T_rec, P_post, regime, H_hot, H_cold, H_total, t_cool, t_adv


# ============================================================
# Magnetic field
# ============================================================

def magnetic_field(U_B):
    '''
    U_B = B**2/(8*pi)

    Parameter:
    ----------
    U_B : float or array
        Magnetif field energy density

    Returns:
    --------
    B : float or array
        Magnetic field [G]
    B_avg : float or array
        Average magnetic field, assuming isotropization
    '''
    B = np.sqrt(8.*np.pi*U_B)
    B_avg = np.sqrt(2./3.) * B

    return B, B_avg