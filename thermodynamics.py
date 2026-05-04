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
    At the apex (θ=0), v_perp = Vstar (maximum).
    """
    s = np.sin(thr)
    c = np.cos(thr)
    Aomega, Az = AA(thr, rr, lam)
    return Vstar * Aomega


def vnorm_wind(thr, rr, lam=0., Vw=None):
    """
    Pre-reverse shock perpendicular velocity.
    At the apex (θ=0), v_perp = Vw (maximum, wind impacts head-on).
    """
    s = np.sin(thr)
    c = np.cos(thr)
    Aomega, Az = AA(thr, rr, lam=lam)
    return np.abs(Vw * (-s * Az + c * Aomega))


def vtan(thr, rr, lam=0., shock='RS', Vw=None, Vstar=None):
    """
    Calculates the post-shock tangential velocity, assuming it is conserved
    through the shock.
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
    else:
        raise ValueError(f"Unknown shock: {shock}")
    
    return np.maximum(V, 1e-10)

def cs(v_perp, comp):
    """
    Calculates the post-shock sound speed
    P_down = 0.75*rho_up*v_perp**2
    rho_down = comp*rho_up
    cs = sqrt(gamma_ad*P_down/rho_dow) = sqrt(gamma_ad*0.75*v_perp**2/comp)

    Parameters:
    -----------
    v_perp : float
        upstream velocity perpendicular to the shock front
    comp : float
        compression factor


    Return:
    -------
    cs : float
    """

    return np.sqrt(gamma_ad*0.75*v_perp**2/comp)


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
    """
    lambda_val = lambda_T(T_post)
    return kB * T_post / (n_post * lambda_val)


# ============================================================
# Advection time
# ============================================================

def advection_time(thr, rr, R0_phys, comp, lam=0., shock='RS', Vw=None, Vstar=None):
    """
    Aproximate the advection velocity as the maximum btw v_tan and the speed of sound
    to avoid numerical errors near the apex
    """
    v_t_val = vtan(thr, rr, lam, shock, Vw, Vstar)

    if shock == 'RS':
        v_perp = vnorm_wind(thr, rr, lam, Vw)
    elif shock == 'FS':
        v_perp = vnorm_forward(thr, rr, lam, Vstar)

    cs_post = cs(v_perp, comp)

    R_phys = rr * R0_phys

    v_adv = np.maximum(v_t_val, cs_post)
    #v_adv = v_t_val

    return R_phys / np.maximum(v_adv, 1e-10)


# ============================================================
# Pre-shock conditions
# ============================================================

def pre_shock_ism(Vstar, n_ism, lam=0.):
    """
    ISM conditions (forward shock pre-shock).
    
    Returns:
    --------
    n_pre : float
        Pre-shock numerical density [cm^-3]
    T_pre : float
        Pre-shock temperature [K]
    cs : float
        Speed of sound [cm/s]
    """
    rho_ISM = n_ism * mu * mp
    alpha = lam / (1 - lam) if lam < 1 else 0
    P_ISM = alpha * rho_ISM * Vstar**2
    cs = np.sqrt(gamma_ad * P_ISM / rho_ISM)
    T_pre = mp * cs**2 / (gamma_ad * kB)
    
    return n_ism, T_pre, cs


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
        T_pre = np.clip(T_pre, 3e4, 2e6)
    elif wind_regime == 'fixed':
        if wind_T_fixed is None:
            raise ValueError("wind_T_fixed must be provided for 'fixed' regime")
        T_pre = np.full_like(r_phys, wind_T_fixed, dtype=float)
    else:
        raise ValueError(f"Unknown regime: {wind_regime}")
    
    cs = np.sqrt(gamma_ad * kB * T_pre / (mu * mp))
    
    return n_pre, T_pre, cs


# ============================================================
# Post-shock conditions
# ============================================================

def post_shock_conditions(thr, rr, shock, R0_phys, T_IL=1e4, **kwargs):
    """
    Calculate post-shock conditions for forward or reverse shock.
    
    When radiative: hot layer + cold recombination layer between hot layer and CD.
    When adiabatic: only hot layer (cold layer properties = hot layer properties, thickness=0)
    
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
    regime : ndarray of str
        'radiative' or 'adiabatic' for each theta
    H_hot : ndarray
        Hot layer thickness [cm] (post-shock layer)
    H_cold : ndarray
        Cold recombination layer thickness [cm] (0 for adiabatic)
    H_total : ndarray
        Total shocked layer thickness [cm] (H_hot + H_cold)
    """
    # Input validation
    if len(thr) != len(rr):
        raise ValueError(f"thr and rr must have same length: {len(thr)} vs {len(rr)}")
    if R0_phys <= 0:
        raise ValueError(f"R0_phys must be positive: {R0_phys}")
    
    R_phys = rr * R0_phys
    n_points = len(thr)
    
    # Pre-shock conditions
    n_pre = np.zeros(n_points)
    T_pre = np.zeros(n_points)
    cs_pre = np.zeros(n_points)
    cs_post = np.zeros(n_points)
    v_adv = np.zeros(n_points)
    
    if shock == 'RS':
        Mdot = kwargs.get('Mdot')
        Vw = kwargs.get('Vw')
        lam = kwargs.get('lam', 0.)
        wind_regime = kwargs.get('wind_regime', 'hot')
        wind_T_fixed = kwargs.get('wind_T_fixed', None)
        
        n_pre, T_pre, cs_pre = pre_shock_wind(
            Mdot, Vw, R_phys, wind_regime, wind_T_fixed
        )
        v_perp = vnorm_wind(thr, rr, lam, Vw)
        v_parallel = Vw
        mu_pre = mu_sh
        
    else:  # FS
        Vstar = kwargs.get('Vstar')
        n_ism = kwargs.get('n_ism')
        lam = kwargs.get('lam', 0.)
        
        n_pre[:] = n_ism
        _, T_pre[:], cs_pre[:] = pre_shock_ism(Vstar, n_ism, lam)
        v_perp = vnorm_forward(thr, rr, lam, Vstar)
        v_parallel = Vstar
        mu_pre = mu
    
    # Mach number
    M = v_perp / cs_pre
    M = np.maximum(M, 1.0)
    
    # Compression factor (Rankine-Hugoniot)
    comp = (gamma_ad + 1) * M**2 / ((gamma_ad - 1) * M**2 + 2)
    
    # Rankine-Hugoniot post-shock density (hot layer)
    rho_pre = n_pre * mu_pre * mp
    rho_RH = rho_pre * comp
    n_RH = rho_RH / (mu_sh * mp)
    cs_post = cs(v_perp, comp)
    
    # Rankine-Hugoniot post-shock temperature (hot layer)
    T_ratio = ((gamma_ad - 1) * M**2 + 2) * (2 * gamma_ad * M**2 - (gamma_ad - 1)) / ((gamma_ad + 1)**2 * M**2)
    T_RH = T_pre * T_ratio
    T_RH = np.maximum(T_RH, 1e4)
    
    # Cooling and advection times
    t_cool = cooling_time(n_RH, T_RH)
    v_t = vtan(thr, rr, lam, shock, kwargs.get('Vw'), kwargs.get('Vstar'))
    #t_adv = R_phys / v_t
    #t_adv = np.maximum(t_adv, 1e-10)
    v_adv = np.maximum(cs_post, v_t)
    #v_adv = v_t
    t_adv = R_phys/v_adv
    #t_adv = np.maximum(R_phys/v_adv, 1e-10)
    
    # Radiative regime if cooling time < advection time
    is_radiative = t_cool < t_adv
    
    # Geometric factor for mass accumulation
    sin_alpha = v_perp / v_parallel
    sin_alpha = np.clip(sin_alpha, 1e-10, 1.0)
    
    # Accumulated mass rate
    _, dA_perp = dL_dAperp(R_phys, thr, sin_alpha)
    dM = rho_pre * v_parallel * dA_perp
    dot_M = np.cumsum(dM)
    
    # Initialize outputs
    n_post = np.zeros(n_points)
    T_post = np.zeros(n_points)
    n_rec = np.zeros(n_points)
    T_rec = np.zeros(n_points)
    H_hot = np.zeros(n_points)
    H_cold = np.zeros(n_points)
    H_total = np.zeros(n_points)
    regime = np.array(['adiabatic'] * n_points, dtype=object)
    
    for i in range(n_points):
        n_post[i] = n_RH[i]
        T_post[i] = T_RH[i]
        
        if is_radiative[i]:
            regime[i] = 'radiative'
            
            # Cold layer (recombination zone) properties
            n_rec[i] = n_RH[i] * (T_RH[i] / T_IL)
            T_rec[i] = T_IL
            
            # Hot layer thickness: cooling layer (post-shock)
            H_hot[i] = (v_perp[i] / comp[i]) * t_cool[i]
            H_hot[i] = max(H_hot[i], 0.0)
            
            # Cold layer thickness from mass conservation
            rho_cold = n_rec[i] * mu_sh * mp
            denominator = 2.0 * np.pi * R_phys[i] * np.sin(thr[i]) * v_t[i] * rho_cold
            
            if denominator > 0 and i > 0:
                H_cold[i] = dot_M[i] / denominator - H_hot[i] * (rho_RH[i] / rho_cold)
            H_cold[i] = max(H_cold[i], 0.0)
            
        else:
            # Adiabatic: only hot layer, cold layer = hot layer (no recombination)
            regime[i] = 'adiabatic'
            
            # Cold layer has same properties as hot layer (no NaN!)
            n_rec[i] = n_RH[i]
            T_rec[i] = T_RH[i]
            
            rho_post = n_RH[i] * mu_sh * mp
            denominator = 2.0 * np.pi * R_phys[i] * np.sin(thr[i]) * v_t[i] * rho_post
            
            if denominator > 0 and i > 0:
                H_hot[i] = dot_M[i] / denominator
            H_hot[i] = max(H_hot[i], 0.0)
            H_cold[i] = 0.0
    
    # Total thickness = hot layer + cold layer
    H_total = H_hot + H_cold
    
    return n_post, T_post, n_rec, T_rec, regime, H_hot, H_cold, H_total


# ============================================================
# Layer thickness wrapper
# ============================================================

def compute_layer_thickness(thr, rr, R0_phys, shock, T_IL=1e4, theta_min=0.1, **kwargs):
    """
    Computes shocked layer width with per-theta regime determination.
    
    Returns:
    --------
    H_total : ndarray
        Total shocked layer width [cm]
    H_hot : ndarray
        Hot layer width [cm]
    H_cold : ndarray
        Cold recombination width [cm] (0 for adiabatic)
    regime : ndarray
        'radiative' or 'adiabatic' for each theta
    dot_M : None
        Placeholder for compatibility
    """
    _, _, _, _, regime, H_hot, H_cold, H_total = post_shock_conditions(
        thr, rr, shock, R0_phys, T_IL, **kwargs
    )
    
    # Extrapolate for small theta
    mask_small = thr < theta_min
    if np.any(mask_small):
        idx_min = np.searchsorted(thr, theta_min)
        if idx_min < len(H_total):
            H_total[mask_small] = H_total[idx_min]
            H_hot[mask_small] = H_hot[idx_min]
            H_cold[mask_small] = H_cold[idx_min]
    
    return H_total, H_hot, H_cold, regime, None
