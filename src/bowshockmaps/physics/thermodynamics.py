# thermodynamics.py
import numpy as np

from bowshockmaps.constants import gamma_ad, kB, mp, mu, mu_sh

# ============================================================
# Geometric function
# ============================================================


def AA(thr, rr, lam=0.0):
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
    fl = np.sqrt((thr - s * c) ** 2 * (1 - lam * rr * rr) ** 2 + (s**4 * (1 - rr * rr) ** 2))
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
        dtheta = theta[i] - theta[i - 1]
        dL[i] = np.sqrt(R[i] ** 2 + R[i - 1] ** 2 - 2.0 * R[i] * R[i - 1] * np.cos(dtheta))

    dA_perp = (R * np.sin(theta)) * (dL * sin_alpha) * 2.0 * np.pi

    return dL, dA_perp


# ============================================================
# Pre-shock perpendicular velocities
# ============================================================


def vnorm_forward(thr, rr, lam=0.0, Vstar=None):
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
    Aomega, Az = AA(thr, rr, lam)
    return np.abs(Vstar * Aomega)


def vnorm_wind(thr, rr, lam=0.0, Vw=None):
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


def vtan(thr, rr, lam=0.0, shock="RS", Vw=None, Vstar=None):
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

    if shock == "RS":
        if Vw is None:
            raise ValueError("Vw must be provided for reverse shock")
        V = Vw * (c * Az + s * Ao)
    elif shock == "FS":
        if Vstar is None:
            raise ValueError("Vstar must be provided for forward shock")
        V = Vstar * np.maximum(-Az, 0.0)

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
        result[mask_low] = 4e-29 * T[mask_low] ** (0.8)  # Muller 2018

    mask1 = (T >= 1e4) & (T <= 1e5)
    if np.any(mask1):
        result[mask1] = 7e-27 * T[mask1]

    mask2 = (T > 1e5) & (T <= 4e7)
    if np.any(mask2):
        result[mask2] = 7e-19 * T[mask2] ** (-0.6)

    mask_high = T > 4e7
    if np.any(mask_high):
        result[mask_high] = 3e-27 * np.sqrt(4e7)
        mask3 = T > 4e7
        if np.any(mask3):
            result[mask3] = 3e-27 * np.sqrt(T[mask3])

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


def pre_shock_ism(Vstar, n_ism, lam=0.0):
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


def pre_shock_wind(Mdot, Vw, r_phys, wind_regime="hot", wind_T_fixed=None):
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
    if wind_regime == "cold":
        T_pre = np.full_like(r_phys, 1e4, dtype=float)
    elif wind_regime == "hot":
        T_pre = 1e5 * (Vw_kms / 2000.0) ** 2
    elif wind_regime == "fixed":
        if wind_T_fixed is None:
            raise ValueError("wind_T_fixed must be provided for 'fixed' regime")
        T_pre = np.full_like(r_phys, wind_T_fixed, dtype=float)
    else:
        raise ValueError(f"Unknown regime: {wind_regime}")

    P_pre = n_pre * kB * T_pre

    cs = np.sqrt(gamma_ad * P_pre / rho_pre)

    return n_pre, T_pre, P_pre, cs


def vadv(thr, rr, R0_phys, v_perp, comp, t_cool, v_pre, P_adi, rho_adi0):
    """
    Advection velocity along the bow shock for an adiabatic shock.
    We calculate the advection velocity from energy conservation using Bernoulli's equation
    and neglecting the pre-shock thermal pressure

    0.5*v_pre**2 = 0.5*v_adv**2 + (gamma_ad/(gamma_ad-1)) * P_adi/rho_adi

    We also set a minimum v_adv for the region near the apex,
    where the cooling length is shorter than
    the length traveled by the fluid

    Parameters
    ----------
    thr, rr, R0_phys, v_perp, comp, t_cool : as before
    v_pre : float
        Pre-shock velocity [cm/s].
    P_adi : array
        Adiabatic post-shock pressure, rho_pre*v_pre*v_perp.
    rho_adi0 : float
        Adiabatic post-shock density at the apex.

    Returns
    -------
    v_adv : array
        Advection velocity [cm/s].
    """
    R_phys = rr * R0_phys

    # Cooling length
    l_cool = (v_perp / comp) * t_cool

    # Distance traveled along the bow shock from the apex
    dL = np.zeros_like(thr)
    for i in range(1, len(thr)):
        dtheta = thr[i] - thr[i - 1]
        dL[i] = np.sqrt(
            R_phys[i] ** 2 + R_phys[i - 1] ** 2 - 2.0 * R_phys[i] * R_phys[i - 1] * np.cos(dtheta)
        )
    s = np.cumsum(dL)

    # Density from the polytropic relation
    rho_adi = rho_adi0 * (P_adi / P_adi[0]) ** (1.0 / gamma_ad)

    cte_ad = gamma_ad / (gamma_ad - 1.0)
    v_adv = np.sqrt(v_pre**2 - 2.0 * cte_ad * P_adi / rho_adi)

    idx_cool = np.where(s >= l_cool)[0]

    if idx_cool.size > 0:
        i_cool = idx_cool[0]
        # We set a minimum value v_adv(min) = v_adv(s = l_cool) near the apex
        v_adv[:i_cool] = v_adv[i_cool]

    return v_adv


# ============================================================
# Post-shock conditions
# ============================================================


def blend_adiabatic_thickness(H_ad, W, l_cool):
    """
    Adiabatic layer thickness, made continuous with the radiative one at the
    transition between the two regimes.

    The adiabatic thickness H_ad is set by Bernoulli's flow speed v_adv and the
    adiabatic density; the radiative one by the kinematic speed v_tan and the
    strong-shock density, and equals W where the cooling length l_cool equals W
    (the cold layer is then zero and the hot layer fills the region). The two
    mass accountings differ by a factor of order 1-2, so switching regime at
    l_cool = W made the thickness jump (x0.49 for the forward shock of RXJ0528+2838
    at V_wind = 500 km/s).

    Far from the transition (l_cool >> W) the gas is adiabatic and Bernoulli
    holds, so H_ad is kept. Approaching it, the thickness is interpolated
    geometrically towards W:

        H = H_ad**(1 - w) * W**w,    w = min(1, W / l_cool)**2

    w is 1 at the transition (continuity) and falls to 0 quickly away from it.
    The form of the weight is a modeling choice, not derived: any w with w(1) = 1
    that decays gives continuity; the square keeps the correction local (the
    first power also changed angles that are only marginally adiabatic by
    factors of up to 2).

    Parameters
    ----------
    H_ad : float or array
        Adiabatic thickness, from mass conservation with v_adv.
    W : float or array
        Width of the shocked region from mass conservation with v_tan and the
        strong-shock density (same units as H_ad).
    l_cool : float or array
        Cooling length, (v_perp / compression) * t_cool (same units).

    Returns
    -------
    float or array
        The blended thickness. H_ad where W, l_cool or H_ad are not positive and
        finite (there is nothing to blend with).
    """
    H_ad = np.asarray(H_ad, dtype=float)
    W = np.asarray(W, dtype=float)
    l_cool = np.asarray(l_cool, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        w = np.clip(W / l_cool, 0.0, 1.0) ** 2
        blended = H_ad ** (1.0 - w) * W**w
    valid = np.isfinite(w) & np.isfinite(blended) & (W > 0) & (H_ad > 0) & (l_cool > 0)
    return np.where(valid, blended, H_ad)


def post_shock_conditions(thr, rr, shock, R0_phys, T_IL=8e3, **kwargs):
    """
    Calculate post-shock conditions for forward or reverse shock.

    When radiative: hot layer + cold recombination layer between hot layer and CD.
    When adiabatic: only hot layer

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
        Hot layer thickness [cm] (post-shock layer). Radiative: the cooling
        length. Adiabatic: from mass conservation with v_adv, blended towards the
        radiative width near the transition (blend_adiabatic_thickness).
    H_cold : ndarray
        Cold recombination layer thickness [cm] (0 for adiabatic)
    H_total : ndarray
        Total shocked layer thickness [cm] (H_hot + H_cold)
    t_cool : ndarray
        Post-shock thermal cooling timescale
    t_adv : ndarray
        Advection time: the time the tangential flow (speed v_tan) takes to
        remove the mass the shock has accumulated, dot_M / (2 pi R sin(theta)
        v_tan rho_pre v_perp). An angle is radiative if t_cool < t_adv, i.e.
        if the cooling length is shorter than the width of the shocked
        region.
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

    if shock == "RS":
        Mdot = kwargs.get("Mdot")
        Vw = kwargs.get("Vw")
        lam = kwargs.get("lam", 0.0)
        wind_regime = kwargs.get("wind_regime", "hot")
        wind_T_fixed = kwargs.get("wind_T_fixed", None)

        n_pre, T_pre, P_pre, cs_pre = pre_shock_wind(Mdot, Vw, R_phys, wind_regime, wind_T_fixed)
        v_perp = vnorm_wind(thr, rr, lam, Vw)
        v_pre = Vw
        mu_pre = mu_sh

    else:  # FS
        Vstar = kwargs.get("Vstar")
        n_ism = kwargs.get("n_ism")
        lam = kwargs.get("lam", 0.0)

        n_pre[:] = n_ism
        _, T_pre[:], P_pre[:], cs_pre[:] = pre_shock_ism(Vstar, n_ism, lam)
        v_perp = vnorm_forward(thr, rr, lam, Vstar)
        v_pre = Vstar
        mu_pre = mu

    # Mach number
    M = v_perp / cs_pre

    # Where the normal Mach number is <= 1 there is no shock at all: the
    # flow across the surface is subsonic. The Rankine-Hugoniot relations
    # below are only valid for M > 1 and give nonsense under it (the
    # compression tends to 0 instead of 1, and pressure/temperature go
    # negative for M < ~0.45 -- which is what raised "invalid value
    # encountered in power" in lambda_T). Evaluate the jump conditions at
    # M = 1 there (no compression, no heating: just the ambient gas) and
    # give those angles zero layer thickness at the end of the function.
    no_shock = ~(M > 1.0)  # also catches NaN
    M = np.where(no_shock, 1.0, M)

    # Compression factor (Rankine-Hugoniot)
    comp = (gamma_ad + 1.0) * M**2 / ((gamma_ad - 1.0) * M**2 + 2.0)

    # Pre shock density
    rho_pre = n_pre * mu_pre * mp

    # Rankine-Hugoniot post-shock density (hot layer) used if radiative
    rho_RH = rho_pre * comp
    n_RH = rho_RH / (mu_sh * mp)
    cte_RH = 2.0 * gamma_ad * M**2 - (gamma_ad - 1.0)
    P_RH = cte_RH / (gamma_ad + 1.0) * P_pre

    T_ratio = ((gamma_ad - 1.0) * M**2 + 2.0) * cte_RH / ((gamma_ad + 1.0) ** 2 * M**2)
    T_RH = T_pre * T_ratio

    # Adiabatic conditions
    P_adi = rho_pre * v_pre * v_perp
    rho_adi[0] = gamma_ad / (gamma_ad - 1.0) * 2.0 * P_adi[0] / v_pre**2.0

    # Cooling time and flow speeds.
    #
    # v_t (tangential velocity, purely kinematic from the pre-shock flow
    # geometry) is regime-independent: it only assumes the tangential
    # velocity component is continuous across the shock, which holds
    # whether the shocked gas cools or not. v_adv (Bernoulli-derived)
    # explicitly assumes adiabatic (energy-conserving) flow, so it is
    # only physically valid *after* we already know the shock is
    # adiabatic. Using v_adv to decide the regime would be circular, so
    # the regime is decided with v_t, and v_adv is reserved for the
    # quantities computed within the adiabatic branch below.
    t_cool = cooling_time(n_RH, T_RH)
    v_t = vtan(thr, rr, lam, shock, kwargs.get("Vw"), kwargs.get("Vstar"))
    v_adv = vadv(thr, rr, R0_phys, v_perp, comp, t_cool, v_pre, P_adi, rho_adi[0])

    # Geometric factor for mass accumulation
    sin_alpha = v_perp / v_pre
    sin_alpha = np.clip(sin_alpha, 1e-10, 1.0)

    # Accumulated mass rate
    _, dA_perp = dL_dAperp(R_phys, thr, sin_alpha)
    dM = rho_pre * v_pre * dA_perp
    dot_M = np.cumsum(dM)

    # ------------------------------------------------------------------
    # Regime: radiative if the gas cools within the shocked region, i.e. if the
    # cooling length l_cool = (v_perp/compression) * t_cool is shorter than the
    # width W of that region.
    #
    # W is the width mass conservation gives the region if the gas stays at the
    # strong-shock density rho_RH, flowing along the surface at v_t:
    #
    #     dot_M = 2 pi R sin(theta) * v_t * rho_RH * W
    #
    # and uses no energy conservation, so it does not presuppose the regime.
    # Equivalently t_cool < t_adv, with
    #
    #     t_adv = W * compression / v_perp = dot_M / (2 pi R sin(theta) v_t rho_pre v_perp)
    #
    # the time it takes the tangential flow to remove the mass accumulated by
    # the shock (the mean residence time of the gas in the region).
    #
    # Unlike R/v_tan (an earlier choice of t_adv), W is finite on the symmetry
    # axis: dot_M ~ theta^2 and sin(theta) * v_t ~ theta^2 there. R/v_tan
    # diverges, so every angle near the axis was "radiative" even when the
    # cooling length exceeded the radius (H_hot/R ~ 1 and a cold layer of
    # negative thickness for V_wind = 100 km/s, RXJ0528+2838 parameters).
    with np.errstate(divide="ignore", invalid="ignore"):
        W = dot_M / (2.0 * np.pi * R_phys * np.sin(thr) * v_t * rho_RH)
    t_adv = W * comp / v_perp
    is_radiative = t_cool < t_adv  # NaN compares False: adiabatic

    # Layer thicknesses if the shock is radiative.
    #
    # Hot layer: the cooling length. Cold layer: from mass conservation, what
    # the accumulated mass leaves once the hot layer has taken its share, with
    # v_t as above:
    #
    #     H_cold = dot_M / (2 pi R sin(theta) v_t rho_cold) - H_hot * rho_RH / rho_cold
    #            = (rho_RH / rho_cold) * (W - l_cool)
    #
    # so H_cold >= 0 exactly where the regime is radiative (l_cool < W): the cold
    # layer starts at zero thickness at the transition and grows from there.
    rho_cold_rad = n_RH * (T_RH / T_IL) * mu_sh * mp
    H_hot_rad = (v_perp / comp) * t_cool
    denominator_rad = 2.0 * np.pi * R_phys * np.sin(thr) * v_t * rho_cold_rad
    with np.errstate(divide="ignore", invalid="ignore"):
        H_cold_rad = dot_M / denominator_rad - H_hot_rad * (rho_RH / rho_cold_rad)

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
    regime = np.array(["adiabatic"] * n_points, dtype=object)

    if is_radiative[0]:
        regime[0] = "radiative"
        rho_post[0] = rho_RH[0]
        P_post[0] = P_RH[0]
        T_post[0] = T_RH[0]
        n_post[0] = n_RH[0]
        n_rec[0] = n_RH[0] * (T_RH[0] / T_IL)
        T_rec[0] = T_IL
    else:
        rho_post[0] = rho_adi[0]
        P_post[0] = P_adi[0]
        T_post[0] = P_post[0] * mp * mu_sh / rho_post[0] / kB
        n_post[0] = rho_post[0] / (mu_sh * mp)
        n_rec[0] = n_post[0]
        T_rec[0] = T_post[0]

    cs_post[0] = np.sqrt(gamma_ad * P_post[0] / rho_post[0])

    supersonic = False
    v_perp_crit = None

    for i in range(1, n_points):

        if is_radiative[i]:
            regime[i] = "radiative"

            rho_post[i] = rho_RH[i]
            P_post[i] = P_RH[i]
            cs_post[i] = np.sqrt(gamma_ad * P_post[i] / rho_post[i])

            if not supersonic:
                if v_adv[i] >= cs_post[i]:
                    supersonic = True
                    v_perp_crit = v_perp[i]

            n_post[i] = n_RH[i]
            T_post[i] = T_RH[i]

            # Cold layer (recombination zone) properties
            n_rec[i] = n_RH[i] * (T_RH[i] / T_IL)
            T_rec[i] = T_IL

            # Hot layer: cooling layer (post-shock); cold layer: from mass
            # conservation (see above; non-negative by construction here).
            H_hot[i] = H_hot_rad[i]
            H_cold[i] = H_cold_rad[i]

        else:
            # Adiabatic: only hot layer, cold layer = hot layer (no recombination)
            regime[i] = "adiabatic"

            if not supersonic:

                P_post[i] = P_adi[i]
                rho_post[i] = rho_post[i - 1] * (P_post[i] / P_post[i - 1]) ** (1.0 / gamma_ad)
                cs_post[i] = np.sqrt(gamma_ad * P_post[i] / rho_post[i])

                if v_adv[i] >= cs_post[i]:
                    supersonic = True
                    v_perp_crit = v_perp[i]

            else:
                P_post[i] = P_adi[i] * (v_perp[i] / v_perp_crit)
                rho_post[i] = rho_post[i - 1] * (P_post[i] / P_post[i - 1]) ** (1.0 / gamma_ad)
                cs_post[i] = np.sqrt(gamma_ad * P_post[i] / rho_post[i])

            n_post[i] = rho_post[i] / (mu_sh * mp)
            T_post[i] = P_post[i] * mp * mu_sh / rho_post[i] / kB

            # Cold layer = hot layer
            n_rec[i] = n_post[i]
            T_rec[i] = T_post[i]

            denominator = 2.0 * np.pi * R_phys[i] * np.sin(thr[i]) * v_adv[i] * rho_post[i]

            # Bernoulli-based thickness, made continuous with the radiative one
            # at the transition (see blend_adiabatic_thickness).
            H_hot[i] = blend_adiabatic_thickness(dot_M[i] / denominator, W[i], H_hot_rad[i])
            H_cold[i] = 0.0

    # Total thickness = hot layer + cold layer
    H_total = H_hot + H_cold

    # Near the symmetry axis (theta -> 0), the mass-flux argument behind
    # the dot_M-based layer-thickness formula above (whichever of
    # H_hot/H_cold uses it, depending on regime) has a genuine geometric
    # singularity: its denominator carries a sin(theta) factor from the
    # "ring" mass flux picture, which vanishes at theta=0. Rather than
    # trust that formula arbitrarily close to the axis, freeze the
    # layer thickness to its value at a small cutoff theta_min for
    # everything below it -- same approach used in the reference
    # Fortran implementation this model is based on.
    theta_min = 0.1  # rad
    below_cutoff = thr < theta_min
    if np.any(below_cutoff) and np.any(~below_cutoff):
        i_min = np.argmax(~below_cutoff)  # first index with thr >= theta_min
        H_hot[below_cutoff] = H_hot[i_min]
        H_cold[below_cutoff] = H_cold[i_min]
        H_total[below_cutoff] = H_total[i_min]

    # No shock (M <= 1) -> no shocked layer.
    H_hot[no_shock] = 0.0
    H_cold[no_shock] = 0.0
    H_total[no_shock] = 0.0

    return (
        n_post,
        T_post,
        n_rec,
        T_rec,
        P_post,
        regime,
        H_hot,
        H_cold,
        H_total,
        t_cool,
        t_adv,
    )


# ============================================================
# Magnetic field
# ============================================================


def magnetic_field(U_B):
    """
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
    """
    B = np.sqrt(8.0 * np.pi * U_B)
    B_avg = np.sqrt(2.0 / 3.0) * B

    return B, B_avg
