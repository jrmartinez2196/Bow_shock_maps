# bow_shock_surface.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from constants import AU, Msun, year, mu, mp

def standoff_distance(Mdot, Vw, Vstar, n_ism):
    '''
    Calculates the distance to the termination shock at te apex (Wilkin 1997)
    Ignores ISM thermal pressure

    Parameters: (all in cgs units)
    -----------
    Mdot : float
        Wind mass-loss rate
    Vw : float
        Stellar wind velocity
    Vstar : float
        Stellar velocity
    n_ism : float
        ISM numerical density

    Returns:
    --------
    R0 : float
        Termination shock distance at the apex
    '''
    rho_ism = n_ism * mu * mp
    R0 = np.sqrt(Mdot* Vw / (4 * np.pi * rho_ism * Vstar**2))
    return R0

def _rhs_christie(theta, r, lam, tiny=1e-14):
    '''
    Right hand side of Eq. 11 (EDO) from Christie+ 2016

    Parameters:
    -----------
    theta : float
        Angle measured from the apex to the bow shock tail
    r : float
        Normalized distance to the bow shock
    lam : float
        lam = alpha/(1+alpha) (Eq. 8), alpha being the thermal pressure fraction

    Returns:
    --------
    num/denom : float
        r' = dr/dtheta
    '''
    s = np.sin(theta)
    c = np.cos(theta)
    num = r * s * ( theta * (1.0 - lam * r**2) + r**2 * (lam - 1.0) * s * c )
    denom = (1.0 - lam * r**2) * (theta - s * c) * c - s**3 * (1.0 - r**2)
    if np.abs(denom) < tiny:
        denom = np.sign(denom) * tiny if denom != 0 else tiny
    return num / denom

def integrate_r_theta_christie(lam, R0, theta_max=np.pi-1e-5, n_theta=800,
                               eps_start=1e-6, atol=1e-9, rtol=1e-9):
    """
    Integrate r(theta) for Eq. 11 from theta=eps_start..theta_max.
    Parameters:
    -----------
    lam : float
        lambda parameter
    R0  : float
        standoff distance
    theta_max : float
        Maximum integration angle
    n_theta : integer
        theta resolution
    eps_start : float
        minimum angle; avoids numerical errors at the apex
    atol : float
        Absolute tolerance for the solver
    rtol : float
        Relative tolerance
    
    Returns:
    --------
    theta_vals : 1D array
        Array with the angle values
    r_vals : 1D array
        Array with the r(theta) = R(theta)/R0 dimensionless values
    """
    theta_eval = np.linspace(eps_start, theta_max, n_theta)
    r0 = 1.0
    def rhs(t, y):
        return _rhs_christie(t, y[0], lam)
    sol = solve_ivp(rhs, (eps_start, theta_max), [r0],
                    t_eval=theta_eval, atol=atol, rtol=rtol, 
                    max_step=(theta_max-eps_start)/100.0)
    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)
    theta_vals = sol.t
    r_vals = sol.y[0]
    r_vals = np.where(np.isfinite(r_vals) & (r_vals > 0), r_vals, np.nan)
    return theta_vals, r_vals