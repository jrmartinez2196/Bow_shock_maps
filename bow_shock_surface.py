# bow_shock_surface.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from constants import AU, Msun, year, mu, mp

def standoff_distance(Mdot, Vw, Vstar, n_ism):
    rho_ism = n_ism * mu * mp
    R0 = np.sqrt(Mdot* Vw / (4 * np.pi * rho_ism * Vstar**2))
    return R0

def _rhs_christie(theta, r, lam, tiny=1e-14):
    s = np.sin(theta)
    c = np.cos(theta)
    num = r * s * ( theta * (1.0 - lam * r**2) + r**2 * (lam - 1.0) * s * c )
    denom = (1.0 - lam * r**2) * (theta - s * c) * c - s**3 * (1.0 - r**2)
    if np.abs(denom) < tiny:
        denom = np.sign(denom) * tiny if denom != 0 else tiny
    return num / denom

def integrate_r_theta_christie(lam, R0, theta_max=np.pi-1e-5, n_theta=800,
                               eps_start=1e-6, atol=1e-9, rtol=1e-9):
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

def make_surface_from_r_theta(r_vals, theta_vals, R0, lam, n_phi=200):
    phi = np.linspace(0, 2*np.pi, n_phi)
    TH, PH = np.meshgrid(theta_vals, phi)
    interp = interp1d(theta_vals, r_vals, bounds_error=False, fill_value=np.nan)
    r_grid = interp(TH)
    alph = lam/(1-lam)
    R_grid = r_grid * R0 * np.sqrt(1/(1+alph))
    X = (R_grid * np.sin(TH) * np.cos(PH)) / AU
    Y = (R_grid * np.sin(TH) * np.sin(PH)) / AU
    Z = (R_grid * np.cos(TH)) / AU
    return X, Y, Z

def bow_shock_surface_christie(R0, lam=0.0, theta_max=np.pi-1e-5, n_theta=800, n_phi=200,
                               eps_start=1e-6):
    theta_vals, r_vals = integrate_r_theta_christie(lam, R0,
                                                    theta_max=theta_max,
                                                    n_theta=n_theta,
                                                    eps_start=eps_start)
    X, Y, Z = make_surface_from_r_theta(r_vals, theta_vals, R0, lam, n_phi=n_phi)
    return X, Y, Z