# radiation.py
import numpy as np
from gaunt_factor import GauntFactor
from constants import h, kB

def emissivity_Halpha_sr(n, T):
    """
    H_alpha [erg s^-1 cm^-3 sr^-1]
    Mackey+ 2013
    """
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    
    # Ensure positive values
    n_arr = np.maximum(n_arr, 0.0)
    T_arr = np.maximum(T_arr, 1e4)
    
    j_arcsec2 = 2.85e-33 * T_arr**-0.9 * n_arr**2
    sr_per_arcsec2 = (206265.)**2
    return j_arcsec2 * sr_per_arcsec2


def emissivity_OIII_sr(n, T):
    """
    [O III] λ5007 [erg s^-1 cm^-3 sr^-1]
    Osterbrock & Ferland 2006
    """
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    
    # Ensure positive values
    n_arr = np.maximum(n_arr, 0.0)
    T_arr = np.maximum(T_arr, 1e4)
    
    nu = 5.997e14
    A21 = 0.02     # s^-1
    gamma_12 = 2.29
    g_12 = 9
    E12_k = 32900  # K
    
    # Collision rate
    q12 = 8.629e-6 * gamma_12 * np.exp(-E12_k / T_arr) / (np.sqrt(T_arr) * g_12)
    
    # O2+ fraction
    no_ion2 = 1e-4 * n_arr
    
    n2 = no_ion2 * n_arr * q12 / A21
    
    # Emissivity
    j = (h * nu / (4 * np.pi)) * n2 * A21
    
    return j


def emissivity_freefree_sr(n, T, Z, nu=2e6*1e9):
    """
    Free-free emissivity [erg s^-1 cm^-3 sr^-1]
    """
    from gaunt_factor import gaunt_ff_calc
    
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    T_arr = np.clip(T_arr, 1e4, 1e8)
    
    gaunt = gaunt_ff_calc(nu, T_arr, Z)
    
    # Emissivity calculation
    exp_factor = np.exp(-h * nu / (kB * T_arr))
    j = (6.8e-38 / (4 * np.pi)) * Z**2 * n_arr**2 * nu * exp_factor * gaunt / np.sqrt(T_arr)
    
    return j