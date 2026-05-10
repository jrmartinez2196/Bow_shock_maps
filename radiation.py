# radiation.py
import numpy as np
from gaunt_factor import GauntFactor
from constants import h, kB, sr_per_arcsec2, Rayleigh

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

    return j_arcsec2/Rayleigh


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
    
    # O2+ fraction (4.57e-4 assuming solar abundances, Gnat & Sternberg 2007)
    no_ion2 = 4.57e-4 * n_arr
    
    n2 = no_ion2 * n_arr * q12 / A21
    
    # Emissivity
    j = (h * nu / (4 * np.pi)) * n2 * A21
    
    return j


def emissivity_freefree_sr(n, T, Z, nu, gaunt_lookup):
    """
    Free-free emissivity [erg s^-1 cm^-3 sr^-1]
    It uses a precomputed lookup table for the gaunt factors
    """
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    T_arr = np.clip(T_arr, 1e4, 1e8)
    
    # Fast gaunt from lookup table (interpolation)
    gaunt = gaunt_lookup(T_arr)
    
    # Emissivity calculation
    exp_factor = np.exp(-h * nu / (kB * T_arr))
    C_j = (6.8e-38 / (4 * np.pi)) * Z**2 * nu
    j = C_j * n_arr**2 * exp_factor * gaunt / np.sqrt(T_arr)
    j_mJy = j / (1e-26 * nu) / sr_per_arcsec2 # Radio plot mJy per arcsec^2
    
    return j, j_mJy

def precompute_gaunt_for_temperatures(nu_ff, Z):
    """
    Create a fast lookup function for gaunt factors at specific temperatures.
    """
    from gaunt_factor import gaunt_ff_calc
    
    T_values = np.logspace(4, 8, 500)  # 10^4 to 10^8 K, 500 points
    logT_values = np.log10(T_values)
    gaunt_values = gaunt_ff_calc(nu_ff, T_values, Z)
    
    # Create interpolation function
    def gaunt_lookup(T):
        """Fast interpolation from precomputed table"""
        T_clipped = np.clip(T, 1e4, 1e8)
        logT_clipped = np.log10(T_clipped)
        return np.interp(logT_clipped, logT_values, gaunt_values)
    
    return gaunt_lookup