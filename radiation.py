# radiation.py
import numpy as np
from gaunt_factor import GauntFactor
from constants import h, kB, sr_per_arcsec2, Rayleigh, qe, me, mec2, c
from scipy.special import gamma

def emissivity_Halpha(n, T, ion_H=1.):
    """
    H_alpha [R cm^-1]
    Mackey+ 2013
    """
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    
    # Ensure positive values
    n_arr = np.maximum(n_arr, 0.0)
    T_arr = np.maximum(T_arr, 1e4)
    
    j_arcsec2 = 2.85e-33 * T_arr**-0.9 * (ion_H*n_arr)**2   # erg/s/cm3/arcsec^2

    return j_arcsec2/Rayleigh


def emissivity_OIII(n, T, ion_H = 1., ion_O=1.):
    """
    [O III] λ5007 [erg s^-1 cm^-3 arcsec^-2]
    Osterbrock & Ferland 2006
    Valid for densities < 10^4 cm^-3
    """
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    
    # Ensure positive values
    n_arr = np.maximum(n_arr, 0.0)
    T_arr = np.maximum(T_arr, 1e4)
    
    nu = 5.997e14
    A21 = 0.02     # s^-1
    A_5007 = 0.0209    # s^-1, 1D2 -> 3P2
    A_4959 = 0.0068    # s^-1, 1D2 -> 3P1
    A_tot  = A_5007 + A_4959

    gamma_12 = 2.29
    g_12 = 9
    E12_k = 28737. # K - Meyer 2016
    
    # Collision rate
    q12 = 8.629e-6 * gamma_12 * np.exp(-E12_k / T_arr) / (np.sqrt(T_arr) * g_12)
    
    # O2+ fraction (4.57e-4 assuming solar abundances, Gnat & Sternberg 2007)
    n_e = ion_H * n_arr
    n_OIII = 4.57e-4 * n_arr * ion_O
    
    n2 = n_OIII * n_e * q12 / A_tot
    
    # Emissivity
    j = (h * nu / (4 * np.pi)) * n2 * A_5007 / sr_per_arcsec2
    #j = 3.23e-21 * np.exp(-28737/T) * n_e**2./(4.*np.pi*np.sqrt(T)) / sr_per_arcsec2
    
    return j


def nu_emissivity_freefree(n, T, ion_H, Z_q, nu, gaunt_lookup):
    """
    Free-free emissivity multiplied by frequency (nu*j_nu) [erg s^-1 cm^-3 arcsec^-2]
    It uses a precomputed lookup table for the gaunt factors
    """
    n_arr = np.asarray(n, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    T_arr = np.clip(T_arr, 1e4, 1e8)
    
    # Fast gaunt from lookup table (interpolation)
    gaunt = gaunt_lookup(T_arr)
    
    # Emissivity calculation
    exp_factor = np.exp(-h * nu / (kB * T_arr))
    C_j = (6.8e-38 / (4 * np.pi)) * Z_q**2
    j = C_j * (n_arr*ion_H)**2 * exp_factor * gaunt / np.sqrt(T_arr) / sr_per_arcsec2 # erg/s/cm^3/arcsec^2/Hz
    j_mJy = j / (1e-26) # Radio mJy/cm/arcsec^2
    
    return nu*j, j_mJy

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

def nu_emissivity_sync(k0, B, p_inj, nu):
    """
    Synchrotron emissivity multiplied by frequency (nu*j_nu) [erg s^-1 cm^-3 arcsec^-2]

    Parameters:
    -----------
    k0 : float or array
        Electron energy distribution (n [1/erg/cm^3]) normalization
    B : float or array
        Post shock magnetic field
    p_inj : float
        Spectral index of the electron power law distribution
        Prescription only valid for p_inj > 2
        We assume p_inj = p, as advection dominates.
    nu : float
        Frequency

    Returns:
    --------
    nu*j_syn_arcsec : array
        [erg s^-1 cm^-3 arcsec^-2]
    j_syn_mJy
        [mJy/cm/arcsec^2]
    """

    eps = h*nu

    a_syn = cte_syn(p_inj)
    cte_s = qe**3. / (h*mec2) * (3.*h*qe/(4.*np.pi*me**3.*c**5.))**((p_inj-1.)/2.)

    j_syn = a_syn * cte_s * k0 * B**((p_inj+1.)/2.) * eps**(-(p_inj-1.)/2.) * eps/nu   # erg/s/cm^3/sr/Hz

    j_syn_arcsec = j_syn / sr_per_arcsec2 # erg/s/cm^3/arcsec^2/Hz

    j_syn_mJy = j_syn_arcsec / (1e-26)      # mJy/cm/arcsec^2

    return nu*j_syn_arcsec, j_syn_mJy

def cte_syn(p):
    '''
    Calculates a constant of the synchrotron emissivity as a function of electron spectral index

    Parameter:
    ----------
    p : float
        Distribution spectral index
    '''

    a_num =  2.**((p-1.)/2.) * np.sqrt(3) * gamma((3.*p-1.)/12.) * gamma((3.*p+19.)/12.) * gamma((p+5.)/4.)
    a_den = 8. * np.sqrt(np.pi) * (p+1.) * gamma((p+7.)/4.)

    return a_num/a_den