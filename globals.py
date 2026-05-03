# globals.py
import numpy as num

# ============================================================
# Thermal cooling (Myasnikov et al. 1998)
# ============================================================

def lambda_T(T):
    """
    Cooling function Myasnikov et al. (1998)
    
    Parameter:
    -----------
    T : float or array
        Temperature [K]
    
    Returns:
    --------
    lambda_T : [erg cm^3 s^-1]
    """
    T = np.asarray(T)
    result = np.zeros_like(T)
    
    mask1 = (T >= 1e4) & (T <= 1e5)
    result[mask1] = 7e-27 * T[mask1]
    
    mask2 = (T > 1e5) & (T <= 4e7)
    result[mask2] = 7e-19 * T[mask2]**(-0.6)
    
    mask3 = T > 4e7
    result[mask3] = 3e-27 * np.sqrt(T[mask3])
    
    if np.any(~(mask1 | mask2 | mask3)):
        print("WARNING: Temperature out of range in lambda_T")
        result[~(mask1 | mask2 | mask3)] = 1e-22
    
    return result


def cooling_time(T, n):
    """
    Cooling timescale [s].

    Parameters:
    -----------
    T : float or ndarray
        Post shock temperature [K]
    n : float or ndarray
    	Numerical post shock density [cm^-3]

    Returns:
    --------
    t_cool
    """
    lambda_T_val = lambda_T(T)
    t_cool = kB * T / (n * lambda_T_val)
    return t_cool