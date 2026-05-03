# gaunt_factor.py - Module for Gaunt factor calculation
import numpy as np
from pathlib import Path
from constants import Ry, kB, h

class GauntFactor:
    """
    Calculates the thermally averaged Gaunt factor using van Hoof (2014) tables
    
    The Gaunt factor is used in free-free (bremsstrahlung) opacity calculations.
    The table provides <gff(u,gamma2)> with:
        -8 < log10(u) < 8, -4 < log10(gamma2) < 4, step = 0.2 dex
    where u = h*nu/(k*T) and gamma2 = Z^2 * Ry / (k*T)
    """
    
    def __init__(self, data_file='gauntff.dat'):
        """
        Initialize the Gaunt factor table
        
        Parameters:
        -----------
        data_file : str
            Path to the gauntff.dat file
        """
        # Table parameters
        self.nu_mu = 146    # number of grid points in log10(u) direction
        self.nu_mga = 81    # number of grid points in log10(gamma2) direction
        self.log_gamma2_start = -6.0
        self.log_u_start = -16.0
        self.step = 0.2
        
        # Load table
        self.gff = self._load_table(data_file)
        
        # Create log10(gamma2) and log10(u) grids
        self.log_gamma2_grid = self.log_gamma2_start + np.arange(self.nu_mga) * self.step
        self.log_u_grid = self.log_u_start + np.arange(self.nu_mu) * self.step
    
    def _load_table(self, data_file):
        """
        Load the Gaunt factor table from file
        
        The file format: first 16 lines are headers, then 146 rows × 81 columns
        
        Returns:
        --------
        gff : ndarray
            Gaunt factor matrix (dimensions: nu_mu x nu_mga)
        """
        # Find the file
        file_path = Path(data_file)
        if not file_path.exists():
            # Try looking in the same directory as this module
            file_path = Path(__file__).parent / data_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Gaunt factor data file not found: {data_file}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines (first 16 lines contain comments and metadata)
        data_lines = lines[16:]
        
        # Read data
        gff = np.zeros((self.nu_mu, self.nu_mga))
        for i in range(self.nu_mu):
            # Each line has self.nu_mga values
            values = data_lines[i].strip().split()
            if len(values) >= self.nu_mga:
                gff[i, :] = [float(v) for v in values[:self.nu_mga]]
            else:
                raise ValueError(f"Line {i+17} does not have enough values")
        
        return gff
    
    def _interpol2(self, dx, dy, fx1y1, fx2y1, fx1y2):
        """
        Bilinear interpolation
        
        Parameters:
        -----------
        dx, dy : float
            Distances from (x1,y1) in units of step
        fx1y1, fx2y1, fx1y2 : float
            Values at (x1,y1), (x2,y1), (x1,y2)
        
        Returns:
        --------
        finterpol : float
            Interpolated value
        """
        fx = (fx2y1 - fx1y1) / 0.2  # derivative in x
        fy = (fx1y2 - fx1y1) / 0.2  # derivative in y
        return fx1y1 + fx * dx + fy * dy
    
    def _extrapol2(self, dx, dy, fx1y1, fx2y1, fx1y2):
        """
        Bilinear extrapolation (used when j_mu >= nu_mga)
        
        Parameters:
        -----------
        dx, dy : float
            Distances from (x1,y1) in units of step
        fx1y1, fx2y1, fx1y2 : float
            Values at (x1,y1), (x2,y1), (x1,y2)
        
        Returns:
        --------
        fextrapol : float
            Extrapolated value
        """
        fx = (fx2y1 - fx1y1) / 0.2
        fy = (fx1y2 - fx1y1) / 0.2
        return fx1y2 + fx * dx + fy * dy
    
    def __call__(self, nu_ff, T_l, Zq_l):
        """
        Calculate the thermally averaged Gaunt factor
        
        Parameters:
        -----------
        nu_ff : float or ndarray
            Frequency [Hz]
        T_l : float
            Temperature [K]
        Zq_l : float
            Effective ion charge
        
        Returns:
        --------
        gaunt_ff : float or ndarray
            Thermally averaged Gaunt factor
        """
        return self.gaunt_ff_calc(nu_ff, T_l, Zq_l)
    
    def gaunt_ff_calc(self, nu_ff, T_l, Zq_l):
        """
        Calculate the thermally averaged Gaunt factor
        
        Parameters:
        -----------
        nu_ff : float or ndarray
            Frequency [Hz]
        T_l : float
            Temperature [K]
        Zq_l : float
            Effective ion charge
        
        Returns:
        --------
        gaunt_ff : float or ndarray
            Thermally averaged Gaunt factor
        """
        # Convert to array if scalar
        scalar_input = np.isscalar(nu_ff)
        nu_ff = np.atleast_1d(nu_ff)
        T_l = np.atleast_1d(T_l)
        
        gaunt_ff = np.zeros_like(nu_ff, dtype=float)
        
        for i in range(len(nu_ff)):
            # Check for valid temperature
            if not np.isfinite(T_l[i]) or T_l[i] <= 0:
                gaunt_ff[i] = 1.0  # Default Gaunt factor for invalid temperature
                continue
            
            # Calculate gamma2 = Z^2 * Ry / (k * T)
            gamma2 = Zq_l**2 * Ry / (kB * T_l[i])
            log_gamma2 = np.log10(gamma2)
            
            # Check for valid log_gamma2
            if not np.isfinite(log_gamma2):
                gaunt_ff[i] = 1.0
                continue
            
            # Index in the gamma2 table
            j_g = int((log_gamma2 + 6.0) / 0.2)
            # Ensure j_g is within valid range
            j_g = max(0, min(j_g, self.nu_mga - 2))
            
            dg = ((log_gamma2 + 6.0) / 0.2 - j_g) * 0.2
            
            # Calculate mu = h * nu / (k * T)
            mu = h * nu_ff[i] / (kB * T_l[i])
            log_mu = np.log10(mu)
            
            # Check for valid log_mu
            if not np.isfinite(log_mu):
                gaunt_ff[i] = 1.0
                continue
            
            # Index in the mu table
            j_mu = int((log_mu + 16.0) / 0.2)
            
            # Check boundaries
            if j_mu < 0:
                j_mu = 0
                dmu = 0.0
            elif j_mu >= self.nu_mu - 1:
                j_mu = self.nu_mu - 2
                dmu = ((log_mu + 16.0) / 0.2 - j_mu) * 0.2
            else:
                dmu = ((log_mu + 16.0) / 0.2 - j_mu) * 0.2
            
            # Ensure j_g is within bounds
            j_g_use = min(j_g, self.nu_mga - 2)
            
            if j_mu < self.nu_mu - 1:
                gaunt_ff[i] = self._interpol2(
                    dg, dmu,
                    self.gff[j_mu, j_g_use],
                    self.gff[j_mu, j_g_use + 1],
                    self.gff[j_mu + 1, j_g_use]
                )
            else:
                gaunt_ff[i] = self._extrapol2(
                    dg, dmu,
                    self.gff[self.nu_mu - 2, j_g_use],
                    self.gff[self.nu_mu - 2, j_g_use + 1],
                    self.gff[self.nu_mu - 1, j_g_use]
                )
        
        if scalar_input:
            return gaunt_ff[0]
            
        return gaunt_ff


# Convenience function (similar to Fortran interface)
def gaunt_ff_calc(nu_ff, T_l, Zq_l, data_file='gauntff.dat'):
    """
    Calculate the thermally averaged Gaunt factor
    
    This is a convenience function that creates a GauntFactor instance
    and calls its method.
    
    Parameters:
    -----------
    nu_ff : float or ndarray
        Frequency [Hz]
    T_l : float
        Temperature [K]
    Zq_l : float
        Effective ion charge
    data_file : str
        Path to the gauntff.dat file
    
    Returns:
    --------
    gaunt_ff : float or ndarray
        Thermally averaged Gaunt factor
    """
    gaunt = GauntFactor(data_file)
    return gaunt.gaunt_ff_calc(nu_ff, T_l, Zq_l)
