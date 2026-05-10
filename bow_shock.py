#!/usr/bin/env python3
"""
Bow Shock Interactive Visualizer
run as: python3 bow_shock.py [source_name] [params_dir]
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from pathlib import Path

from constants import AU, pc, Msun_yr, mu, mp, mu_sh, h, kB
from params_loader import get_source_params, validate_params
from bow_shock_surface import (
    standoff_distance,
    integrate_r_theta_christie,
)
from thermodynamics import (
    post_shock_conditions,
    vnorm_forward,
    vnorm_wind,
    vtan,
    cooling_time,
    advection_time
)
from utils import (
    make_projection_maps_fast,
    radial_profile,
    arcsecond
)

def sanitize_log_data(data):
    """Ensure data is valid for LogNorm"""
    data = np.asarray(data)
    data = np.where(np.isfinite(data), data, 0.0)
    return np.clip(data, 1e-300, None)

def get_norm(data):
    data = data[np.isfinite(data) & (data > 0)]
    
    if len(data) == 0:
        return LogNorm(vmin=1e-30, vmax=1e-20)
    
    vmax = np.max(data)
    
    vmin = vmax / 1000.
    
    return LogNorm(vmin=vmin, vmax=1.1*vmax)


def clip_data(data):
    """Clip extreme bright pixels to improve contrast"""
    data = np.asarray(data)
    pmax = np.percentile(data[np.isfinite(data)], 99.5)
    return np.clip(data, None, pmax)


class BowShock:
    def __init__(self, source_name='RXJ0528+2838', params_dir=None):
        """
        Initialize bow shock model with parameters from file.
        
        Parameters:
        -----------
        source_name : str
            Name of the source (e.g., 'RXJ0528+2838')
        params_dir : str or Path
            Directory containing parameter files (converted to Path internally)
        """
        # Convert to Path if string provided
        if params_dir is None:
            params_dir = Path.cwd()
        else:
            params_dir = Path(params_dir)
        
        print(f"Loading parameters from: {params_dir}")
        # Load parameters from file
        self.params = get_source_params(source_name, params_dir)
        validate_params(self.params)
        
        # Extract parameters with conversions to cgs
        # Mdot: from Msun/yr to g/s
        self.Mdot_msun = self.params['Mdot']
        self.Mdot = self.params['Mdot'] * Msun_yr
        
        # Velocities: from km/s to cm/s
        self.Vw = self.params['Vw'] * 1e5
        self.Vstar = self.params['Vstar'] * 1e5
        
        # Density: already in cm^-3
        self.n_ism = self.params['n_ism']
        
        # Distance: in pc (no conversion needed for storage, but used in arcsec calc)
        self.distance = self.params.get('dist')
        
        # Inclination: degrees
        self.inclination = self.params['inclination']

        # Projected angle (PA = 0° -> towards north, measured from north to east in degrees)
        self.PA = self.params['PA']
        
        # Initialize T_ism (will be controlled by slider)
        self.T_ism = self.params.get('T_ism', 8.e3)
        
        # Calculate lam from T_ism (physical relation)
        self.update_lam_from_T_ism()
        
        # Recombination zone temperature
        self.T_IL = self.params.get('T_IL', 1e4)
        
        # Wind regime
        self.wind_regime = self.params.get('wind_regime', 'hot')
        self.wind_T_fixed = self.params.get('wind_T_fixed', None)
        
        # Visualization parameters
        self.zmax = self.params.get('zmax', 5e15)
        self.nz = self.params.get('nz', 150)
        self.nx = self.params.get('nx', 300)
        self.ny = self.params.get('ny', 300)
        self.xlim_factor = self.params.get('xlim_factor', 10.0)
        self.ylim_factor = self.params.get('ylim_factor', 10.0)
        
        # Frequency for free-free emission [Hz]
        self.continuum_frequencies = {
            'radio': 3e9,           # 3 GHz - radio
            'IR': 3e12,             # 3 THz - MIR
            'optical_R': 4.3e14,    # 700 nm - OP - Red
            'optical_V': 5.5e14,    # 545 nm - OP - Green
            'optical_B': 6.9e14,    # 435 nm - OP - Blue
            'FUV': 2e15,            # 150 nm - FUV
            'EUV': 3e16,            # 10 nm - Extreme UV
            'Xray_soft': 3e17,      # 1 keV - Soft X-rays
            'Xray_hard': 3e18       # 10 keV - Hard X-rays
        }

        band_name = self.params.get('continuum_band', 'FUV')
        self.band_name = band_name
        self.nu_ff = self.continuum_frequencies.get(band_name, self.continuum_frequencies['FUV'])
        
        # Store references to plot objects
        self.fig1 = None
        self.fig2 = None
        self.fig3 = None
        self.fig_sliders = None
        self.sliders = {}
        self.lines = {}
        self.images = {}
        self.colorbars = {}
        self.profiles = {}
        self.fig3_ax = None
        self.rgb_image = None
        self.fig3_colorbars = {}
        
        # Pre-compute theta grid
        print("Computing theta grid...")
        self.theta_grid = np.linspace(0.01, 2.*np.pi/3., 500)
        
        # Load initial R_RS function
        print("Loading R_RS function...")
        self.update_R_RS_func()
        
        # Calculate R0 for verification
        R0_test = self.update_R0()
        print(f"R0 = {R0_test/AU:.1f} AU")
        #print(f"Projected stagnation point distance = {arcsecond(R0_test*np.cos(self.inclination*np.pi/180.), self.distance)} ''")
        
        print(f"Loaded parameters for {source_name}")
        print(f"  Mdot = {self.Mdot_msun:.2e} Msun/yr = {self.Mdot:.2e} g/s")
        print(f"  Vw = {self.Vw/1e5:.1f} km/s")
        print(f"  Vstar = {self.Vstar/1e5:.1f} km/s")
        print(f"  n_ism = {self.n_ism:.2f} cm^-3")
        print(f"  T_ism = {self.T_ism:.2f} K")
        print(f"  inclination = {self.inclination:.1f} deg")

    def get_continuum_bands(self):
        """Returns the available frequencies to compute free-free emission"""
        return list(self.continuum_frequencies.keys())

    def set_continuum_band(self, band_name):
        """Modifies the frequency"""
        if band_name in self.continuum_frequencies:
            self.band_name = band_name
            self.nu_ff = self.continuum_frequencies[band_name]
            print(f"Continuum band changed to {band_name} ({self.nu_ff:.2e} Hz)")
            if hasattr(self, 'fig2') and self.fig2 is not None:
                self.update_figure2()
        else:
            available = ', '.join(self.get_continuum_bands())
            raise ValueError(f"Unknown band: {band_name}. Available: {available}")
    
    def update_lam_from_T_ism(self):
        """
        Calculate lam from T_ism:
        
        alpha = P_th / P_kin
        P_th = n_ism * kB * T_ism
        P_kin = rho_ism * Vstar**2 = (n_ism * mu * mp) * Vstar**2
        
        -> alpha = (kB * T_ism) / (mu * mp * Vstar**2)
        
        -> lambda = alpha / (1 + alpha)
        """
        # Calculate the kinetic and thermal ISM pressure
        rho_ism = self.n_ism * mu * mp
        P_therm = self.n_ism * kB * self.T_ism
        P_kin = rho_ism * self.Vstar**2
        
        # Alpha = P_therm / P_kin
        self.alpha = P_therm / P_kin
        
        # Lambda = alpha / (1 + alpha)
        self.lam = self.alpha / (1.0 + self.alpha)
        self.lmb = np.log10(self.lam)
    
    def update_R0(self):
        """Calculate standoff radius [cm]"""
        return standoff_distance(self.Mdot, self.Vw, self.Vstar, self.n_ism)
    
    def update_R_RS_func(self):
        """
        Updates the R_RS function using the Christie ODE integration.
    
        Produces the bow shock surface
        """
        theta_vals, r_vals = integrate_r_theta_christie(
            self.lam, 
            R0=1.0, 
            theta_max=np.pi - 1e-5,
            n_theta=800,
            eps_start=1e-6
        )
        
        theta_vals = np.insert(theta_vals, 0, 0.0)
        r_vals = np.insert(r_vals, 0, 1.0)
        
        self.R_RS_func = interp1d(
            theta_vals, 
            r_vals, 
            kind='linear',
            bounds_error=False, 
            fill_value=(r_vals[0], r_vals[-1])
        )
    
    def get_params_from_sliders(self):
        """
        Get current parameter values from sliders and convert to CGS
        """
        self.Mdot_msun = 10**self.sliders['Mdot_log'].val
        self.Mdot = self.Mdot_msun * Msun_yr
        self.Vw = self.sliders['Vw'].val * 1e5      # km/s -> cm/s
        self.Vstar = self.sliders['Vstar'].val * 1e5 # km/s -> cm/s
        self.n_ism = self.sliders['n_ism'].val
        self.T_ism = self.sliders['T_ism'].val
        self.inclination = self.sliders['inclination'].val
        
        self.update_lam_from_T_ism()
        
        # Update R_RS_func when lam changes
        self.update_R_RS_func()
    
    def compute_thermo(self):
        """
        Computes the hydro and thermodynamic variables as a function of theta
        Computes the radiative or adiabatic nature of each shock at each position
        
        The calculation accounts for:
        - Physical standoff distance R0 corrected for thermal pressure (alpha parameter)
        - Wind density profile n_wind along the shock surface
        - Post-shock conditions using the post_shock_conditions function
        - Radiative vs. adiabatic regimes (cold vs. hot components)
        - Normalized layer thicknesses H/R for hot and cold phases
        - Perpendicular and tangential velocity components
        - Cooling-to-advection time ratios to determine shock regime
        
        Returns:
        --------
        dict: Dictionary containing all computed quantities with keys
        (_RS-> reverse shock; _FS -> Forward shock)
            'theta' : Polar angle array [rad]
            'n_hot', 'n_cold' : densities from the post shock and recombination layers [cm^-3]
            'T_hot', 'T_cold' : RS temperatures [K]
            'regime' : Shock regime ('radiative'/'adiabatic')
            'H_hot', 'H_cold', 'H_RS_total' : normalized thicknesses
            'v_perp'' : Perpendicular velocities [cm/s]
            'v_tan' : Tangential velocities [cm/s]
            'ratio' : Cooling-to-advection time ratios
        """
        R0_phys = self.update_R0()
        alpha = self.alpha
        lam = self.lam
        R0_corrected = R0_phys * np.sqrt(1/(1+alpha))

        print(f"Projected stagnation point distance = {arcsecond(R0_corrected*np.cos(self.inclination*np.pi/180.), self.distance):.1f} ''")
        
        thr, rr_norm = integrate_r_theta_christie(lam=lam, R0=1.0, theta_max=np.pi)
        
        r_interp = interp1d(thr, rr_norm, bounds_error=False, fill_value=1.0)
        rr = r_interp(self.theta_grid)
        rr = np.where(np.isfinite(rr), rr, 1.0)

        n_wind = self.Mdot/(4.*np.pi*(rr*R0_phys)**2*self.Vw)/mu_sh/mp
        
        n_RS, T_RS, n_rec_RS, T_rec_RS, regime_RS, H_hot_RS, H_cold_RS, H_total_RS = post_shock_conditions(
            self.theta_grid, rr, 'RS', R0_corrected,
            T_IL=self.T_IL,
            Mdot=self.Mdot, Vw=self.Vw, lam=lam,
            wind_regime=self.wind_regime, wind_T_fixed=self.wind_T_fixed
        )
        
        n_FS, T_FS, n_rec_FS, T_rec_FS, regime_FS, H_hot_FS, H_cold_FS, H_total_FS = post_shock_conditions(
            self.theta_grid, rr, 'FS', R0_corrected,
            T_IL=self.T_IL,
            Vstar=self.Vstar, n_ism=self.n_ism, lam=lam
        )
        
        n_RS_cold = np.where(regime_RS == 'radiative', n_rec_RS, np.nan)
        T_RS_cold = np.where(regime_RS == 'radiative', T_rec_RS, np.nan)
        H_RS_cold_norm = np.where(regime_RS == 'radiative', H_cold_RS / (rr * R0_corrected), np.nan)
        H_RS_hot_norm = H_hot_RS / (rr * R0_corrected)
        
        n_FS_cold = np.where(regime_FS == 'radiative', n_rec_FS, np.nan)
        T_FS_cold = np.where(regime_FS == 'radiative', T_rec_FS, np.nan)
        H_FS_cold_norm = np.where(regime_FS == 'radiative', H_cold_FS / (rr * R0_corrected), np.nan)
        H_FS_hot_norm = H_hot_FS / (rr * R0_corrected)
        
        H_R_norm_RS = H_total_RS / (rr * R0_corrected)
        H_R_norm_FS = H_total_FS / (rr * R0_corrected)
        
        v_perp_RS = vnorm_wind(self.theta_grid, rr, lam, self.Vw)
        v_perp_FS = vnorm_forward(self.theta_grid, rr, lam, self.Vstar)
        v_tan_RS = vtan(self.theta_grid, rr, lam, shock='RS', Vw=self.Vw)
        v_tan_FS = vtan(self.theta_grid, rr, lam, shock='FS', Vstar=self.Vstar)
        
        t_cool_RS = cooling_time(n_RS, T_RS)
        t_adv_RS = advection_time(self.theta_grid, rr, R0_corrected, (n_RS/n_wind), lam, shock='RS', Vw=self.Vw)
        ratio_RS = t_cool_RS / np.maximum(t_adv_RS, 1e-10)
        
        t_cool_FS = cooling_time(n_FS, T_FS)
        t_adv_FS = advection_time(self.theta_grid, rr, R0_corrected, (n_FS*mu_sh)/(self.n_ism*mu), lam, shock='FS', Vstar=self.Vstar)
        ratio_FS = t_cool_FS / np.maximum(t_adv_FS, 1e-10)
        
        return {
            'theta': self.theta_grid,
            'n_RS_hot': n_RS, 'n_RS_cold': n_RS_cold,
            'n_FS_hot': n_FS, 'n_FS_cold': n_FS_cold,
            'T_RS_hot': T_RS, 'T_RS_cold': T_RS_cold,
            'T_FS_hot': T_FS, 'T_FS_cold': T_FS_cold,
            'regime_RS': regime_RS, 'regime_FS': regime_FS,
            'H_RS_hot': H_RS_hot_norm, 'H_RS_cold': H_RS_cold_norm, 'H_RS_total': H_R_norm_RS,
            'H_FS_hot': H_FS_hot_norm, 'H_FS_cold': H_FS_cold_norm, 'H_FS_total': H_R_norm_FS,
            'v_perp_RS': v_perp_RS, 'v_perp_FS': v_perp_FS,
            'v_tan_RS': v_tan_RS, 'v_tan_FS': v_tan_FS,
            'ratio_RS': ratio_RS, 'ratio_FS': ratio_FS,
        }
    
    def compute_maps(self):
        """Compute emission maps using vectorized code."""
        R0_phys = self.update_R0()
        alpha = self.alpha
        R0_corrected = R0_phys * np.sqrt(1/(1+alpha))
        
        xlim = 5.*self.xlim_factor * R0_corrected
        ylim = 5.*self.ylim_factor * R0_corrected
        inclination_rad = np.deg2rad(90 - self.inclination)
        zmax = max(self.zmax, 5 * R0_corrected)
        
        x_vals_arcsec, y_vals_arcsec, result = make_projection_maps_fast(
            xlim, ylim, self.nx, self.ny, self.R_RS_func,
            inclination=inclination_rad, PA = self.PA,
            zmax=zmax, nz=self.nz,
            lmb=self.lmb, R0_phys=R0_corrected,
            T_IL=self.T_IL,
            Vstar=self.Vstar, n_ism=self.n_ism,
            Mdot=self.Mdot, Vw=self.Vw,
            wind_regime=self.wind_regime, wind_T_fixed=self.wind_T_fixed,
            nu_ff=self.nu_ff,
            distance=self.distance
        )
        
        return {
            'x': x_vals_arcsec,
            'y': y_vals_arcsec,
            'I_Halpha': result['I_Halpha'],
            'I_OIII': result['I_OIII'],
            'I_ff_total': result['I_ff_total'],
            'I_ff_mJy' : result['I_ff_mJy']
        }
    
    def create_figure1(self):
        self.fig1 = plt.figure(figsize=(15, 12))
        self.fig1.suptitle('Bow Shock Profiles', fontsize=16)
        
        ax1 = self.fig1.add_subplot(3, 2, 1)
        ax1.set_xlabel(r'$\theta$ [rad]')
        ax1.set_ylabel(r'Density [cm$^{-3}$]')
        ax1.set_yscale('log')
        ax1.set_title('Post-shock Density')
        ax1.grid(True, alpha=0.3)
        
        ax2 = self.fig1.add_subplot(3, 2, 2)
        ax2.set_xlabel(r'$\theta$ [rad]')
        ax2.set_ylabel(r'Temperature [K]')
        ax2.set_yscale('log')
        ax2.set_title('Post-shock Temperature')
        ax2.grid(True, alpha=0.3)
        
        ax3 = self.fig1.add_subplot(3, 2, 3)
        ax3.set_xlabel(r'$\theta$ [rad]')
        ax3.set_ylabel(r'H/R')
        ax3.set_yscale('linear')
        ax3.set_ylim(1e-1,1.)
        ax3.set_title('Normalized Layer Thickness')
        ax3.grid(True, alpha=0.3)
        
        ax4 = self.fig1.add_subplot(3, 2, 4)
        ax4.set_xlabel(r'$\theta$ [rad]')
        ax4.set_ylabel(r'Velocity [km/s]')
        ax4.set_title('Velocities')
        ax4.grid(True, alpha=0.3)
        
        ax5 = self.fig1.add_subplot(3, 1, 3)
        ax5.set_xlabel(r'$\theta$ [rad]')
        ax5.set_ylabel(r'$t_{cool} / t_{adv}$')
        ax5.set_yscale('log')
        ax5.set_title('Cooling to Advection Time Ratio')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        self.fig1_axes = [ax1, ax2, ax3, ax4, ax5]
        self.lines['fig1'] = {}
        
        self.lines['fig1']['n_RS_hot'], = ax1.plot([], [], 'b-', linewidth=2, label='RS hot')
        self.lines['fig1']['n_RS_cold'], = ax1.plot([], [], 'b--', linewidth=2, label='RS cold')
        self.lines['fig1']['n_FS_hot'], = ax1.plot([], [], 'r-', linewidth=2, label='FS hot')
        self.lines['fig1']['n_FS_cold'], = ax1.plot([], [], 'r--', linewidth=2, label='FS cold')
        ax1.legend()
        
        self.lines['fig1']['T_RS_hot'], = ax2.plot([], [], 'b-', linewidth=2, label='RS hot')
        self.lines['fig1']['T_RS_cold'], = ax2.plot([], [], 'b--', linewidth=2, label='RS cold')
        self.lines['fig1']['T_FS_hot'], = ax2.plot([], [], 'r-', linewidth=2, label='FS hot')
        self.lines['fig1']['T_FS_cold'], = ax2.plot([], [], 'r--', linewidth=2, label='FS cold')
        ax2.legend()
        
        self.lines['fig1']['H_RS_hot'], = ax3.plot([], [], 'b-', linewidth=2, label='RS hot')
        self.lines['fig1']['H_RS_cold'], = ax3.plot([], [], 'b--', linewidth=2, label='RS cold')
        self.lines['fig1']['H_RS_total'], = ax3.plot([], [], 'b:', linewidth=2, label='RS total')
        self.lines['fig1']['H_FS_hot'], = ax3.plot([], [], 'r-', linewidth=2, label='FS hot')
        self.lines['fig1']['H_FS_cold'], = ax3.plot([], [], 'r--', linewidth=2, label='FS cold')
        self.lines['fig1']['H_FS_total'], = ax3.plot([], [], 'r:', linewidth=2, label='FS total')
        ax3.legend()
        
        self.lines['fig1']['v_perp_RS'], = ax4.plot([], [], 'b-', linewidth=2, label=r'$v_{\perp}$ RS')
        self.lines['fig1']['v_tan_RS'], = ax4.plot([], [], 'b--', linewidth=2, label=r'$v_{tan}$ RS')
        self.lines['fig1']['v_perp_FS'], = ax4.plot([], [], 'r-', linewidth=2, label=r'$v_{\perp}$ FS')
        self.lines['fig1']['v_tan_FS'], = ax4.plot([], [], 'r--', linewidth=2, label=r'$v_{tan}$ FS')
        ax4.legend(fontsize=8)
        
        self.lines['fig1']['ratio_RS'], = ax5.plot([], [], 'b-', linewidth=2, label='RS')
        self.lines['fig1']['ratio_FS'], = ax5.plot([], [], 'r-', linewidth=2, label='FS')
        ax5.legend()
        
        self.fig1.tight_layout()
    
    def create_figure2(self):
        self.fig2 = plt.figure(figsize=(15, 12))
        self.fig2.suptitle('Emission Maps and Radial Profiles', fontsize=16)
        self.star_markers = {'Halpha': None, 'OIII': None, 'ff': None}
        self.arrows = {'Halpha': None, 'OIII': None, 'ff': None}
        
        # Top row: maps
        ax_Halpha = self.fig2.add_subplot(2, 3, 1)
        ax_OIII = self.fig2.add_subplot(2, 3, 2)
        ax_ff = self.fig2.add_subplot(2, 3, 3)
       
        # Bottom row: radial profiles
        ax_radial_profiles = self.fig2.add_subplot(2, 3, (4,6))
        
        # Configure map axes
        for ax, title in zip([ax_Halpha, ax_OIII, ax_ff], 
                           [r'H$\alpha$', '[O III]', 'Free-free']):
            ax.set_xlabel('x [arcsec]')
            ax.set_ylabel('y [arcsec]')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
        
        # Configure profile axes
        ax_radial_profiles.set_title('Normalizaed radial Emission Profiles')
        ax_radial_profiles.set_xlabel('Radius [arcsec]')
        ax_radial_profiles.set_ylabel('')
        ax_radial_profiles.set_xlim(0,150)
        
        # Store axes and images
        self.fig2_axes = {
            'maps': [ax_Halpha, ax_OIII, ax_ff],
            'profiles' : ax_radial_profiles
        }
        
        # Initialize image placeholders
        self.images = {
            'Halpha': None,
            'OIII': None,
            'ff': None
        }
        
        self.colorbars = {
            'Halpha': None,
            'OIII': None,
            'ff': None
        }
        
        self.profiles = {
            'Halpha': None,
            'OIII': None,
            'ff': None
        }

        self.profiles['Halpha'], = self.fig2_axes['profiles'].plot([], [], 'r-', linewidth=2, label=r'H$\alpha$')
        self.profiles['OIII'], = self.fig2_axes['profiles'].plot([], [], 'g-.', linewidth=2, label='[O III]')
        self.profiles['ff'], = self.fig2_axes['profiles'].plot([], [], 'b--', linewidth=2, label='Free-free')

        self.fig2_axes['profiles'].legend()
       
        self.fig2.tight_layout()
    
    
    def create_sliders(self):
        """Create sliders in a separate window"""
        self.fig_sliders = plt.figure(figsize=(12, 8))
        self.fig_sliders.suptitle('Bow Shock Parameters', fontsize=16)
        
        left_x = 0.15
        width = 0.7
        height = 0.04
        start_y = 0.85
        spacing = 0.055
        
        all_params = [
            ('Mdot_log', r'log₁₀(Mdot [M☉/yr])', (-12, -5), np.log10(self.Mdot_msun)),
            ('Vw', 'Vw [km/s]', (50, 5000), self.Vw/1e5),
            ('Vstar', 'Vstar [km/s]', (10, 500), self.Vstar/1e5),
            ('n_ism', 'n_ISM [cm⁻³]', (0.01, 10.0), self.n_ism),
            ('T_ism', 'T_ISM [K]', (100, 1.e6), self.T_ism),
            ('inclination', 'Inclination [°]', (0, 90), self.inclination),
        ]
        
        for i, (name, label, vrange, valinit) in enumerate(all_params):
            ax = self.fig_sliders.add_axes([left_x, start_y - i*spacing, width, height])
            if name == 'T_ism':
                slider = Slider(ax, label, vrange[0], vrange[1], valinit=valinit, valfmt='%.0f')
            else:
                slider = Slider(ax, label, vrange[0], vrange[1], valinit=valinit, valfmt='%.4f')
            slider.on_changed(self.update_all)
            self.sliders[name] = slider
        
        ax_reset = self.fig_sliders.add_axes([0.4, 0.02, 0.2, 0.04])
        btn_reset = Button(ax_reset, 'Reset All')
        btn_reset.on_clicked(self.reset_all)
        
        plt.show(block=False)
    
    def reset_all(self, event):
        """Reset all sliders to initial values"""
        self.Mdot_msun = 1e-9
        self.Mdot = 1e-9 * Msun_yr
        self.Vw = 100.0 * 1e5
        self.Vstar = 128.5 * 1e5
        self.n_ism = 0.2
        self.T_ism = 8000.0
        self.inclination = 75.0
        
        for name, slider in self.sliders.items():
            if name == 'Mdot_log':
                val = np.log10(self.Mdot_msun)
            elif name == 'Vw':
                val = self.Vw/1e5
            elif name == 'Vstar':
                val = self.Vstar/1e5
            elif name == 'T_ism':
                val = self.T_ism
            else:
                val = getattr(self, name)
            slider.set_val(val)
        
        self.update_lam_from_T_ism()
        self.update_R_RS_func()
    
    def update_figure1(self):
        prof = self.compute_thermo()
        
        self.lines['fig1']['n_RS_hot'].set_data(prof['theta'], prof['n_RS_hot'])
        self.lines['fig1']['n_RS_cold'].set_data(prof['theta'], prof['n_RS_cold'])
        self.lines['fig1']['n_FS_hot'].set_data(prof['theta'], prof['n_FS_hot'])
        self.lines['fig1']['n_FS_cold'].set_data(prof['theta'], prof['n_FS_cold'])
        
        self.lines['fig1']['T_RS_hot'].set_data(prof['theta'], prof['T_RS_hot'])
        self.lines['fig1']['T_RS_cold'].set_data(prof['theta'], prof['T_RS_cold'])
        self.lines['fig1']['T_FS_hot'].set_data(prof['theta'], prof['T_FS_hot'])
        self.lines['fig1']['T_FS_cold'].set_data(prof['theta'], prof['T_FS_cold'])
        
        self.lines['fig1']['H_RS_hot'].set_data(prof['theta'], prof['H_RS_hot'])
        self.lines['fig1']['H_RS_cold'].set_data(prof['theta'], prof['H_RS_cold'])
        self.lines['fig1']['H_RS_total'].set_data(prof['theta'], prof['H_RS_total'])
        self.lines['fig1']['H_FS_hot'].set_data(prof['theta'], prof['H_FS_hot'])
        self.lines['fig1']['H_FS_cold'].set_data(prof['theta'], prof['H_FS_cold'])
        self.lines['fig1']['H_FS_total'].set_data(prof['theta'], prof['H_FS_total'])
        
        self.lines['fig1']['v_perp_RS'].set_data(prof['theta'], prof['v_perp_RS']/1e5)
        self.lines['fig1']['v_tan_RS'].set_data(prof['theta'], prof['v_tan_RS']/1e5)
        self.lines['fig1']['v_perp_FS'].set_data(prof['theta'], prof['v_perp_FS']/1e5)
        self.lines['fig1']['v_tan_FS'].set_data(prof['theta'], prof['v_tan_FS']/1e5)
        
        self.lines['fig1']['ratio_RS'].set_data(prof['theta'], prof['ratio_RS'])
        self.lines['fig1']['ratio_FS'].set_data(prof['theta'], prof['ratio_FS'])
        
        all_ratios = np.concatenate([prof['ratio_RS'][np.isfinite(prof['ratio_RS'])],
                                      prof['ratio_FS'][np.isfinite(prof['ratio_FS'])]])
        if len(all_ratios) > 0:
            ymin = max(1e-6, np.min(all_ratios) * 0.5)
            ymax = np.max(all_ratios) * 2
            self.fig1_axes[4].set_ylim(ymin, ymax)
        
        self.fig1_axes[4].set_xlim(0, np.max(prof['theta']))
        
        for ax in self.fig1_axes[:4]:
            ax.relim()
            ax.autoscale_view()
        
        self.fig1.canvas.draw_idle()
    
    def update_figure2(self):
        try:
            maps = self.compute_maps()
            
            # Clean data
            I_Halpha = sanitize_log_data(maps['I_Halpha'])
            I_OIII   = sanitize_log_data(maps['I_OIII'])
            if (self.band_name == 'radio'):
                I_ff = sanitize_log_data(maps['I_ff_mJy'])
            else:
                I_ff = sanitize_log_data(maps['I_ff_total'])

            extent = [
                np.min(maps['x']), np.max(maps['x']),
                np.min(maps['y']), np.max(maps['y'])
            ]

            #Projected values
            R0_phys = self.update_R0()
            # Taking into account ISM thermal pressure
            R0_corrected = R0_phys / np.sqrt(1 + self.alpha)

            inc = np.deg2rad(self.inclination)

            # Star translation from the origin
            dx_star = arcsecond(R0_corrected * np.cos(inc), self.distance)
            dy_star = 0.0

            # Rotate according to PA. PA = 0 -> BS pointing towards north, measured from north to east
            PA_rot = np.deg2rad(self.PA - 90.)

            x_star = dx_star*np.cos(PA_rot) + dy_star*np.sin(PA_rot)
            y_star = dx_star*np.sin(PA_rot) - dy_star*np.cos(PA_rot)

            # Apex
            x0 = 0.0
            y0 = 0.0

            # Map limits
            xmap_min = np.min(maps['x'])
            xmap_max = np.max(maps['x'])

            ymap_min = np.min(maps['y'])
            ymap_max = np.max(maps['y'])

            zoom_factor = 10.0

            xmin = -zoom_factor * arcsecond(R0_corrected, self.distance)
            xmax =  zoom_factor * arcsecond(R0_corrected, self.distance)

            ymin = -zoom_factor * arcsecond(R0_corrected, self.distance)
            ymax =  zoom_factor * arcsecond(R0_corrected, self.distance)

            # Emission maps
            data_list = [I_Halpha, I_OIII, I_ff]
            keys = ['Halpha', 'OIII', 'ff']
            cmaps = ['inferno', 'inferno', 'inferno']
            
            for i, (key, I_data, cmap) in enumerate(zip(keys, data_list, cmaps)):
                ax = self.fig2_axes['maps'][i]
                img = self.images[key]
                
                norm = get_norm(I_data)
                
                if img is None:
                    img_obj = ax.imshow(
                        I_data,
                        origin='lower',
                        extent=extent,
                        cmap=cmap,
                        norm=norm
                    )

                    # contour leves
                    ax.contour(
                        maps['x'],
                        maps['y'],
                        I_data,
                        levels=np.max(I_data) * np.array([0.01, 0.03, 0.1, 0.6]),
                        colors='lime'
                    )
                    
                    self.images[key] = img_obj
                    if key == 'ff' and self.band_name == 'radio':
                        cbar_label = r'Surface brightness [mJy arcsec$^{-2}$]'
                    elif key == 'Halpha':
                        cbar_label = r'Surface brightness [R]'
                    else:
                        cbar_label = r'Surface brightness [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]'

                    self.colorbars[key] = plt.colorbar(
                        img_obj,
                        ax=ax,
                        label=cbar_label
                        )

                else:
                    img.set_data(I_data)
                    img.set_extent(extent)
                    img.set_norm(norm)

                # Star
                if self.star_markers[key] is None:
                    star, = ax.plot(
                        x_star, y_star,
                        marker='*',
                        color='black',
                        markersize=10,
                        zorder=5
                    )
                    self.star_markers[key] = star
                else:
                    self.star_markers[key].set_data([x_star], [y_star])

                # Arrow towars R0
                dx_arrow = x0 - x_star
                dy_arrow = y0 - y_star

                if self.arrows[key] is not None:
                    self.arrows[key].remove()

                arr = ax.arrow(
                    x_star, y_star,
                    dx_arrow, dy_arrow,
                    color='black',
                    width=0.0,
                    head_width=0.05 * arcsecond(R0_corrected, self.distance),
                    length_includes_head=True,
                    zorder=5
                )
                self.arrows[key] = arr

            # Zoom
            for ax in self.fig2_axes['maps']:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect('equal')

            # Radial profiles
            if np.any(I_Halpha > 0):
                r_prof, prof_Halpha = radial_profile(
                    maps['x'], maps['y'], I_Halpha,
                    dX=0.0, nbins=33, r_min=0.0, r_max=150.0
                )
                _, prof_OIII = radial_profile(
                    maps['x'], maps['y'], I_OIII,
                    dX=0.0, nbins=33, r_min=0.0, r_max=150.0
                )
                _, prof_ff = radial_profile(
                    maps['x'], maps['y'], I_ff,
                    dX=0.0, nbins=33, r_min=0.0, r_max=150.0
                )
                
                # Normalize each profile to its maximum
                prof_Halpha_norm = prof_Halpha / np.max(prof_Halpha)
                prof_OIII_norm = prof_OIII / np.max(prof_OIII)
                prof_ff_norm = prof_ff / np.max(prof_ff)
                
                # Update profiles with normalized data
                self.profiles['Halpha'].set_data(r_prof, prof_Halpha_norm)
                self.profiles['OIII'].set_data(r_prof, prof_OIII_norm)
                self.profiles['ff'].set_data(r_prof, prof_ff_norm)
                
                # Yaxis covering shorter range
                self.fig2_axes['profiles'].set_ylim(0, 1.1)
                self.fig2_axes['profiles'].set_ylabel('Normalized Intensity')
                
                self.fig2_axes['profiles'].relim()
                self.fig2_axes['profiles'].autoscale_view(scaley=False)
                
        except Exception as e:
            print(f"Error in update_figure2: {e}")
            import traceback
            traceback.print_exc()
        
        self.fig2.canvas.draw_idle()
        
    
        
    def update_all(self, val):
        """Update all figures"""
        self.get_params_from_sliders()
        self.update_figure1()
        self.update_figure2()
    
    def run(self):
        """Run the application"""
        print("Creating figures...")
        self.create_figure1()
        self.create_figure2()
        self.create_sliders()
        
        print("Initial update...")
        self.update_all(None)
        
        print("Showing plots...")
        plt.show()
        print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bow Shock Interactive Visualizer',
        epilog="""
E.g.:
  python3 bow_shock.py
  python3 bow_shock.py --source RXJ0528+2838
  python3 bow_shock.py --source RXJ0528+2838 --params-dir ./params_file
  python3 bow_shock.py --list-bands

Available frequencies:
  radio, IR, optical_R, optical_V, optical_B, FUV, EUV, Xray_soft, Xray_hard
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='RXJ0528+2838',
        help='Source name (default: RXJ0528+2838)'
    )
    
    parser.add_argument(
        '--params-dir', '-p',
        type=str,
        default='Systems',
        help='Path to parameters file (default: Systems)'
    )
    
    parser.add_argument(
        '--list-bands', '-l',
        action='store_true',
        help='List the available spectrum bands to compute monoenergetic free-free emission'
    )
    
    parser.add_argument(
        '--band', '-b',
        type=str,
        default=None,
        help='Spectrum band to compute free-free emission (e.g.: FUV, IR, Xray_soft)'
    )
    
    args = parser.parse_args()
    
    if args.list_bands:
        print("Available spectrum bands to calculate free-free:")
        bands = {
            'radio': '3 GHz - Radio',
            'IR': '3 THz - Mid infrared',
            'optical_R': '700 nm - Opt, red',
            'optical_V': '545 nm - Opt, green',
            'optical_B': '435 nm - Opt, blue',
            'FUV': '150 nm - Far ultraviolet',
            'EUV': '10 nm - Extreme ultraviolet',
            'Xray_soft': '1 keV - Soft x-rays',
            'Xray_hard': '10 keV - Hard x-rays'
        }
        for band, desc in bands.items():
            print(f"  {band:12s} : {desc}")
        sys.exit(0)
    
    print(f"Loading. Source: {args.source}, params_dir: {args.params_dir}")
    app = BowShock(args.source, args.params_dir)
    
    if args.band:
        app.set_continuum_band(args.band)
    
    app.run()