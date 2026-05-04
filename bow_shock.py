#!/usr/bin/env python3
"""
Bow Shock Interactive Visualizer
run as: python3 bow_shock.py [source_name] [params_dir]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

from constants import AU, pc, Msun_yr, mu, mp, mu_sh, h, kB
from params_loader import get_source_params, validate_params
from bow_shock_surface import (
    standoff_distance,
    integrate_r_theta_christie,
    bow_shock_surface_christie
)
from thermodynamics import (
    post_shock_conditions,
    compute_layer_thickness,
    normalized_thickness,
    vnorm_forward,
    vnorm_wind,
    vtan,
    cooling_time,
    advection_time,
    pre_shock,
    mach_number,
    compression_factor_rh,
    temperature_post_rh,
    density_post_rh
)
from utils import (
    make_projection_maps_fast,
    radial_profile,
    arcsecond
)


class BowShock:
    def __init__(self, source_name='RXJ0528+2838', params_dir='.'):
        """
        Initialize bow shock model with parameters from file.
        
        Parameters:
        -----------
        source_name : str
            Name of the source (e.g., 'RXJ0528+2838')
        params_dir : str
            Directory containing parameter files
        """
        print("Loading parameters...")
        # Load parameters from file
        self.params = get_source_params(source_name, params_dir)
        validate_params(self.params)
        
        # Extract parameters with CONVERSIONS to CGS
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
        self.nz = self.params.get('nz', 75)
        self.nx = self.params.get('nx', 50)
        self.ny = self.params.get('ny', 50)
        self.xlim_factor = self.params.get('xlim_factor', 5.0)
        self.ylim_factor = self.params.get('ylim_factor', 5.0)
        
        # Frequency for free-free emission [Hz]
        self.nu_ff = self.params.get('nu_ff', 2e6 * 1e9)   # 2e6 GHz == 1500 A -> FUV
        
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
        self.theta_grid = np.linspace(0.01, np.pi - 0.01, 500)
        
        # Load initial R_RS function
        print("Loading R_RS function...")
        self.update_R_RS_func()
        
        # Calculate R0 for verification
        R0_test = self.update_R0()
        print(f"R0 = {R0_test/AU:.1f} AU")
        print(f"Projected stagnation point distance = {arcsecond(R0_test*np.cos(self.inclination*np.pi/180.), self.distance)} ''")
        
        print(f"Loaded parameters for {source_name}")
        print(f"  Mdot = {self.Mdot_msun:.2e} Msun/yr = {self.Mdot:.2e} g/s")
        print(f"  Vw = {self.Vw/1e5:.1f} km/s")
        print(f"  Vstar = {self.Vstar/1e5:.1f} km/s")
        print(f"  n_ism = {self.n_ism:.2f} cm^-3")
        print(f"  T_ism = {self.T_ism:.2f} K")
        print(f"  inclination = {self.inclination:.1f} deg")
        print("Initialization complete!")
    
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
        self.lmb = np.log10(self.lam) if self.lam > 0 else -10
    
    def update_R0(self):
        """Calculate standoff radius [cm]"""
        return standoff_distance(self.Mdot, self.Vw, self.Vstar, self.n_ism)
    
    def update_R_RS_func(self):
        """
        Update the R_RS function using the Christie ODE integration.
        This matches exactly what the notebook does.
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
    
    def compute_profiles(self):
        R0_phys = self.update_R0()
        lam = self.lam
        alpha = lam/(1-lam) if lam < 1 else 0
        R0_corrected = R0_phys * np.sqrt(1/(1+alpha))
        
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
        lam = self.lam
        alpha = lam/(1-lam) if lam < 1 else 0
        R0_corrected = R0_phys * np.sqrt(1/(1+alpha))
        
        xlim = self.xlim_factor * R0_corrected
        ylim = self.ylim_factor * R0_corrected
        inclination_rad = np.deg2rad(90 - self.inclination)
        zmax = max(self.zmax, 5 * R0_corrected)
        
        x_vals_arcsec, y_vals_arcsec, result = make_projection_maps_fast(
            xlim, ylim, self.nx, self.ny, self.R_RS_func,
            inclination=inclination_rad, zmax=zmax, nz=self.nz,
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
            'I_ff_total': result['I_ff_total']
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
        ax3.set_yscale('log')
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
        
        # Top row: maps
        ax_Halpha = self.fig2.add_subplot(2, 3, 1)
        ax_OIII = self.fig2.add_subplot(2, 3, 2)
        ax_ff = self.fig2.add_subplot(2, 3, 3)
       
        # Bottom row: radial profiles
        ax_prof_Halpha = self.fig2.add_subplot(2, 3, 4)
        ax_prof_OIII = self.fig2.add_subplot(2, 3, 5)
        ax_prof_ff = self.fig2.add_subplot(2, 3, 6)
        
        # Configure map axes
        for ax, title in zip([ax_Halpha, ax_OIII, ax_ff], 
                           [r'H$\alpha$', '[O III]', 'Free-free']):
            ax.set_xlabel('x [arcsec]')
            ax.set_ylabel('y [arcsec]')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
        
        # Configure profile axes
        for ax, title in zip([ax_prof_Halpha, ax_prof_OIII, ax_prof_ff],
                            [r'H$\alpha$ Profile', '[O III] Profile', 'Free-free Profile']):
            ax.set_xlabel('Radius [arcsec]')
            ax.set_ylabel('Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 250)
        
        # Store axes and images
        self.fig2_axes = {
            'maps': [ax_Halpha, ax_OIII, ax_ff],
            'profiles': [ax_prof_Halpha, ax_prof_OIII, ax_prof_ff]
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
        
        # Initialize profile lines
        for i, key in enumerate(['Halpha', 'OIII', 'ff']):
            self.profiles[key], = self.fig2_axes['profiles'][i].plot([], [], 'b-', linewidth=2)
       
        self.fig2.tight_layout()
    
    def create_figure3(self):
        """
        Creates third figure with RGB composite of all three emission maps.
        Three independent colorbars with correct colormaps (Reds, Greens, Purples).
        """
        self.fig3 = plt.figure(figsize=(14, 10))
        self.fig3.suptitle('Overlaid Emission Maps (RGB Composite)', fontsize=16)
        
        # Main axes for the RGB image
        self.fig3_ax = self.fig3.add_subplot(1, 1, 1)
        self.fig3_ax.set_xlabel('x [arcsec]')
        self.fig3_ax.set_ylabel('y [arcsec]')
        self.fig3_ax.set_title('Hα (Red) + [O III] (Green) + Free-free (Blue)')
        self.fig3_ax.set_aspect('equal')
        self.fig3_ax.grid(True, alpha=0.2)
        
        self.rgb_image = None
        self.fig3_colorbars = {}
        
        self.fig3.tight_layout()
    
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
            ('T_ism', 'T_ISM [K]', (10, 1e6), self.T_ism),
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
        prof = self.compute_profiles()
        
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
            
            extent = [maps['x'][0], maps['x'][-1], maps['y'][0], maps['y'][-1]]
            
            # Halpha
            if self.images['Halpha'] is None:
                self.images['Halpha'] = self.fig2_axes['maps'][0].imshow(
                    maps['I_Halpha'], origin='lower', extent=extent,
                    cmap='magma', norm=LogNorm()
                )
                self.colorbars['Halpha'] = plt.colorbar(self.images['Halpha'], 
                                                         ax=self.fig2_axes['maps'][0],
                                                         label='Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
            else:
                self.images['Halpha'].set_data(maps['I_Halpha'])
                self.images['Halpha'].set_extent(extent)
                if np.any(np.isfinite(maps['I_Halpha'])):
                    vmin_val = np.nanmin(maps['I_Halpha'])
                    vmax_val = np.nanmax(maps['I_Halpha'])
                    if vmin_val < vmax_val:
                        self.images['Halpha'].set_clim(vmin=vmin_val, vmax=vmax_val)
            
            # OIII
            if self.images['OIII'] is None:
                self.images['OIII'] = self.fig2_axes['maps'][1].imshow(
                    maps['I_OIII'], origin='lower', extent=extent,
                    cmap='magma', norm=LogNorm()
                )
                self.colorbars['OIII'] = plt.colorbar(self.images['OIII'],
                                                       ax=self.fig2_axes['maps'][1],
                                                       label='Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
            else:
                self.images['OIII'].set_data(maps['I_OIII'])
                self.images['OIII'].set_extent(extent)
                if np.any(np.isfinite(maps['I_OIII'])):
                    vmin_val = np.nanmin(maps['I_OIII'])
                    vmax_val = np.nanmax(maps['I_OIII'])
                    if vmin_val < vmax_val:
                        self.images['OIII'].set_clim(vmin=vmin_val, vmax=vmax_val)
            
            # Free-free total
            if self.images['ff'] is None:
                self.images['ff'] = self.fig2_axes['maps'][2].imshow(
                    maps['I_ff_total'], origin='lower', extent=extent,
                    cmap='magma', norm=LogNorm()
                )
                self.colorbars['ff'] = plt.colorbar(self.images['ff'],
                                                     ax=self.fig2_axes['maps'][2],
                                                     label='Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
            else:
                self.images['ff'].set_data(maps['I_ff_total'])
                self.images['ff'].set_extent(extent)
                if np.any(np.isfinite(maps['I_ff_total'])):
                    vmin_val = np.nanmin(maps['I_ff_total'])
                    vmax_val = np.nanmax(maps['I_ff_total'])
                    if vmin_val < vmax_val:
                        self.images['ff'].set_clim(vmin=vmin_val, vmax=vmax_val)
            
            # Radial profiles
            if np.any(maps['I_Halpha'] > 0):
                r, prof_Halpha = radial_profile(maps['x'], maps['y'], maps['I_Halpha'],
                                                 dX=0.0, nbins=33, r_min=0.0, r_max=250.0)
                _, prof_OIII = radial_profile(maps['x'], maps['y'], maps['I_OIII'],
                                              dX=0.0, nbins=33, r_min=0.0, r_max=250.0)
                _, prof_ff = radial_profile(maps['x'], maps['y'], maps['I_ff_total'],
                                             dX=0.0, nbins=33, r_min=0.0, r_max=250.0)
               
                self.profiles['Halpha'].set_data(r, prof_Halpha)
                self.profiles['OIII'].set_data(r, prof_OIII)
                self.profiles['ff'].set_data(r, prof_ff)
                
                for ax, prof in zip(self.fig2_axes['profiles'], 
                                   [prof_Halpha, prof_OIII, prof_ff]):
                    if len(prof) > 0 and np.max(prof) > 0:
                        ax.set_ylim(0, np.max(prof) * 1.1)
                    ax.relim()
                    ax.autoscale_view()
           
        except Exception as e:
            print(f"Error in update_figure2: {e}")
            import traceback
            traceback.print_exc()
       
        self.fig2.canvas.draw_idle()
    
    def update_figure3(self):
        """
        Update the RGB composite figure.
        Halpha -> Red channel, OIII -> Green channel, Free-free -> Blue channel.
        Three separate colorbars with correct colormaps (Reds, Greens, Purples).
        """
        try:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib as mpl
            
            maps = self.compute_maps()
            
            extent = [maps['x'][0], maps['x'][-1], maps['y'][0], maps['y'][-1]]
            
            def normalize_map_log(img):
                """Normalize image to [0, 1] using log scaling."""
                img_finite = img[np.isfinite(img)]
                if len(img_finite) == 0:
                    return np.zeros_like(img), 1e-10, 1.0
                
                # Use log10 scaling
                img_pos = img_finite[img_finite > 0]
                if len(img_pos) == 0:
                    return np.zeros_like(img), 1e-10, 1.0
                
                vmin = np.percentile(img_pos, 1)
                vmax = np.percentile(img_pos, 99)
                
                if vmax <= vmin:
                    vmax = vmin * 100
                
                # Log normalization
                log_img = np.log10(np.maximum(img, vmin/10))
                log_vmin = np.log10(vmin)
                log_vmax = np.log10(vmax)
                norm_img = (log_img - log_vmin) / (log_vmax - log_vmin)
                norm_img = np.clip(norm_img, 0, 1)
                
                return norm_img, vmin, vmax
            
            # Normalize each channel
            red, vmin_r, vmax_r = normalize_map_log(maps['I_Halpha'])
            green, vmin_g, vmax_g = normalize_map_log(maps['I_OIII'])
            blue, vmin_b, vmax_b = normalize_map_log(maps['I_ff_total'])
            
            # Store vmin/vmax for colorbars
            self.channel_limits = {
                'Halpha': {'vmin': vmin_r, 'vmax': vmax_r},
                'OIII': {'vmin': vmin_g, 'vmax': vmax_g},
                'ff': {'vmin': vmin_b, 'vmax': vmax_b}
            }
            
            # Stack into RGB array
            rgb = np.stack([red, green, blue], axis=-1)
            
            # Create or update RGB image
            if self.rgb_image is None:
                self.rgb_image = self.fig3_ax.imshow(rgb, origin='lower', extent=extent)
                
                # Create three separate colorbar axes using make_axes_locatable
                divider = make_axes_locatable(self.fig3_ax)
                
                # Colorbar 1: Halpha (Reds colormap)
                ax_cbar_halpha = divider.append_axes("right", size="5%", pad=0.05)
                norm_halpha = mpl.colors.Normalize(vmin=vmin_r, vmax=vmax_r)
                sm_halpha = mpl.cm.ScalarMappable(norm=norm_halpha, cmap='Reds')
                sm_halpha.set_array(maps['I_Halpha'])
                cbar_halpha = self.fig3.colorbar(sm_halpha, cax=ax_cbar_halpha)
                cbar_halpha.set_label('Hα Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
                
                # Colorbar 2: OIII (Greens colormap)
                ax_cbar_oiii = divider.append_axes("right", size="5%", pad=0.35)
                norm_oiii = mpl.colors.Normalize(vmin=vmin_g, vmax=vmax_g)
                sm_oiii = mpl.cm.ScalarMappable(norm=norm_oiii, cmap='Greens')
                sm_oiii.set_array(maps['I_OIII'])
                cbar_oiii = self.fig3.colorbar(sm_oiii, cax=ax_cbar_oiii)
                cbar_oiii.set_label('[O III] Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
                
                # Colorbar 3: Free-free (Purples colormap)
                ax_cbar_ff = divider.append_axes("right", size="5%", pad=0.65)
                norm_ff = mpl.colors.Normalize(vmin=vmin_b, vmax=vmax_b)
                sm_ff = mpl.cm.ScalarMappable(norm=norm_ff, cmap='Purples')
                sm_ff.set_array(maps['I_ff_total'])
                cbar_ff = self.fig3.colorbar(sm_ff, cax=ax_cbar_ff)
                cbar_ff.set_label('Free-free Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
                
                self.fig3_colorbars = {
                    'Halpha': cbar_halpha,
                    'OIII': cbar_oiii,
                    'ff': cbar_ff
                }
                
            else:
                # Update existing RGB image
                self.rgb_image.set_data(rgb)
                self.rgb_image.set_extent(extent)
                
                # Update colorbars with new limits
                norm_halpha = mpl.colors.Normalize(vmin=vmin_r, vmax=vmax_r)
                self.fig3_colorbars['Halpha'].set_norm(norm_halpha)
                self.fig3_colorbars['Halpha'].set_label('Hα Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
                
                norm_oiii = mpl.colors.Normalize(vmin=vmin_g, vmax=vmax_g)
                self.fig3_colorbars['OIII'].set_norm(norm_oiii)
                self.fig3_colorbars['OIII'].set_label('[O III] Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
                
                norm_ff = mpl.colors.Normalize(vmin=vmin_b, vmax=vmax_b)
                self.fig3_colorbars['ff'].set_norm(norm_ff)
                self.fig3_colorbars['ff'].set_label('Free-free Intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]')
            
            self.fig3.canvas.draw_idle()
        
        except Exception as e:
            print(f"Error in update_figure3: {e}")
            import traceback
            traceback.print_exc()
        
    def update_all(self, val):
        """Update all figures"""
        self.get_params_from_sliders()
        self.update_figure1()
        self.update_figure2()
        self.update_figure3()
    
    def run(self):
        """Run the application"""
        print("Creating figures...")
        self.create_figure1()
        self.create_figure2()
        self.create_figure3()
        self.create_sliders()
        
        print("Initial update...")
        self.update_all(None)
        
        print("Showing plots...")
        plt.show()
        print("Done.")


if __name__ == "__main__":
    import sys
    
    source_name = 'RXJ0528+2838'
    params_dir = '.'
    
    if len(sys.argv) > 1:
        source_name = sys.argv[1]
    if len(sys.argv) > 2:
        params_dir = sys.argv[2]
    
    print(f"Starting with source: {source_name}, params_dir: {params_dir}")
    app = BowShock(source_name, params_dir)
    app.run()