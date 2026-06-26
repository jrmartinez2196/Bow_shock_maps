# constants.py - cgs units
import numpy as np

mu = 1.3                    # mean molecular weight
mu_sh = 0.6 				# shocked regoin and unshocked wind mean molecular weight
mp = 1.6726219e-24          # proton mass
kB = 1.380649e-16           # Boltzmann constant [erg/K]
me = 9.1093837e-28          # electron mass
qe = 4.8032068e-10			# electron charge
c = 2.99792458e10           # speed of light
G = 6.67430e-8              # Gravitational constant
eV = 1.602176e-12
mec2 = 0.511e6*eV 			# electron rest energy
sigma_T = 6.6524587e-25     # Thomson cross section
Ry = 2.179872e-11  			# Rydberg constant [erg]
h = 6.6260755e-27			# Planck constant
sr_per_arcsec2 = (206265.)**2
Rayleigh = 5.66e-18  		# erg/s/cm^2/arcsec^2 to Rayleigh. Useful for Halpha

Msun = 1.989e33             # Solar mass
Rsun = 6.957e10             # Solar radius
Lsun = 3.828e33             # Solar luminosity
AU = 1.495978707e13
pc = 3.085677581e18
year = 3.15576e7

Msun_yr = Msun / year

gamma_ad = 5./3.			# Adiabatic coefficient

z_w = 1.23
z_ism = 1.