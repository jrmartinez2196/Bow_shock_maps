"""Physical and astronomical constants, in cgs units."""

mu = 1.3  # mean molecular weight
mu_sh = 0.6  # shocked-region and unshocked-wind mean molecular weight
mp = 1.6726219e-24  # proton mass [g]
kB = 1.380649e-16  # Boltzmann constant [erg/K]
me = 9.1093837e-28  # electron mass [g]
qe = 4.8032068e-10  # electron charge [esu]
c = 2.99792458e10  # speed of light [cm/s]
G = 6.67430e-8  # gravitational constant [cm^3 g^-1 s^-2]
eV = 1.602176e-12  # electronvolt [erg]
mec2 = 0.511e6 * eV  # electron rest energy [erg]
sigma_T = 6.6524587e-25  # Thomson cross section [cm^2]
Ry = 2.179872e-11  # Rydberg constant [erg]
h = 6.6260755e-27  # Planck constant [erg s]
sr_per_arcsec2 = (206265.0) ** 2  # steradians per square arcsecond
Rayleigh = 5.66e-18  # erg/s/cm^2/arcsec^2 to Rayleigh; useful for H-alpha

Msun = 1.989e33  # solar mass [g]
Rsun = 6.957e10  # solar radius [cm]
Lsun = 3.828e33  # solar luminosity [erg/s]
AU = 1.495978707e13  # astronomical unit [cm]
pc = 3.085677581e18  # parsec [cm]
year = 3.15576e7  # Julian year [s]

Msun_yr = Msun / year  # solar masses per year, in cgs (g/s)

gamma_ad = 5.0 / 3.0  # adiabatic index

z_w = 1.23  # mean charge, stellar wind
z_ism = 1.0  # mean charge, ISM
