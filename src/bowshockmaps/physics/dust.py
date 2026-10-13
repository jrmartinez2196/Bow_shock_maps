"""Thermal infrared emission of the dust of the shocked ISM.

Only the forward-shock layers carry dust: the swept-up, compressed ISM. The
stellar wind and the reverse-shock layers have none (del Valle & Pohl 2018,
Sec. 4.4.2; a white-dwarf wind has no dust either).

Each grain (radius ``a``, all grains equal) is in thermal equilibrium: what it
absorbs equals what it radiates. It is heated by

* collisions with the electrons and protons of the shocked gas (the dominant
  term for a faint central source),
* the starlight, absorbed with unit efficiency (UV) over its cross section
  and diluted as r^-2 (the heating of del Valle & Pohl's Eq. 11; the star is
  not attenuated by the dust it heats), and
* the radiation of the shocks themselves, which Iłkiewicz et al. (2026, ApJL)
  propose as the likely heater of the dust of BZ Cam ("shock-generated UV
  photons"). It is the energy the layers radiate, ``shock_radiated_luminosity``,
  added to the stellar luminosity as one more source at the star (an
  optically thin, order-unity-geometry approximation: the dust of the
  forward shock sits at about the distance of the shell that radiates).

and radiates as a modified blackbody, with the emission efficiency
Q_em(nu) = Q100 (a / 0.1 um) (nu / nu_100um)^2 (a power law in frequency, as
in del Valle & Pohl; normalized to the far-infrared absorption of
~0.1 um silicate/graphite grains, kappa_abs(100 um) ~ 20 cm^2 g^-1).
Because Q_em goes as nu^2 the balance has a closed form, P_em = K T^6.

Limits of this treatment: grains at equilibrium only (no stochastic heating of
small grains, which is what lifts mid-infrared emission above the equilibrium
value), no sputtering, a single grain size, Q_em extrapolated as nu^2 into the
mid infrared (capped at 1).
"""

import numpy as np
from scipy.special import zeta

from bowshockmaps.constants import c, h, kB, me, mp, mu_sh
from bowshockmaps.physics.thermodynamics import lambda_T

# Dust material density [g cm^-3] (del Valle & Pohl 2018; Draine & Li 2007)
RHO_GRAIN = 2.0

# Emission efficiency at 100 um of a grain of radius 0.1 um
Q100 = 7e-4
A_REF_UM = 0.1
NU_100UM = c / 100e-4  # [Hz]

# int_0^inf nu^2 B_nu dnu = (2 h / c^2) (kT / h)^6 Gamma(6) zeta(6)
_INT_NU2_B = 120.0 * zeta(6)  # Gamma(6) = 120


def emission_efficiency(nu, a_um):
    """Q_em(nu) of a grain of radius ``a_um`` [um]: Q100 (a/0.1um) (nu/nu_100um)^2, at most 1."""
    q = Q100 * (a_um / A_REF_UM) * (np.asarray(nu, dtype=float) / NU_100UM) ** 2
    return np.minimum(q, 1.0)


def collisional_heating_per_grain(n_e, T_gas, a_um):
    """Power [erg/s] deposited in one grain by the electrons and protons of the gas.

    Each species delivers its flux onto the cross section pi a^2 with the
    mean energy 2 kT per collision, all of it absorbed (sticking and energy
    deposition efficiency of 1: at the temperatures of the shocked ISM the
    electrons stop within the grain). ``n_e`` is the electron density; the
    protons have the same.
    """
    a = a_um * 1e-4
    T = np.maximum(np.asarray(T_gas, dtype=float), 1.0)
    n_e = np.maximum(np.asarray(n_e, dtype=float), 0.0)
    v_e = np.sqrt(8.0 * kB * T / (np.pi * me))
    v_p = np.sqrt(8.0 * kB * T / (np.pi * mp))
    return np.pi * a**2 * 2.0 * kB * T * n_e * (v_e + v_p)


def stellar_heating_per_grain(r, L_star, a_um):
    """Power [erg/s] absorbed from the star (unit UV absorption efficiency) at distance ``r`` [cm]."""
    a = a_um * 1e-4
    r = np.maximum(np.asarray(r, dtype=float), 1.0)
    return np.pi * a**2 * L_star / (4.0 * np.pi * r**2)


def _cooling_constant(a_um):
    """K in P_em = K T^6, the power radiated by one grain [erg s^-1 K^-6]."""
    a = a_um * 1e-4
    q = Q100 * (a_um / A_REF_UM)
    return (
        4.0
        * np.pi**2
        * a**2
        * q
        * (2.0 * h / c**2)
        * (kB / h) ** 6
        * _INT_NU2_B
        / NU_100UM**2
    )


def dust_temperature(n_e, T_gas, r, L_star, a_um):
    """Equilibrium grain temperature [K]: collisional + stellar heating = emission."""
    heating = collisional_heating_per_grain(n_e, T_gas, a_um) + stellar_heating_per_grain(
        r, L_star, a_um
    )
    return (heating / _cooling_constant(a_um)) ** (1.0 / 6.0)


def dust_emissivity(n, n_e, T_gas, r, nu, L_star, a_um=0.1, f_dust_gas=0.01):
    """Dust emissivity [erg s^-1 cm^-3 Hz^-1 sr^-1] at frequency ``nu`` [Hz].

    j_nu = n_gr pi a^2 Q_em(nu) B_nu(T_gr), with n_gr = f_dust_gas rho / m_grain
    grains per unit volume, rho = n mu_sh m_p the gas density and
    m_grain = 4/3 pi a^3 rho_grain.

    Parameters
    ----------
    n : array
        Total particle density of the gas [cm^-3].
    n_e : array
        Electron density [cm^-3].
    T_gas : array
        Gas temperature [K].
    r : array
        Distance to the star [cm].
    nu : float
        Observing frequency [Hz].
    L_star : float
        Luminosity of the star [erg/s] that heats the grains (0 for none).
    a_um : float
        Grain radius [um].
    f_dust_gas : float
        Dust-to-gas mass ratio.
    """
    a = a_um * 1e-4
    T_d = dust_temperature(n_e, T_gas, r, L_star, a_um)
    rho = np.maximum(np.asarray(n, dtype=float), 0.0) * mu_sh * mp
    m_grain = 4.0 / 3.0 * np.pi * a**3 * RHO_GRAIN
    n_gr = f_dust_gas * rho / m_grain
    x = h * nu / (kB * T_d)
    # B_nu = 2 h nu^3 / c^2 / (exp(x) - 1); exp overflow -> no emission
    with np.errstate(over="ignore"):
        B_nu = 2.0 * h * nu**3 / c**2 / np.expm1(np.minimum(x, 700.0))
    B_nu = np.where(x > 700.0, 0.0, B_nu)
    return n_gr * np.pi * a**2 * emission_efficiency(nu, a_um) * B_nu


def shock_radiated_luminosity(shell, rs_props, fs_props):
    """Power radiated by the layers of each shock [erg/s], from the model itself.

    Each layer radiates n^2 Lambda(T) per unit volume (the same cooling
    function that decides whether it is radiative); integrated over its
    thickness and over the shell surface 2 pi rho dl. Returns a dict
    ``{"RS": (hot, cold), "FS": (hot, cold), "total": ...}``.

    This is the bolometric cooling radiation, an upper bound for the UV that can
    heat dust: gas of 1e5-1e6 K radiates mostly in the EUV/FUV lines, hotter gas
    mostly in X-rays.
    """
    theta = shell.theta
    dl = np.hypot(np.gradient(shell.C_rho), np.gradient(shell.C_z))
    dA = 2.0 * np.pi * np.abs(shell.C_rho) * dl

    out = {}
    total = 0.0
    for name, props in (("RS", rs_props), ("FS", fs_props)):
        n_hot = np.nan_to_num(props["n_post"](theta))
        T_hot = np.nan_to_num(props["T_post"](theta), nan=1.0)
        H_hot = np.nan_to_num(props["H_hot"](theta))
        n_cold = np.nan_to_num(props["n_IL"](theta))
        T_cold = np.nan_to_num(props["T_IL_arr"](theta), nan=1.0)
        H_cold = np.nan_to_num(props["H_cold"](theta))
        hot = float(np.sum(n_hot**2 * lambda_T(T_hot) * np.maximum(H_hot, 0.0) * dA))
        cold = float(np.sum(n_cold**2 * lambda_T(T_cold) * np.maximum(H_cold, 0.0) * dA))
        out[name] = (hot, cold)
        total += hot + cold
    out["total"] = total
    return out
