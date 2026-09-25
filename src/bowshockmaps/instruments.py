"""Telescope beam-size lookup, used to drive the convolution step.

Provides a simple diffraction-limited estimate of the beam FWHM for a
named telescope (single-dish, via its aperture diameter; or
interferometer, via its array configuration's maximum baseline) at a
given observing frequency:

    theta_FWHM ~ 1.22 * (c / freq) / D

This is a standard, order-of-magnitude resolution estimate, not a full
synthesized-beam calculation: for a real interferometric observation,
the actual beam shape/size also depends on the uv-coverage, the
source's declination, and the imaging weights used, none of which are
modeled here. Use `beam_fwhm_arcsec` as a reasonable starting point; if
you have a beam size from an actual observation or proposal tool,
override it directly instead (see the --beam-fwhm CLI flag / the
`beam_fwhm` argument to `BowShock`).
"""

from dataclasses import dataclass, field

import numpy as np

from bowshockmaps.constants import c

# c is in cgs (cm/s); convert to m/s for use with diameters/baselines in meters.
_C_M_S = c * 1e-2


@dataclass(frozen=True)
class Telescope:
    """A single-dish telescope or an interferometer array.

    For a single dish, set `diameter_m`. For an interferometer, set
    `configs`: a mapping of configuration name -> maximum baseline [m].
    """

    kind: str  # "dish" or "interferometer"
    diameter_m: float = None
    configs: dict = field(default_factory=dict)


# Approximate, publicly documented values (max baseline for
# interferometers, dish diameter for single-dish telescopes). Good
# enough for an order-of-magnitude beam estimate; refine per-source if
# you need precision (e.g. from an actual observation's synthesized
# beam).
TELESCOPES = {
    "VLA": Telescope(
        kind="interferometer",
        configs={
            "A": 36_400.0,
            "B": 11_100.0,
            "C": 3_400.0,
            "D": 1_030.0,
        },
    ),
    "ALMA": Telescope(
        kind="interferometer",
        configs={
            "C43-1": 161.0,
            "C43-2": 314.0,
            "C43-3": 500.0,
            "C43-4": 784.0,
            "C43-5": 1_400.0,
            "C43-6": 2_500.0,
            "C43-7": 3_600.0,
            "C43-8": 8_500.0,
            "C43-9": 13_900.0,
            "C43-10": 16_200.0,
        },
    ),
    "ATCA": Telescope(
        kind="interferometer",
        configs={
            "6A": 6_000.0,
            "6B": 6_000.0,
            "6C": 6_000.0,
            "6D": 6_000.0,
            "1.5A": 1_500.0,
            "1.5B": 1_500.0,
            "1.5C": 1_500.0,
            "1.5D": 1_500.0,
            "750A": 750.0,
            "750B": 750.0,
            "750C": 750.0,
            "750D": 750.0,
        },
    ),
    "GMRT": Telescope(kind="interferometer", configs={"default": 25_000.0}),
    "MeerKAT": Telescope(kind="interferometer", configs={"default": 8_000.0}),
    "GBT": Telescope(kind="dish", diameter_m=100.0),
    "Effelsberg": Telescope(kind="dish", diameter_m=100.0),
    "Parkes": Telescope(kind="dish", diameter_m=64.0),
    "IRAM-30m": Telescope(kind="dish", diameter_m=30.0),
}


def list_telescopes():
    """Return {telescope_name: description} for every telescope in TELESCOPES."""
    descriptions = {}
    for name, telescope in TELESCOPES.items():
        if telescope.kind == "dish":
            descriptions[name] = f"single dish, {telescope.diameter_m:.0f} m"
        else:
            configs = ", ".join(sorted(telescope.configs))
            descriptions[name] = f"interferometer, configs: {configs}"
    return descriptions


def beam_fwhm_arcsec(telescope_name, freq_hz, config=None):
    """
    Diffraction-limited beam FWHM [arcsec] for a named telescope.

    Parameters
    ----------
    telescope_name : str
        Key into `TELESCOPES` (see `list_telescopes()` for options).
    freq_hz : float
        Observing frequency [Hz].
    config : str, optional
        Array configuration name, for interferometers (e.g. "B" for
        the VLA). Required when the telescope has more than one
        configuration; ignored for single-dish telescopes.

    Returns
    -------
    float
        Beam FWHM in arcsec.

    Raises
    ------
    ValueError
        If the telescope or configuration name is not recognized, or a
        required configuration was not given.
    """
    try:
        telescope = TELESCOPES[telescope_name]
    except KeyError:
        raise ValueError(
            f"Unknown telescope '{telescope_name}'. Available: {sorted(TELESCOPES)}"
        ) from None

    if telescope.kind == "dish":
        D = telescope.diameter_m
    else:
        configs = telescope.configs
        if config is None:
            if len(configs) == 1:
                config = next(iter(configs))
            else:
                raise ValueError(
                    f"Telescope '{telescope_name}' needs a config. " f"Available: {sorted(configs)}"
                )
        try:
            D = configs[config]
        except KeyError:
            raise ValueError(
                f"Unknown config '{config}' for telescope '{telescope_name}'. "
                f"Available: {sorted(configs)}"
            ) from None

    wavelength_m = _C_M_S / freq_hz
    theta_rad = 1.22 * wavelength_m / D
    return np.degrees(theta_rad) * 3600.0
