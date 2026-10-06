"""Telescope beam-size lookup, used to drive the convolution step.

Provides a simple diffraction-limited estimate of the beam FWHM for a
named telescope: for a single dish, via the Airy-disk first-null angle
(1.22 * lambda / D); for an interferometer, via the standard
lambda / D_max rule of thumb for the synthesized-beam resolution at its
array configuration's maximum baseline.

These are two genuinely different formulas, not the same one applied
to two kinds of "D": the 1.22 factor is specific to the diffraction
pattern of a filled circular aperture (the Airy disk) and does not
apply to a sparse interferometer array, which has no such pattern.

Both are order-of-magnitude estimates, not full synthesized-beam
calculations: for an interferometer, the actual beam shape/size also
depends on the uv-coverage, the source's declination, and the imaging
weights used (uniform weighting is close to lambda/D_max; natural
weighting is typically ~1.2-1.5x wider). Use `beam_fwhm_arcsec` as a
reasonable starting point; if you have a beam size from an actual
observation or proposal tool, override it directly instead (see the
--beam-fwhm CLI flag / the `beam_fwhm` argument to `BowShock`).
"""

from dataclasses import dataclass, field

import numpy as np

from bowshockmaps.constants import c
from bowshockmaps.spectral_bands import spec_bands

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
    # Approximate range [Hz] covered by the telescope's receivers. Only
    # used to catch nonsensical requests (e.g. a radio telescope at UV
    # frequencies, where the diffraction formula gives a beam of 1e-6
    # arcsec), so order-of-magnitude values are enough.
    freq_range_hz: tuple = None


# Approximate, publicly documented values (max baseline for
# interferometers, dish diameter for single-dish telescopes). Good
# enough for an order-of-magnitude beam estimate; refine per-source if
# you need precision (e.g. from an actual observation's synthesized
# beam).
TELESCOPES = {
    "VLA": Telescope(
        kind="interferometer",
        freq_range_hz=(5.4e7, 5.0e10),
        configs={
            "A": 36_400.0,
            "B": 11_100.0,
            "C": 3_400.0,
            "D": 1_030.0,
        },
    ),
    "ALMA": Telescope(
        kind="interferometer",
        freq_range_hz=(3.5e10, 9.5e11),
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
        freq_range_hz=(1.1e9, 1.05e11),
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
    "GMRT": Telescope(
        kind="interferometer", configs={"default": 25_000.0}, freq_range_hz=(5.0e7, 1.5e9)
    ),
    "MeerKAT": Telescope(
        kind="interferometer", configs={"default": 8_000.0}, freq_range_hz=(5.8e8, 1.55e10)
    ),
    "GBT": Telescope(kind="dish", diameter_m=100.0, freq_range_hz=(2.9e8, 1.16e11)),
    "Effelsberg": Telescope(kind="dish", diameter_m=100.0, freq_range_hz=(3.0e8, 9.5e10)),
    "Parkes": Telescope(kind="dish", diameter_m=64.0, freq_range_hz=(7.0e8, 2.6e10)),
    "IRAM-30m": Telescope(kind="dish", diameter_m=30.0, freq_range_hz=(7.3e10, 3.73e11)),
}


def list_telescopes():
    """Return {telescope_name: description} for every telescope in TELESCOPES."""
    descriptions = {}
    for name, telescope in TELESCOPES.items():
        if telescope.kind == "dish":
            text = f"single dish, {telescope.diameter_m:.0f} m"
        else:
            configs = ", ".join(sorted(telescope.configs))
            text = f"interferometer, configs: {configs}"
        if telescope.freq_range_hz is not None:
            lo, hi = telescope.freq_range_hz
            text += f"; ~{lo/1e9:.3g}-{hi/1e9:.3g} GHz"
        descriptions[name] = text
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
        If the telescope or configuration name is not recognized, a
        required configuration was not given, or `freq_hz` is outside the
        range the telescope observes in.
    """
    try:
        telescope = TELESCOPES[telescope_name]
    except KeyError:
        raise ValueError(
            f"Unknown telescope '{telescope_name}'. Available: {sorted(TELESCOPES)}"
        ) from None

    if telescope.kind == "dish":
        D = telescope.diameter_m
        airy_factor = 1.22  # first null of the Airy pattern (filled circular aperture)
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
        # No Airy factor here: a sparse interferometer array doesn't
        # produce the same circular-aperture diffraction pattern a
        # single dish does. lambda/D_max is the standard rule-of-thumb
        # resolution estimate for a synthesized beam.
        airy_factor = 1.0

    wavelength_m = _C_M_S / freq_hz

    if telescope.freq_range_hz is not None:
        lo, hi = telescope.freq_range_hz
        if not (lo <= freq_hz <= hi):
            in_range = [b for b, info in spec_bands.items() if lo <= info["frequency"] <= hi]
            hint = (
                f"Predefined bands in its range: {', '.join(in_range)} (use --band)."
                if in_range
                else "None of the predefined bands is in its range."
            )
            raise ValueError(
                f"{telescope_name} does not observe at {freq_hz:.3g} Hz "
                f"(wavelength {wavelength_m:.3g} m): its receivers cover roughly "
                f"{lo:.3g}-{hi:.3g} Hz, and the diffraction-limited beam is meaningless "
                f"outside that. {hint} To give the beam directly, use --beam-fwhm."
            )

    theta_rad = airy_factor * wavelength_m / D
    return np.degrees(theta_rad) * 3600.0
