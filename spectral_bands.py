#spectral_bands.py
"""
Available spectral bands for free-free emission calculations.
"""

spec_bands = {
    'low_radio': {
        'frequency': 325e6,
        'description': '325 MHz - low freq radio'
    },
    'radio': {
        'frequency': 3e9,
        'description': '3 GHz - Radio'
    },
    'IR': {
        'frequency': 3e12,
        'description': '3 THz - Mid infrared'
    },
    'optical_R': {
        'frequency': 4.3e14,
        'description': '700 nm - Opt, red'
    },
    'optical_V': {
        'frequency': 5.5e14,
        'description': '545 nm - Opt, green'
    },
    'optical_B': {
        'frequency': 6.9e14,
        'description': '435 nm - Opt, blue'
    },
    'FUV': {
        'frequency': 2e15,
        'description': '150 nm - Far ultraviolet'
    },
    'EUV': {
        'frequency': 3e16,
        'description': '10 nm - Extreme ultraviolet'
    },
    'Xray_soft': {
        'frequency': 3e17,
        'description': '1 keV - Soft x-rays'
    },
    'Xray_hard': {
        'frequency': 3e18,
        'description': '10 keV - Hard x-rays'
    },
}


def get_frequency(band_name):
    try:
        return spec_bands[band_name]['frequency']
    except KeyError:
        available = ', '.join(spec_bands)
        raise ValueError(
            f"Unknown band: {band_name}. Available: {available}"
        )