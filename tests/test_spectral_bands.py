"""Tests for the spectral band lookup table."""

import pytest

from bowshockmaps.spectral_bands import get_frequency, spec_bands


@pytest.mark.parametrize("band_name", list(spec_bands.keys()))
def test_get_frequency_returns_positive_value_for_known_bands(band_name):
    assert get_frequency(band_name) > 0


def test_get_frequency_raises_for_unknown_band():
    with pytest.raises(ValueError):
        get_frequency("not-a-real-band")


def test_bands_are_frequency_ordered_low_to_high_energy():
    # Sanity check: radio bands should have lower frequency than X-ray bands.
    assert get_frequency("low_radio") < get_frequency("radio")
    assert get_frequency("radio") < get_frequency("IR")
    assert get_frequency("Xray_soft") < get_frequency("Xray_hard")
