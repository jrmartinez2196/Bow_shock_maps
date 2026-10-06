"""Tests for the telescope beam-size lookup used by convolution."""

import pytest

from bowshockmaps.instruments import TELESCOPES, beam_fwhm_arcsec, list_telescopes


def test_list_telescopes_covers_every_entry():
    descriptions = list_telescopes()
    assert set(descriptions) == set(TELESCOPES)


@pytest.mark.parametrize(
    "telescope_name,config",
    [("VLA", "A"), ("VLA", "B"), ("VLA", "C"), ("VLA", "D")],
)
def test_vla_beam_shrinks_with_more_extended_configs(telescope_name, config):
    # A (most extended) should give a sharper (smaller) beam than D
    # (most compact) at the same frequency.
    freq_hz = 5e9
    fwhm_A = beam_fwhm_arcsec(telescope_name, freq_hz, "A")
    fwhm_D = beam_fwhm_arcsec(telescope_name, freq_hz, "D")
    assert fwhm_A < fwhm_D


def test_beam_fwhm_shrinks_with_increasing_frequency():
    fwhm_low = beam_fwhm_arcsec("GBT", 1e9)
    fwhm_high = beam_fwhm_arcsec("GBT", 10e9)
    assert fwhm_high < fwhm_low


def test_single_dish_config_is_optional():
    # Single-dish telescopes should not require a `config`.
    assert beam_fwhm_arcsec("GBT", 5e9) == beam_fwhm_arcsec("GBT", 5e9, config=None)


def test_interferometer_with_single_config_does_not_require_it_explicitly():
    # GMRT/MeerKAT only have one ("default") configuration.
    assert beam_fwhm_arcsec("GMRT", 1.4e9) > 0


def test_multi_config_interferometer_requires_config():
    with pytest.raises(ValueError):
        beam_fwhm_arcsec("VLA", 5e9)


def test_unknown_telescope_raises():
    with pytest.raises(ValueError):
        beam_fwhm_arcsec("NotATelescope", 5e9)


def test_unknown_config_raises():
    with pytest.raises(ValueError):
        beam_fwhm_arcsec("VLA", 5e9, config="Z")


def test_gbt_beam_matches_known_c_band_value():
    # GBT at ~5 GHz (C band) has a well-known beam of ~2.5 arcmin.
    fwhm = beam_fwhm_arcsec("GBT", 5e9)
    assert 100 < fwhm < 200  # arcsec


def test_every_telescope_declares_a_frequency_range():
    for name, telescope in TELESCOPES.items():
        assert telescope.freq_range_hz is not None, name
        lo, hi = telescope.freq_range_hz
        assert 0 < lo < hi


def test_radio_telescope_at_uv_frequencies_is_rejected_with_a_useful_message():
    # FUV (2e15 Hz): a VLA gives a ~1e-6 arcsec "beam", which is meaningless.
    with pytest.raises(ValueError) as excinfo:
        beam_fwhm_arcsec("VLA", 2e15, "A")
    message = str(excinfo.value)
    assert "does not observe" in message
    assert "radio" in message  # names a band that would work
    assert "--beam-fwhm" in message  # and the escape hatch


@pytest.mark.parametrize("telescope", sorted(TELESCOPES))
def test_beam_is_available_inside_each_telescopes_own_range(telescope):
    lo, hi = TELESCOPES[telescope].freq_range_hz
    config = None
    if TELESCOPES[telescope].kind == "interferometer":
        config = sorted(TELESCOPES[telescope].configs)[0]
    nu = (lo * hi) ** 0.5  # geometric middle of the range
    assert beam_fwhm_arcsec(telescope, nu, config) > 0


def test_list_telescopes_shows_the_frequency_range():
    assert "GHz" in list_telescopes()["VLA"]
