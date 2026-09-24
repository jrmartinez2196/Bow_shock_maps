"""Tests for loading and validating per-source parameter files."""

import pytest

from bowshockmaps.io_utils import get_source_params, validate_params
from bowshockmaps.paths import SYSTEMS_DIR

KNOWN_SOURCES = ["RXJ0528+2838", "BD+43", "Vela_X-1"]


@pytest.mark.parametrize("source_name", KNOWN_SOURCES)
def test_get_source_params_loads_known_systems(source_name):
    params = get_source_params(source_name)
    assert isinstance(params, dict)
    assert "Mdot" in params


def test_systems_dir_contains_known_sources():
    files = {p.stem for p in SYSTEMS_DIR.iterdir()}
    for source_name in KNOWN_SOURCES:
        assert source_name in files


def test_get_source_params_raises_for_unknown_source():
    with pytest.raises(FileNotFoundError):
        get_source_params("this-source-does-not-exist")


def test_validate_params_rejects_negative_mdot():
    params = {"Mdot": -1.0, "Vw": 1.0, "Vstar": 1.0, "n_ism": 1.0, "dist": 1.0}
    with pytest.raises(ValueError):
        validate_params(params)


def test_validate_params_requires_mandatory_keys():
    with pytest.raises(KeyError):
        validate_params({"Mdot": 1.0})


def test_validate_params_fills_in_defaults_for_optional_keys():
    params = {"Mdot": 1.0, "Vw": 1.0, "Vstar": 1.0, "n_ism": 1.0, "dist": 1.0}
    validated = validate_params(dict(params))
    assert "inclination" in validated
    assert "PA" in validated
    assert "R_str" in validated
    assert 0 <= validated["inclination"] <= 180
    assert 0 <= validated["PA"] < 360
