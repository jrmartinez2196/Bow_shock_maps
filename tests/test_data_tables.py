"""Tests that the bundled data tables load and behave sensibly."""

import numpy as np

from bowshockmaps.paths import GAUNT_FACTOR_FILE, IONIZATION_TABLE_FILE
from bowshockmaps.physics.gaunt_factor import GauntFactor
from bowshockmaps.physics.ionization import IonizationTable


def test_data_files_exist():
    assert GAUNT_FACTOR_FILE.exists()
    assert IONIZATION_TABLE_FILE.exists()


def test_gaunt_factor_table_loads_and_interpolates():
    gaunt = GauntFactor()
    result = gaunt.gaunt_ff_calc(nu_ff=1e14, T=1e4, Z=1.0)
    assert np.isfinite(result).all()
    assert (result > 0).all()


def test_gaunt_factor_handles_invalid_temperature_gracefully():
    gaunt = GauntFactor()
    # T=0 / negative T should not raise, per the "valid" mask in gaunt_ff_calc.
    result = gaunt.gaunt_ff_calc(nu_ff=1e14, T=np.array([0.0, -1.0, 1e4]), Z=1.0)
    assert np.isfinite(result).all()


def test_ionization_table_loads():
    table = IonizationTable(IONIZATION_TABLE_FILE)
    assert table is not None


def test_ionization_fractions_are_between_zero_and_one():
    table = IonizationTable(IONIZATION_TABLE_FILE)
    ion_H, ion_O = table.fractions(T=1e5)
    assert 0.0 <= ion_H <= 1.0
    assert 0.0 <= ion_O <= 1.0
