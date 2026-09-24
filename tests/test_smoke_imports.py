"""Smoke tests: every module in the package must import cleanly.

These don't test physics correctness, but they catch the class of bug
that pure refactoring can introduce: broken imports, missing files,
circular imports, typos in module paths.
"""

import importlib

import pytest

MODULES = [
    "bowshockmaps",
    "bowshockmaps.cli",
    "bowshockmaps.config",
    "bowshockmaps.constants",
    "bowshockmaps.io_utils",
    "bowshockmaps.maps",
    "bowshockmaps.paths",
    "bowshockmaps.spectral_bands",
    "bowshockmaps.physics.bow_shock_surface",
    "bowshockmaps.physics.gaunt_factor",
    "bowshockmaps.physics.ionization",
    "bowshockmaps.physics.normalization",
    "bowshockmaps.physics.radiation",
    "bowshockmaps.physics.thermodynamics",
    "bowshockmaps.visualization.plot_maps",
    "bowshockmaps.visualization.app",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
