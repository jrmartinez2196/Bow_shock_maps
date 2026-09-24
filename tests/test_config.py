"""Tests for the default grid/resolution configuration."""

import numpy as np

from bowshockmaps.config import (
    DEFAULT_GRID_CONFIG,
    GridConfig,
    max_theta,
    nx,
    ny,
    nz,
    zmax,
)


def test_default_grid_config_matches_module_level_aliases():
    assert DEFAULT_GRID_CONFIG.nx == nx
    assert DEFAULT_GRID_CONFIG.ny == ny
    assert DEFAULT_GRID_CONFIG.nz == nz
    assert DEFAULT_GRID_CONFIG.zmax == zmax
    assert DEFAULT_GRID_CONFIG.max_theta == max_theta


def test_grid_config_values_are_positive():
    for value in (nx, ny, nz, zmax):
        assert value > 0


def test_max_theta_is_within_valid_angle_range():
    assert 0 < max_theta <= np.pi


def test_grid_config_is_overridable_without_mutating_default():
    custom = GridConfig(nx=10, ny=10)
    assert custom.nx == 10
    assert DEFAULT_GRID_CONFIG.nx == 50  # unaffected by the override
