"""Default grid/resolution configuration for the bow-shock model.

Previously these were bare module-level names (``nx``, ``ny``, ...).
They are grouped here into a small dataclass so a caller can override
one configuration without mutating shared module state.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridConfig:
    """Grid resolution and line-of-sight integration settings."""

    nx: int = 50
    ny: int = 50
    nz: int = 5000
    zmax: float = 10.0  # In R0 units
    max_theta: float = np.deg2rad(120.0)


DEFAULT_GRID_CONFIG = GridConfig()

# Backwards-compatible module-level aliases (kept so existing call sites
# that do ``from bowshockmaps.config import nx, ny, ...`` keep working).
nx = DEFAULT_GRID_CONFIG.nx
ny = DEFAULT_GRID_CONFIG.ny
nz = DEFAULT_GRID_CONFIG.nz
zmax = DEFAULT_GRID_CONFIG.zmax
max_theta = DEFAULT_GRID_CONFIG.max_theta
