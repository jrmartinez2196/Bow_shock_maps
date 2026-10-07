"""Default grid/resolution configuration for the bow-shock model.

Previously these were bare module-level names (``nx``, ``ny``, ...).
They are grouped here into a small dataclass so a caller can override
one configuration without mutating shared module state.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GridConfig:
    """Grid resolution and line-of-sight integration settings."""

    nx: int = 100
    ny: int = 100
    nz: int = 5000
    zmax: float = 5.0  # In R0 units
    max_theta: float = np.deg2rad(135.0)
    # Ceiling on the number of pixels per axis when the grid is refined to
    # sample a fine instrumental beam (see maps.make_projection_maps). The
    # cost of a map grows like nx * ny * nz (2000 x 2000 x 1000 takes ~35 s on one core).
    max_pixels: int = 2000
    # Automatic grid. With ``auto_los`` the line-of-sight range (zmax) and
    # number of steps (nz) are derived from the shell and ``accuracy``
    # instead of the ``zmax`` and ``nz`` above, which are then only used if
    # ``auto_los`` is False. ``accuracy`` sets the line-of-sight step as the
    # thickness of the thinnest layer that matters divided by 3 ("fast"),
    # 6 ("normal") or 12 ("fine").
    auto_los: bool = True
    accuracy: str = "normal"
    # Half-width of the (square) field of view in R0 units. None: derived
    # from the shell for each source and inclination.
    fov: Optional[float] = None


DEFAULT_GRID_CONFIG = GridConfig()

# Backwards-compatible module-level aliases (kept so existing call sites
# that do ``from bowshockmaps.config import nx, ny, ...`` keep working).
nx = DEFAULT_GRID_CONFIG.nx
ny = DEFAULT_GRID_CONFIG.ny
nz = DEFAULT_GRID_CONFIG.nz
zmax = DEFAULT_GRID_CONFIG.zmax
max_theta = DEFAULT_GRID_CONFIG.max_theta
max_pixels = DEFAULT_GRID_CONFIG.max_pixels
auto_los = DEFAULT_GRID_CONFIG.auto_los
accuracy = DEFAULT_GRID_CONFIG.accuracy
fov = DEFAULT_GRID_CONFIG.fov
