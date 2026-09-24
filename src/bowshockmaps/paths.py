"""Central definitions of filesystem locations used across the package.

Data files (Gaunt-factor tables, ionization tables, source parameter
files) live outside the installed package, under ``<repo_root>/data``.
Resolving them relative to this module (rather than to the current
working directory) means the code works regardless of where it is
invoked from.
"""

from pathlib import Path

# <repo_root>/src/bowshockmaps/paths.py -> <repo_root>
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
SYSTEMS_DIR = DATA_DIR / "systems"

GAUNT_FACTOR_FILE = DATA_DIR / "gauntff.dat"
IONIZATION_TABLE_FILE = DATA_DIR / "ionization_table.dat"
