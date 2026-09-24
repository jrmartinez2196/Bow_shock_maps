# Bow Shock Maps

Modeling and visualization of forbidden-line (and free-free/synchrotron
continuum) emission from wind-driven stellar bow shocks. Computes 2D
projected emission maps via line-of-sight integration through the
shocked wind/ISM, with optional Gaussian-beam convolution to simulate
an instrument's angular resolution.

## Installation

```bash
git clone https://github.com/jrmartinez2196/Bow_shock_maps.git
cd Bow_shock_maps
git checkout develop
pip install -e .          # runtime dependencies only
# or, for development (tests, formatting, linting):
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Usage

```bash
# Interactive visualizer for the default source (RXJ0528+2838)
bowshockmaps

# Choose a different source
bowshockmaps --source Vela_X-1

# List available spectral bands for free-free emission
bowshockmaps --list-bands

# Compute free-free emission in a specific band
bowshockmaps --source BD+43 --band FUV

# Disable Gaussian-beam convolution
bowshockmaps --convolve false

# Verbose (debug-level) logging
bowshockmaps --verbose
```

Source parameter files live in `data/systems/` (one `.txt` file per
source, Python-dict syntax). To add a new source, drop a new file
there named `<SourceName>.txt`, containing at minimum: `Mdot`, `Vw`,
`Vstar`, `n_ism`, `dist`.

## Project layout

```
src/bowshockmaps/
├── cli.py                     # command-line entry point
├── config.py                  # default grid/resolution settings (GridConfig)
├── constants.py                # physical constants, cgs units
├── io_utils.py                 # loading/validating source parameter files
├── maps.py                     # projection maps, radial profiles, LOS integration
├── paths.py                    # filesystem locations (data dir, etc.)
├── spectral_bands.py           # named frequency bands for free-free emission
├── physics/
│   ├── bow_shock_surface.py    # bow-shock geometry (Wilkin 1997, Christie+ 2016)
│   ├── gaunt_factor.py         # tabulated free-free Gaunt factor
│   ├── ionization.py           # ionization-fraction interpolation
│   ├── normalization.py        # non-thermal particle normalization
│   ├── radiation.py            # emissivities (Halpha, [OIII], free-free, sync)
│   └── thermodynamics.py       # Rankine-Hugoniot jump conditions, cooling, advection
└── visualization/
    ├── app.py                  # BowShock: interactive Matplotlib application
    └── plot_maps.py            # plotting helper functions

data/
├── gauntff.dat                 # free-free Gaunt factor table
├── ionization_table.dat        # CIE ionization fractions (Gnat & Sternberg 2007)
└── systems/                    # per-source parameter files

tests/                          # pytest suite
```

## Testing

```bash
pytest
```

The suite includes import/smoke tests, unit tests for individual
physics functions, and one integration test that instantiates the
full `BowShock` pipeline end to end. It does **not** yet include
regression tests against reference/golden emission maps — see
"Known issues" below.

## Development

```bash
black src/ tests/
isort src/ tests/ --profile black
ruff check src/ tests/
```

## Known issues / things flagged for review

- No regression/reference dataset exists yet to verify that numerical
  outputs are unchanged across refactors. Consider saving a reference
  emission map (e.g. for `RXJ0528+2838` at default resolution) to
  `tests/data/` and adding a comparison test.
- Proton non-thermal normalization (`k0p_RS`/`k0p_FS` in `maps.py`) is
  computed but not yet consumed by any emission channel — kept
  intentionally (see comment at its definition) for a future
  hadronic-emission feature.
- The default resolution (`nx=ny=50`, `nz=5000`, or higher once the
  beam-size check kicks in) is compute-heavy; the test suite uses a
  much coarser grid to stay fast, and does not exercise the exact
  resolution used in production runs.
