# Bow Shock Maps

Modeling and visualization of emission from wind-driven stellar bow shocks. Computes 2D
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

Requires Python >= 3.8.

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

# Set the convolution beam from a real telescope + array configuration.
# Give --band too: the default band is FUV, which no radio telescope observes
# (the program says so and exits rather than compute a meaningless beam).
bowshockmaps --band radio --telescope VLA --telescope-config B

# List available telescopes/configurations
bowshockmaps --list-telescopes

# Give the beam FWHM directly instead (arcsec), e.g. from a real observation
bowshockmaps --band radio --beam-fwhm 2.5

# Verbose (debug-level) logging
bowshockmaps --verbose
```

**Resolution and cost.** To convolve with a beam the map grid must sample it
(pixel <= 0.5 FWHM), so a fine beam on a large source asks for many pixels: BD+43
(~1500" across) with VLA-A at 3 GHz (beam 0.57") would need ~5400 pixels per side.
The refinement is therefore capped at `--max-pixels` (default 2000,
`config.max_pixels`). When the cap binds, the beam is smaller than a pixel, so it is
treated as unresolved: a Gaussian that narrow does not change the map, which is only
converted to per-beam units, and a warning says so.

**Grid: field of view, line of sight, accuracy.** By default nothing about the grid
needs tuning per system: the program derives it from the shell of each source
(`config.auto_los = True`).

* *Field of view*: the extent of the shell projected on the sky for the current
  inclination, plus a 3% margin, with square pixels (`max(nx, ny)` pixels along the
  longer side). A fixed `+-(6 + 2 sin^2 i) R0` is not enough in general: at
  `max_theta = 135` the shell needs up to ~7 R0, at 160 deg up to ~26 R0. Give
  `--fov H` (half-width in R0 units) or `config.fov` to force a square field.
* *Line-of-sight range* `zmax`: the part of each line of sight where the shell can be.
* *Line-of-sight step* `dz`: the integral through a layer has an error that goes as
  `dz / thickness`, so `dz` is the thickness of the thinnest layer that matters (one
  that carries at least 5% of the emission measure, `n^2 * volume`) divided by 3, 6 or
  12 for `--accuracy fast|normal|fine` (default `normal`, `config.accuracy`), kept
  within [0.0015, 0.05] R0. Measured on three systems, `normal` gives a median error of
  ~1% in the bright pixels (~0.4%-0.9% for the continuum and for sources with thicker
  layers; a few % for H-alpha dominated by a very thin layer) and `fine` about half.
  `nz` follows from `dz` and `zmax`.

To use fixed values instead, set `config.auto_los = False` (then `zmax` and `nz` from
`config.py` are used) or assign `app.zmax` / `app.nz` by hand; both always win over the
derived ones.

Run time grows as nx*ny*nz, plus the building of a 2D table of emissivities whose cost
depends on `dz` (about 30 s when the thinnest layer is ~0.014 R0). Integration alone
takes ~7 s for 1000 x 1000 pixels with 1000 steps and ~35 s for 2000 x 2000.

Without `--telescope`/`--beam-fwhm`, the convolution beam falls back to
a placeholder (the source's projected stagnation radius) -- pass one of the two
for a physically meaningful beam. See `src/bowshockmaps/instruments.py`
for the telescope database and the (diffraction-limited, order-of-
magnitude) formula used; add entries there for telescopes not yet
listed.

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
│   ├── bow_shock_surface.py    # bow-shock geometry (Wilkin 1996, Christie+ 2016)
│   ├── gaunt_factor.py         # tabulated free-free Gaunt factor
│   ├── ionization.py           # ionization-fraction interpolation
│   ├── normalization.py        # non-thermal particle normalization
│   ├── radiation.py            # emissivities (Halpha, [OIII], free-free, sync)
│   ├── shell_geometry.py       # layered shell in normal coordinates (foot point + distance along the normal)
│   └── thermodynamics.py       # Adiabtatic vs radiative shocks, Rankine-Hugoniot jump conditions, cooling, advection
└── visualization/
    ├── app.py                  # BowShock: interactive Matplotlib application
    └── plot_maps.py            # plotting helper functions

data/
├── gauntff.dat                 # free-free Gaunt factor table from van Hoof+ 2014
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
