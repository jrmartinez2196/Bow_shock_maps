"""Command-line entry point for the bow-shock interactive visualizer."""

import argparse
import logging

from bowshockmaps.instruments import TELESCOPES, list_telescopes
from bowshockmaps.paths import SYSTEMS_DIR
from bowshockmaps.spectral_bands import spec_bands
from bowshockmaps.visualization.app import BowShock

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Bow Shock Interactive Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bowshockmaps
  bowshockmaps --source RXJ0528+2838
  bowshockmaps --source RXJ0528+2838 --params-dir ./params_file
  bowshockmaps --list-bands
  bowshockmaps --list-telescopes
  bowshockmaps --band radio --telescope VLA --telescope-config B
  bowshockmaps --band radio --beam-fwhm 2.5

Available frequencies:
  low_radio, radio, IR, optical_R, optical_V, optical_B, FUV, EUV, Xray_soft, Xray_hard
        """,
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="RXJ0528+2838",
        help="Source name (default: RXJ0528+2838)",
    )
    parser.add_argument(
        "--params-dir",
        "-p",
        type=str,
        default=str(SYSTEMS_DIR),
        help=f"Path to parameters directory (default: {SYSTEMS_DIR})",
    )
    parser.add_argument(
        "--list-bands",
        "-l",
        action="store_true",
        help="List the available spectrum bands to compute monoenergetic free-free emission",
    )
    parser.add_argument(
        "--band",
        "-b",
        type=str,
        default=None,
        help="Spectrum band to compute free-free emission (e.g.: FUV, IR, Xray_soft)",
    )
    parser.add_argument(
        "--convolve",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether the emission map is convolved with a Gaussian beam instrument",
    )
    parser.add_argument(
        "--list-telescopes",
        action="store_true",
        help="List the telescopes/array configurations available for --telescope",
    )
    parser.add_argument(
        "--telescope",
        type=str,
        default=None,
        choices=sorted(TELESCOPES) or None,
        help=(
            "Telescope used to set the convolution beam FWHM, diffraction-limited "
            "at the current --band frequency (see --list-telescopes). Ignored if "
            "--beam-fwhm is also given."
        ),
    )
    parser.add_argument(
        "--telescope-config",
        type=str,
        default=None,
        help="Array configuration for --telescope (e.g. 'B' for the VLA); see --list-telescopes",
    )
    parser.add_argument(
        "--beam-fwhm",
        type=float,
        default=None,
        help="Beam FWHM [arcsec] to convolve with directly, overriding --telescope",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and launch the interactive visualizer."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.list_bands:
        print("Available spectrum bands to calculate free-free:")
        for band, info in spec_bands.items():
            print(f"  {band:12s} : {info['description']}")
        return

    if args.list_telescopes:
        print("Available telescopes for --telescope:")
        for name, description in list_telescopes().items():
            print(f"  {name:12s} : {description}")
        return

    logger.info("Loading. Source: %s, params_dir: %s", args.source, args.params_dir)
    app = BowShock(
        args.source,
        args.params_dir,
        convolve=args.convolve,
        telescope=args.telescope,
        telescope_config=args.telescope_config,
        beam_fwhm=args.beam_fwhm,
    )

    if args.band:
        app.set_continuum_band(args.band)

    app.run()


if __name__ == "__main__":
    main()
