#main.py

import argparse

from bow_shock import BowShock
from spectral_bands import spec_bands

def main():
    
    parser = argparse.ArgumentParser(
        description='Bow Shock Interactive Visualizer',
        epilog="""
E.g.:
  python3 bow_shock.py
  python3 bow_shock.py --source RXJ0528+2838
  python3 bow_shock.py --source RXJ0528+2838 --params-dir ./params_file
  python3 bow_shock.py --list-bands

Available frequencies:
  low_radio, radio, IR, optical_R, optical_V, optical_B, FUV, EUV, Xray_soft, Xray_hard
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='RXJ0528+2838',
        help='Source name (default: RXJ0528+2838)'
    )
    
    parser.add_argument(
        '--params-dir', '-p',
        type=str,
        default='Systems',
        help='Path to parameters file (default: Systems)'
    )
    
    parser.add_argument(
        '--list-bands', '-l',
        action='store_true',
        help='List the available spectrum bands to compute monoenergetic free-free emission'
    )
    
    parser.add_argument(
        '--band', '-b',
        type=str,
        default=None,
        help='Spectrum band to compute free-free emission (e.g.: FUV, IR, Xray_soft)'
    )

    parser.add_argument(
        '--convolve',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Determines if the emission map is convolved with a Gaussian beam instrument or not'
    )
    
    args = parser.parse_args()
    
    if args.list_bands:
        print("Available spectrum bands to calculate free-free:")

        for band, info in spec_bands.items():
            print(f"  {band:12s} : {info['description']}")

        return
    
    print(f"Loading. Source: {args.source}, params_dir: {args.params_dir}")
    app = BowShock(args.source, args.params_dir, convolve=args.convolve)
    
    if args.band:
        app.set_continuum_band(args.band)
    
    app.run()


if __name__ == "__main__":
    main()