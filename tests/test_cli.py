"""Smoke tests for the command-line interface."""

import pytest

from bowshockmaps.cli import build_parser


def test_default_args():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.source == "RXJ0528+2838"
    assert args.convolve is True
    assert args.list_bands is False


def test_list_bands_flag():
    parser = build_parser()
    args = parser.parse_args(["--list-bands"])
    assert args.list_bands is True


def test_convolve_flag_parses_false():
    parser = build_parser()
    args = parser.parse_args(["--convolve", "false"])
    assert args.convolve is False


def test_telescope_and_config_flags():
    parser = build_parser()
    args = parser.parse_args(["--telescope", "VLA", "--telescope-config", "B"])
    assert args.telescope == "VLA"
    assert args.telescope_config == "B"


def test_unknown_telescope_rejected_by_argparse():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--telescope", "NotATelescope"])


def test_beam_fwhm_flag():
    parser = build_parser()
    args = parser.parse_args(["--beam-fwhm", "2.5"])
    assert args.beam_fwhm == 2.5


def test_list_telescopes_flag():
    parser = build_parser()
    args = parser.parse_args(["--list-telescopes"])
    assert args.list_telescopes is True


def test_list_telescopes_prints_known_telescopes(capsys):
    import sys

    from bowshockmaps.cli import main

    old_argv = sys.argv
    try:
        sys.argv = ["bowshockmaps", "--list-telescopes"]
        main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "VLA" in captured.out
    assert "GBT" in captured.out


def test_list_bands_prints_all_bands(capsys):
    import sys

    from bowshockmaps.cli import main

    old_argv = sys.argv
    try:
        sys.argv = ["bowshockmaps", "--list-bands"]
        main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "low_radio" in captured.out
    assert "Xray_hard" in captured.out


def test_telescope_without_a_compatible_band_is_a_clean_usage_error(capsys):
    # The default band is FUV; a VLA cannot observe there. This must exit
    # with a usage error that says why, not crash later in the computation.
    import sys

    from bowshockmaps.cli import main

    old_argv = sys.argv
    try:
        sys.argv = [
            "bowshockmaps",
            "-s",
            "RXJ0528+2838",
            "--telescope",
            "VLA",
            "--telescope-config",
            "A",
        ]
        with pytest.raises(SystemExit) as excinfo:
            main()
    finally:
        sys.argv = old_argv

    assert excinfo.value.code == 2
    assert "does not observe" in capsys.readouterr().err


def test_max_pixels_flag():
    parser = build_parser()
    assert parser.parse_args(["--max-pixels", "300"]).max_pixels == 300
    assert parser.parse_args([]).max_pixels is None


def test_fov_and_accuracy_flags():
    parser = build_parser()
    args = parser.parse_args(["--fov", "4.5", "--accuracy", "fine"])
    assert args.fov == 4.5
    assert args.accuracy == "fine"
    defaults = parser.parse_args([])
    assert defaults.fov is None and defaults.accuracy is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--accuracy", "ultra"])
