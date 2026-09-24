"""Smoke tests for the command-line interface."""

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
