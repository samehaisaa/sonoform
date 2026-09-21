"""Tests for the command line interface."""

import numpy as np
import pytest

from sonoform.cli import _parse_target, ascii_shape, main
from sonoform.geometry import FourierShape


def test_chords_lists_every_chord(capsys):
    assert main(["chords"]) == 0
    out = capsys.readouterr().out
    for name in ["major", "minor7", "octaves", "fifths"]:
        assert name in out
    assert "impossible" in out
    assert "buildable" in out


def test_impossible_chord_exits_nonzero_and_explains(capsys):
    code = main(["tune", "octaves"])
    out = capsys.readouterr().out
    assert code == 2
    assert "cannot be built" in out
    assert "Ashbaugh-Benguria" in out
    assert "nearest achievable" in out


def test_third_partial_refusal_is_reported(capsys):
    assert main(["tune", "fifths"]) == 2
    assert "third partial" in capsys.readouterr().out


def test_ratios_can_be_given_directly(capsys):
    """An impossible explicit target should be refused the same way."""
    assert main(["tune", "1", "2", "3"]) == 2
    assert "cannot be built" in capsys.readouterr().out


def test_parse_target_distinguishes_names_from_numbers():
    assert _parse_target(["major"]) == "major"
    assert _parse_target(["1", "1.25", "1.5"]) == [1.0, 1.25, 1.5]


def test_parse_target_rejects_nonsense():
    with pytest.raises(SystemExit, match="could not read"):
        _parse_target(["1.0", "banana"])


def test_unknown_chord_is_reported(capsys):
    assert main(["tune", "wombat"]) == 2
    assert "unknown chord" in capsys.readouterr().err


def test_ascii_shape_is_rectangular_and_nonempty():
    rows = ascii_shape(FourierShape(1.0), width=40)
    assert len({len(r) for r in rows}) == 1
    assert len(rows) >= 8
    assert any(ch != " " for row in rows for ch in row)


def test_ascii_shape_has_no_blank_border_rows():
    rows = ascii_shape(FourierShape(0.9, [0.3, 0.1], [0.05, 0.2]), width=40)
    assert rows[0].strip()
    assert rows[-1].strip()


def test_ascii_shape_of_a_disc_is_symmetric():
    rows = ascii_shape(FourierShape(1.0), width=40)
    filled = np.array([[ch != " " for ch in row] for row in rows])
    # left-right mirror symmetry, allowing a column of slack for rounding
    assert np.abs(filled.sum(axis=0) - filled.sum(axis=0)[::-1]).max() <= 1


def test_ascii_shape_falls_back_when_the_console_cannot_encode(monkeypatch):
    """Windows consoles default to cp1252 and cannot print half blocks."""

    class Cp1252Stdout:
        encoding = "cp1252"

    monkeypatch.setattr("sonoform.cli.sys.stdout", Cp1252Stdout())
    rows = ascii_shape(FourierShape(1.0), width=30)
    body = "".join(rows)
    assert "█" not in body
    body.encode("cp1252")  # must not raise


@pytest.mark.slow
def test_tune_solves_and_writes_files(tmp_path, capsys):
    wav = tmp_path / "out.wav"
    png = tmp_path / "out.png"
    code = main(
        [
            "tune",
            "major",
            "--seed",
            "1",
            "--restarts",
            "2",
            "--wav",
            str(wav),
            "--png",
            str(png),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "solved in" in out
    assert wav.exists() and wav.stat().st_size > 1000
    assert png.exists() and png.stat().st_size > 1000
