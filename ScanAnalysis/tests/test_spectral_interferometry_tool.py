"""The one-file spectral-interferometry command: read, analyse, report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scan_analysis.tools.spectral_interferometry import analyze_spectrum_file, main


def _write_spectrum(path, tau_fs=600.0, gdd_fs2=300.0):
    """A Thorlabs-shaped file: two tab-separated columns, no header, CRLF."""
    wavelength = np.linspace(497.7, 1103.2, 3648)
    omega = 2 * np.pi * 299.792458 / wavelength
    centre = 2 * np.pi * 299.792458 / 800.0
    width = 2 * np.pi * 299.792458 * 40.0 / 800.0**2
    intensity = 1e4 * np.exp(-4 * np.log(2) * ((omega - centre) / width) ** 2)
    phase = tau_fs * (omega - centre) + 0.5 * gdd_fs2 * (omega - centre) ** 2
    counts = 2 * intensity * (1 + np.cos(phase)) + 100
    path.write_bytes(
        "".join(f"{w:.3f}\t{c:.2f}\r\n" for w, c in zip(wavelength, counts)).encode()
    )
    return path


def test_reads_and_analyses_a_file(tmp_path):
    result = analyze_spectrum_file(
        _write_spectrum(tmp_path / "Scan001_SPEC_001.txt")
    ).result
    assert result.reliable
    assert result.delta_gdd_fs2 == pytest.approx(300.0, rel=0.01)


def test_the_command_prints_json_and_writes_the_figure(tmp_path, capsys):
    spectrum = _write_spectrum(tmp_path / "s.txt")
    code = main(
        [
            str(spectrum),
            "--reference-nm",
            "800",
            "--png",
            str(tmp_path / "s.png"),
            "--json",
            str(tmp_path / "s.json"),
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["tau0_fs"] == pytest.approx(600.0, abs=0.1)
    assert printed["reference_wavelength_nm"] == pytest.approx(800.0)
    assert json.loads((tmp_path / "s.json").read_text()) == printed
    assert (tmp_path / "s.png").stat().st_size > 10_000


def test_an_unreliable_result_exits_one(tmp_path, capsys):
    assert (
        main([str(_write_spectrum(tmp_path / "s.txt", tau_fs=0.0, gdd_fs2=0.0))]) == 1
    )
    assert json.loads(capsys.readouterr().out)["tau0_fs"] is None


def test_never_creates_an_output_folder(tmp_path):
    spectrum = _write_spectrum(tmp_path / "s.txt")
    with pytest.raises(SystemExit, match="not creating"):
        main([str(spectrum), "--png", str(tmp_path / "missing" / "s.png")])
    assert not (tmp_path / "missing").exists()


def test_reads_a_thorlabs_fts_export(tmp_path):
    """The new spectrometer's .csv: FTS header, ``[Data]`` rows ``x;y``, coarser grid."""
    wavelength = np.linspace(183.9, 1006.8, 2048)
    omega = 2 * np.pi * 299.792458 / wavelength
    centre = 2 * np.pi * 299.792458 / 795.0
    width = 2 * np.pi * 299.792458 * 25.0 / 795.0**2
    intensity = 1e3 * np.exp(-4 * np.log(2) * ((omega - centre) / width) ** 2)
    phase = 400.0 * (omega - centre) + 0.5 * 300.0 * (omega - centre) ** 2
    counts = intensity * (1 + np.cos(phase)) + 100
    lines = ["#Thorlabs FTS", "[SpectrumHeader]", "#XAxisUnit;nm_air", "[Data]"]
    lines += [f"{w:.9e};{c:.9e}" for w, c in zip(wavelength, counts)]
    lines.append("[EndOfFile]")
    path = tmp_path / "Scan013_SPEC-PL1-ThorlabsNew_001.csv"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    result = analyze_spectrum_file(path).result
    assert result.tau0_fs == pytest.approx(400.0, rel=0.02)


def test_a_bad_fts_row_says_which(tmp_path):
    """The reader's own error reaches the user, not a generic refusal."""
    path = tmp_path / "s.csv"
    path.write_bytes(b"#Thorlabs FTS\r\n[Data]\r\n7.9e+02;oops\r\n[EndOfFile]\r\n")
    with pytest.raises(ValueError, match="bad data row"):
        analyze_spectrum_file(path)


def test_a_third_column_is_still_read(tmp_path):
    """Tab-separated files with more columns read x and counts as before."""
    two = _write_spectrum(tmp_path / "two.txt")
    rows = two.read_text().splitlines()
    three = tmp_path / "three.txt"
    three.write_text("".join(f"{r}\t0\n" for r in rows))
    assert analyze_spectrum_file(three).result.tau0_fs == pytest.approx(
        analyze_spectrum_file(two).result.tau0_fs
    )
