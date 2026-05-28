"""Tests for FROG spectral phase fitting analyzer."""

from pathlib import Path

import numpy as np
import pytest

from image_analysis.analyzers.frog_spectral_phase_analyzer import (
    FrogSpectralPhaseAnalyzer,
    wavelength_nm_to_omega_rad_per_fs,
)
from image_analysis.config.array1d_processing import (
    Data1DConfig,
    Line1DConfig,
    ROI1DConfig,
)


def _make_config() -> Line1DConfig:
    return Line1DConfig(
        name="frog_phase",
        description="synthetic FROG spectral phase",
        data_loading=Data1DConfig(
            data_type="tsv",
            delimiter="\t",
            x_column=3,
            y_column=5,
            auxiliary_columns={"weights": 4},
        ),
        roi=ROI1DConfig(x_min=780.0, x_max=820.0),
        x_units="nm",
        y_units="rad",
        analysis={
            "fit_order": 3,
            "mask_threshold": 0.05,
            "reference_wavelength_nm": 800.0,
            "sign_reference_order": 2,
            "sign_reference": 1.0,
        },
    )


def _write_frog_lineout(path: Path) -> dict[str, float]:
    wavelength_nm = np.linspace(780.0, 820.0, 120)
    omega = wavelength_nm_to_omega_rad_per_fs(wavelength_nm)
    omega0 = wavelength_nm_to_omega_rad_per_fs(np.asarray([800.0]))[0]
    omega_detuning = omega - omega0

    phi0 = 0.2
    gd = 12.0
    gdd = 180.0
    tod = -75.0

    spectral_phase = (
        phi0
        + gd * omega_detuning
        + 0.5 * gdd * omega_detuning**2
        + (tod / 6.0) * omega_detuning**3
    )
    spectral_intensity = np.exp(-((wavelength_nm - 800.0) ** 2) / (2.0 * 7.0**2))

    time_fs = np.linspace(-60.0, 60.0, 90)
    temporal_intensity = np.exp(-(time_fs**2) / (2.0 * 12.0**2))
    temporal_phase = 0.01 * time_fs

    max_len = wavelength_nm.size
    pad_len = max_len - time_fs.size
    data = np.column_stack(
        [
            np.pad(time_fs, (0, pad_len), constant_values=np.nan),
            np.pad(temporal_intensity, (0, pad_len), constant_values=np.nan),
            np.pad(temporal_phase, (0, pad_len), constant_values=np.nan),
            wavelength_nm,
            spectral_intensity,
            spectral_phase,
        ]
    )
    header = "\t".join(
        [
            "time_fs",
            "temporal_intensity",
            "temporal_phase",
            "wavelength_nm",
            "spectral_intensity",
            "spectral_phase",
        ]
    )
    np.savetxt(path, data, delimiter="\t", header=header, comments="")

    return {"phi0": phi0, "gd": gd, "gdd": gdd, "tod": tod}


class TestFrogSpectralPhaseAnalyzer:
    """Analyzer behavior on synthetic retrieved lineout files."""

    def test_fits_physical_dispersion_terms(self, tmp_path):
        file_path = tmp_path / "retrieved_lineouts.tsv"
        expected = _write_frog_lineout(file_path)

        analyzer = FrogSpectralPhaseAnalyzer(_make_config())
        result = analyzer.analyze_image_file(file_path)

        assert result.data_type == "1d"
        assert result.line_data is not None
        assert result.line_data.shape[1] == 2

        scalars = result.scalars
        assert scalars["frog_phase_phi0_rad"] == pytest.approx(
            expected["phi0"], abs=1e-8
        )
        assert scalars["frog_phase_gd_fs"] == pytest.approx(expected["gd"], rel=1e-6)
        assert scalars["frog_phase_gdd_fs2"] == pytest.approx(expected["gdd"], rel=1e-6)
        assert scalars["frog_phase_tod_fs3"] == pytest.approx(expected["tod"], rel=1e-4)
        assert scalars["frog_phase_flipped"] == 0.0

    def test_metric_suffix_and_prefix(self, tmp_path):
        file_path = tmp_path / "retrieved_lineouts.tsv"
        _write_frog_lineout(file_path)

        analyzer = FrogSpectralPhaseAnalyzer(
            _make_config(),
            metric_prefix="custom_frog",
            metric_suffix="v2",
        )
        result = analyzer.analyze_image_file(file_path)

        assert "custom_frog_gdd_fs2_v2" in result.scalars
        assert "frog_phase_gdd_fs2" not in result.scalars

    def test_direct_array_fit_works_without_weights(self, tmp_path):
        file_path = tmp_path / "retrieved_lineouts.tsv"
        _write_frog_lineout(file_path)
        data = np.genfromtxt(file_path, delimiter="\t", skip_header=1)[:, [3, 5]]

        analyzer = FrogSpectralPhaseAnalyzer(_make_config())
        # Direct analyze_image with no auxiliary_data — the analyzer
        # falls back to an unweighted fit when no `weights` aux column
        # was loaded.
        result = analyzer.analyze_image(data)

        assert result.scalars["frog_phase_gdd_fs2"] == pytest.approx(180.0, rel=1e-6)
