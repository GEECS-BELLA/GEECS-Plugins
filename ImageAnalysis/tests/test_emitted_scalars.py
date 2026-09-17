"""Pin analyzer declarations to real in-memory analysis, without vendor DLLs."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from geecs_schemas.analysis import ANALYZER_SPECS, AnalysisDiagnostic, BeamAnalyzerSpec
from image_analysis.config import create_image_analyzer


@pytest.mark.parametrize("kind", sorted(ANALYZER_SPECS))
def test_declared_scalars_are_emitted(kind):
    if kind in {"haso", "frog_retrieval"}:
        pytest.skip("vendor SDK/DLL; declarations reviewed against analyze_image")
    if kind == "bcave_magspec_stitcher":
        pytest.skip(
            "legacy dict-return analyzer; no guaranteed ImageAnalyzerResult scalars"
        )
    spec = {"kind": kind}
    if kind == "magspec":
        spec.update(
            calibration={"kind": "polynomial", "coeffs": [1, 1]}, energy_range=[1, 128]
        )
    if kind == "line_stitcher":
        spec.update(sibling_devices=["Sibling"], output_label="Stitched")
    if kind == "phase_downramp":
        spec.update(pixel_scale=1, wavelength_nm=800)
    model = ANALYZER_SPECS[kind](**spec)
    image = (
        None if model.image_kind is None else {"type": model.image_kind, "pipeline": []}
    )
    if model.image_kind == "line":
        image["data_loading"] = {"data_type": "tsv"}
    diagnostic = AnalysisDiagnostic(name="Synthetic", analyzer=model, image=image)
    y, x = np.mgrid[:128, :128]
    frame = 1000 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / 200)
    if model.image_kind == "line":
        axis = np.linspace(750, 850, 128)
        frame = np.column_stack((axis, np.exp(-(((axis - 800) / 10) ** 2))))
    auxiliary = None
    if kind == "frog_spectral_phase":
        # The spectral phase analyzer consumes a two-column frequency/phase trace.
        frame = np.column_stack((axis, (axis - 800) ** 2 * 0.001))
    analyzer = create_image_analyzer(diagnostic)
    if kind == "phase_downramp":
        analyzer.bg_data = np.zeros_like(
            frame
        )  # Synthetic equivalent of its background loader.
    result = analyzer.analyze_image(frame, auxiliary_data=auxiliary)
    assert model.emitted_scalars() <= set(result.scalars)
    if kind in {"beam", "standard", "line", "ict", "magspec", "trace", "bcave_mag_opt"}:
        assert model.emitted_scalars() == set(result.scalars)


@pytest.mark.parametrize(
    "enabled,slopes", [(None, False), ([], False), (["x_CoM"], True), (None, True)]
)
def test_beam_declaration_tracks_settings(enabled, slopes):
    spec = BeamAnalyzerSpec(enabled_stats=enabled, compute_slopes=slopes)
    diag = AnalysisDiagnostic(
        name="Synthetic", analyzer=spec, image={"type": "camera", "pipeline": []}
    )
    y, x = np.mgrid[:32, :32]
    frame = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 20)
    assert spec.emitted_scalars() == set(
        create_image_analyzer(diag).analyze_image(frame).scalars
    )
