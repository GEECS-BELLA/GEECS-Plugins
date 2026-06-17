"""Integration smoke test for BeamAnalyzer.

Verifies the end-to-end path: real image file → BeamAnalyzer → finite scalars.
Algorithm correctness is tested in ImageAnalysis/tests/analyzers/test_beam_analyzer.py
using synthetic data (no network required).

Run with:
    pytest -m "integration and data" tests/integration/data/image_analysis/
"""

import math

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

DEV_NAME = "UC_Amp4_IR_input"
# Standalone BeamAnalyzer emits BARE scalar keys post-#412. ScanAnalysis
# is the layer that applies the ``output_name`` prefix; a Mode-1
# notebook user calling ``analyzer.analyze_image_file(path)`` sees the
# raw key names below.
EXPECTED_SCALARS = ["x_CoM", "y_CoM", "image_total", "image_peak_value"]


def test_beam_analyzer_end_to_end(canonical_scan):
    """BeamAnalyzer loads a real image and returns finite scalars."""
    from image_analysis.analyzers.beam_analyzer import BeamAnalyzer
    from image_analysis.config.array2d_processing import (
        BackgroundConfig,
        CameraConfig,
    )

    scan = canonical_scan("undulator_2d")
    file_path = scan.data_frame[f"{DEV_NAME}_expected_path"].iloc[0]
    assert file_path is not None and str(file_path) != "nan"

    # ``CameraConfig.name`` was removed in #412; identity now lives on
    # the analyzer instance via ``output_name``. Standalone construction
    # without an output_name is fine — keys come out bare.
    analyzer = BeamAnalyzer(
        CameraConfig(
            bit_depth=16,
            background=BackgroundConfig(method="constant", constant_level=0),
        )
    )
    result = analyzer.analyze_image_file(file_path)

    assert result.processed_image is not None
    assert result.processed_image.ndim == 2
    for key in EXPECTED_SCALARS:
        assert key in result.scalars, f"Missing scalar: {key}"
        assert math.isfinite(result.scalars[key]), f"Non-finite: {key}"


def test_load_diagnostic_with_production_metadata(canonical_scan):
    """End-to-end: ``load_diagnostic`` on a production YAML with ``metadata:``.

    Regression test for the ``extra="forbid"`` migration. Before this
    test existed, the integration suite constructed ``CameraConfig``
    programmatically (no YAML lookup) for 2D tests, so it never
    exercised production YAMLs that carry free-form ``metadata:``
    fields. Flipping ``CameraConfig`` to ``extra="forbid"`` without
    ALSO declaring ``metadata`` as a typed ``Optional[Dict[str, Any]]``
    field broke ``load_diagnostic("Amp4Input")`` in production while
    the test suite stayed green.

    This test loads the same ``Amp4Input.yaml`` (Undulator HTU) the
    user hit in production. It pins the contract:

    * ``metadata:`` round-trips intact (typed as
      ``Optional[Dict[str, Any]]`` — keys are not validated).
    * Other top-level documentation-only fields stay in the schema as
      they're added.
    """
    from image_analysis.config import load_diagnostic, create_image_analyzer

    diag = load_diagnostic("Amp4Input")
    # The metadata block (location/notes/spatial_calibration) must
    # round-trip; the validator can no longer accept it via
    # ``extra="allow"``, so this catches future regressions where
    # ``metadata`` gets removed or its type narrowed.
    assert diag.image is not None
    assert isinstance(diag.image.metadata, dict)
    assert "location" in diag.image.metadata
    assert "spatial_calibration" in diag.image.metadata

    # And the full factory path still works end-to-end — analyzer
    # construction is the real failure mode users hit.
    analyzer = create_image_analyzer(diag)
    file_path = (
        canonical_scan("undulator_2d").data_frame[f"{DEV_NAME}_expected_path"].iloc[0]
    )
    result = analyzer.analyze_image_file(file_path)
    assert result.processed_image is not None
