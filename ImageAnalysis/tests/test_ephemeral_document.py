"""The document-based ephemeral runners: the config editor's live preview seam."""

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic

from image_analysis.ephemeral import render_document_ephemeral, run_document_ephemeral


def _diag(**image):
    return AnalysisDiagnostic(
        name="UC_Test",
        analyzer={"kind": "standard"},
        image={"type": "camera", "bit_depth": 16, **image},
    )


def test_runs_an_unsaved_document_over_frames():
    diag = _diag(pipeline=["roi"], roi={"x_min": 2, "x_max": 6, "y_min": 1, "y_max": 4})
    frame = np.arange(100, dtype=np.uint16).reshape(10, 10)
    (result,) = run_document_ephemeral(diag, [frame])
    assert result.processed_image.shape == (3, 4)


def test_render_returns_object_api_figures():
    (fig,) = render_document_ephemeral(_diag(), [np.ones((4, 4))])
    assert fig.axes


def test_denylisted_kind_is_refused_by_kind():
    diag = AnalysisDiagnostic(
        name="U_Haso",
        analyzer={"kind": "haso", "wavekit_config_file_path": "/wfs.dat"},
    )
    with pytest.raises(ValueError, match="cannot run ephemerally"):
        run_document_ephemeral(diag, [np.ones((4, 4))])


def test_file_path_in_auxiliary_data_is_refused():
    with pytest.raises(ValueError, match="file_path"):
        run_document_ephemeral(
            _diag(), [np.ones((4, 4))], auxiliary_data={"file_path": "x"}
        )
