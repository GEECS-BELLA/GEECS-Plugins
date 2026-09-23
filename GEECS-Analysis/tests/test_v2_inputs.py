"""File declarations stay inert until a source explicitly supplies frame inputs."""

import subprocess
import sys

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic

from geecs_analysis.compat.v2 import UnsupportedRecipe, analyze_v2, compile_v2
from geecs_data_utils.frames import Frame


def document():
    return AnalysisDiagnostic.model_validate(
        {
            "name": "Camera",
            "analyzer": {"kind": "standard"},
            "image": {
                "type": "camera",
                "pipeline": ["background", "background"],
                "background": {
                    "method": "from_file",
                    "file_path": "{scan_dir}/dark.npy",
                    "constant_level": 7,
                    "additional_constant": 2,
                },
            },
        }
    )


def test_explicit_opt_in_snapshots_one_request_for_repeated_steps():
    doc = document()
    with pytest.raises(UnsupportedRecipe, match="background"):
        compile_v2(doc)
    recipe = compile_v2(doc, allow_file_backgrounds=True)
    (request,) = recipe.file_backgrounds
    assert request.path == "{scan_dir}/dark.npy"
    assert request.fallback_level == 7
    doc.image.background.file_path = "changed.npy"
    doc.image.background.additional_constant = 100
    assert request.path == "{scan_dir}/dark.npy"
    data = np.arange(6).reshape(2, 3)
    dark = Frame.from_array(np.full((2, 3), 10))
    with pytest.raises(ValueError, match="Missing frame input"):
        analyze_v2(data, recipe)
    actual = analyze_v2(data, recipe, inputs={request.key: dark})
    np.testing.assert_array_equal(actual.frame.data, data - 24)


def test_inactive_background_has_no_request_and_scan_sources_stay_unsupported():
    doc = document()
    doc.image.pipeline = []
    assert not compile_v2(doc).file_backgrounds
    doc.scan.background_source = {"scan_number": 2}
    with pytest.raises(UnsupportedRecipe, match="Scan backgrounds"):
        compile_v2(doc, allow_file_backgrounds=True)


def test_opt_in_compilation_does_not_import_readers_or_numerical_dependencies():
    code = f"""
import sys
from geecs_schemas.analysis import AnalysisDiagnostic
from geecs_analysis.compat.v2 import compile_v2
document = AnalysisDiagnostic.model_validate_json({document().model_dump_json()!r})
assert compile_v2(document, allow_file_backgrounds=True).file_backgrounds
for name in ('numpy', 'scipy', 'matplotlib', 'geecs_data_utils', 'image_analysis'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)
