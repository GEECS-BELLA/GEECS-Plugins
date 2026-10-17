"""``load_diagnostic``: one document by stem or path, the client config root as the default."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from geecs_schemas.analysis import AnalysisRecipe, HimgJob

from scan_analysis.config import load_diagnostic

RECIPE = {
    "schema_version": 3,
    "device": "UC_Cam",
    "input": {"kind": "camera"},
    "measure": {"kind": "beam"},
    "scan": {"priority": 7, "average_frames_first": False},
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    analyzers = tmp_path / "analyzers" / "HTU"
    analyzers.mkdir(parents=True)
    (analyzers / "UC_Cam.yaml").write_text(yaml.safe_dump(RECIPE))
    (analyzers / "HasoStack.yaml").write_text(
        yaml.safe_dump({"kind": "himg_to_stack", "device": "U_HasoLift"})
    )
    (analyzers / "Legacy.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "name": "Old", "image_analyzer": "x.Y"})
    )
    return tmp_path


def test_a_stem_resolves_under_the_tree_and_carries_its_source_id(tree):
    document = load_diagnostic("UC_Cam", config_dir=tree)
    assert isinstance(document, AnalysisRecipe)
    assert document.source_id == "UC_Cam" and document.scan.priority == 7


def test_a_path_needs_no_tree_and_a_job_loads_as_its_kind(tree):
    document = load_diagnostic(tree / "analyzers" / "HTU" / "HasoStack.yaml")
    assert isinstance(document, HimgJob) and document.source_id == "HasoStack"


def test_overrides_merge_into_the_yaml_before_validation(tree):
    document = load_diagnostic(
        "UC_Cam", config_dir=tree, overrides={"scan": {"average_frames_first": True}}
    )
    assert document.scan.average_frames_first is True
    assert document.scan.priority == 7, "a nested override keeps its siblings"
    with pytest.raises(ValueError, match="Invalid diagnostic config"):
        load_diagnostic(
            "UC_Cam", config_dir=tree, overrides={"scan": {"priority": "x"}}
        )


def test_refusals_name_the_problem(tree, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Invalid diagnostic config"):
        load_diagnostic("Legacy", config_dir=tree)
    with pytest.raises(KeyError):
        load_diagnostic("Missing", config_dir=tree)
    with pytest.raises(FileNotFoundError):
        load_diagnostic(tmp_path / "absent.yaml")
    from geecs_data_utils import ScanPaths

    # No client config.ini at all (the hermetic case) reads the same way.
    monkeypatch.setattr(ScanPaths, "paths_config", None, raising=False)
    with pytest.raises(ValueError, match="config_dir was not provided"):
        load_diagnostic("UC_Cam")
