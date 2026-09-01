"""Tests for ``image_analysis.ephemeral`` — the write-free run seam.

Hermetic: diagnostics are tmp_path YAML trees, frames are synthetic
arrays. The contract under test is the one documented in CLAUDE.md's
"Ephemeral runs" section: in-memory frames only, ``file_path`` refused,
unconditional writers denylisted before import, and nothing written
anywhere on the happy path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from image_analysis.config import list_diagnostics
from image_analysis.ephemeral import EPHEMERAL_DENYLIST, run_diagnostic_ephemeral
from image_analysis.types import ImageAnalyzerResult

_STANDARD_PATH = "image_analysis.analyzers.standard_analyzer.StandardAnalyzer"
_HASO_PATH = "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor"
_GRENOUILLE_PATH = "image_analysis.analyzers.grenouille_analyzer.GrenouilleAnalyzer"


def _write_diagnostic(path: Path, name: str, *, image_analyzer=_STANDARD_PATH) -> None:
    """Write a minimal 2D diagnostic YAML at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "image_analyzer": image_analyzer,
        "image": {"type": "camera", "bit_depth": 16},
        "scan": {"priority": 100},
    }
    if image_analyzer == _HASO_PATH:
        # HASO-style: no embedded image config, kwargs on the spec.
        payload = {
            "name": name,
            "image_analyzer": {"class_path": image_analyzer, "kwargs": {}},
            "scan": {"priority": 100},
        }
    path.write_text(yaml.safe_dump(payload))


@pytest.fixture
def configs_tree(tmp_path: Path) -> Path:
    """A unified-configs tree with one plain 2D diagnostic."""
    _write_diagnostic(tmp_path / "analyzers" / "HTU" / "UC_Test.yaml", "UC_Test")
    return tmp_path


def _tree_snapshot(root: Path) -> set[Path]:
    return set(root.rglob("*"))


class TestRunDiagnosticEphemeral:
    """The runner: config in, results out, filesystem untouched."""

    def test_one_result_per_frame_in_order(self, configs_tree):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 100, (8, 6)).astype(np.uint16) for _ in range(3)]
        results = run_diagnostic_ephemeral("UC_Test", frames, config_dir=configs_tree)
        assert len(results) == 3
        for result, frame in zip(results, frames):
            assert isinstance(result, ImageAnalyzerResult)
            assert result.data_type == "2d"
            assert result.processed_image.shape == frame.shape

    def test_nothing_is_written(self, configs_tree, tmp_path, monkeypatch):
        """The happy path leaves the configs tree and cwd untouched."""
        workdir = tmp_path / "cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        before = _tree_snapshot(tmp_path)
        run_diagnostic_ephemeral(
            "UC_Test", [np.ones((4, 4), dtype=np.uint16)], config_dir=configs_tree
        )
        assert _tree_snapshot(tmp_path) == before

    def test_overrides_reach_the_pipeline(self, configs_tree):
        """A per-call ROI override crops the processed image.

        The pipeline list is the source of truth for what runs
        (empty default = raw pass-through), so the override supplies
        both the step and its config.
        """
        frame = np.arange(100, dtype=np.uint16).reshape(10, 10)
        (result,) = run_diagnostic_ephemeral(
            "UC_Test",
            [frame],
            config_dir=configs_tree,
            overrides={
                "image": {
                    "pipeline": {"steps": ["roi"]},
                    "roi": {"x_min": 2, "x_max": 6, "y_min": 1, "y_max": 4},
                }
            },
        )
        assert result.processed_image.shape == (3, 4)

    def test_auxiliary_data_is_forwarded_per_frame_copy(
        self, configs_tree, monkeypatch
    ):
        """Aux reaches analyze_image as a fresh top-level copy per frame."""
        from image_analysis.analyzers.standard_analyzer import StandardAnalyzer

        seen: list[dict] = []
        original = StandardAnalyzer.analyze_image

        def recording(self, image, auxiliary_data=None):
            seen.append(auxiliary_data)
            return original(self, image, auxiliary_data)

        monkeypatch.setattr(StandardAnalyzer, "analyze_image", recording)
        aux = {"shot": 7}
        results = run_diagnostic_ephemeral(
            "UC_Test",
            [np.ones((4, 4), dtype=np.uint16)] * 2,
            config_dir=configs_tree,
            auxiliary_data=aux,
        )
        assert all(isinstance(r, ImageAnalyzerResult) for r in results)
        assert seen == [{"shot": 7}, {"shot": 7}]
        assert seen[0] is not aux and seen[0] is not seen[1]
        assert aux == {"shot": 7}

    def test_file_path_in_auxiliary_data_is_refused(self, configs_tree, tmp_path):
        with pytest.raises(ValueError, match="file_path.*forbidden"):
            run_diagnostic_ephemeral(
                "UC_Test",
                [np.ones((4, 4), dtype=np.uint16)],
                config_dir=configs_tree,
                auxiliary_data={"file_path": tmp_path / "shot.png"},
            )

    def test_denylisted_analyzer_is_refused_before_import(self, tmp_path):
        """HASO is refused by class-path string, not by a failed import.

        A ``ValueError`` naming the ephemeral contract (rather than an
        ``ImportError`` from the vendor SDK) proves the check runs
        before ``create_image_analyzer`` touches the class.
        """
        _write_diagnostic(
            tmp_path / "analyzers" / "HTU" / "U_HasoLift.yaml",
            "U_HasoLift",
            image_analyzer=_HASO_PATH,
        )
        assert _HASO_PATH in EPHEMERAL_DENYLIST
        assert (
            _GRENOUILLE_PATH in EPHEMERAL_DENYLIST
        )  # un-gated temp files + DLL subprocess
        with pytest.raises(ValueError, match="cannot run ephemerally"):
            run_diagnostic_ephemeral(
                "U_HasoLift", [np.ones((4, 4))], config_dir=tmp_path
            )

    def test_empty_frames_returns_empty_list(self, configs_tree):
        assert run_diagnostic_ephemeral("UC_Test", [], config_dir=configs_tree) == []


class TestListDiagnostics:
    """``list_diagnostics``: sorted stems over the analyzers/ tree."""

    def test_sorted_stems_across_namespaces(self, tmp_path):
        _write_diagnostic(tmp_path / "analyzers" / "HTU" / "UC_B.yaml", "UC_B")
        _write_diagnostic(tmp_path / "analyzers" / "PW" / "UC_A.yaml", "UC_A")
        assert list_diagnostics(config_dir=tmp_path) == ["UC_A", "UC_B"]

    def test_missing_analyzers_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Analyzer directory"):
            list_diagnostics(config_dir=tmp_path)
