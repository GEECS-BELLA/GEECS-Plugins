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


class TestRenderedEphemeral:
    """The render form: the analyzer's own figure, object API, still write-free."""

    def test_one_figure_per_frame_with_image_and_colorbar(self, configs_tree):
        from image_analysis.ephemeral import render_diagnostic_ephemeral

        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 100, (8, 6)).astype(np.uint16) for _ in range(2)]
        figs = render_diagnostic_ephemeral(
            "UC_Test", frames, config_dir=configs_tree, cmap="viridis", window=(1, 99)
        )
        assert len(figs) == 2
        for fig in figs:
            ax = fig.axes[0]
            assert len(ax.images) == 1  # the processed image
            assert len(fig.axes) == 2  # + the colorbar the base renderer skipped
            assert ax.images[0].get_cmap().name == "viridis"
            lo, hi = ax.images[0].get_clim()
            assert 0 <= lo < hi <= 100  # the percentile window became vmin/vmax

    def test_no_pyplot_state_is_created(self, configs_tree):
        """Object-API figures only — safe on a request threadpool."""
        import matplotlib.pyplot as plt

        from image_analysis.ephemeral import render_diagnostic_ephemeral

        plt.close("all")
        render_diagnostic_ephemeral(
            "UC_Test", [np.ones((4, 4))], config_dir=configs_tree
        )
        assert plt.get_fignums() == []

    def test_render_form_keeps_the_write_gate(self, configs_tree, tmp_path):
        from image_analysis.ephemeral import render_diagnostic_ephemeral

        with pytest.raises(ValueError, match="file_path"):
            render_diagnostic_ephemeral(
                "UC_Test",
                [np.ones((4, 4))],
                config_dir=configs_tree,
                auxiliary_data={"file_path": tmp_path / "x.png"},
            )
        _write_diagnostic(
            tmp_path / "analyzers" / "HTU" / "U_HasoR.yaml",
            "U_HasoR",
            image_analyzer=_HASO_PATH,
        )
        with pytest.raises(ValueError, match="cannot run ephemerally"):
            render_diagnostic_ephemeral(
                "U_HasoR", [np.ones((4, 4))], config_dir=tmp_path
            )

    def test_frame_figure_is_base_render_only(self):
        from image_analysis.ephemeral import render_frame_figure

        fig = render_frame_figure(np.arange(16.0).reshape(4, 4), window=(0, 100))
        ax = fig.axes[0]
        assert len(ax.images) == 1 and len(fig.axes) == 2
        assert ax.images[0].get_clim() == (0.0, 15.0)
        assert not ax.lines  # no overlays on an averaged image

    def test_the_analyzers_own_overlays_are_drawn(self):
        """The point of the seam: render_image is the ANALYZER's, not the base."""
        from image_analysis.analyzers.beam_analyzer import BeamAnalyzer
        from image_analysis.tools.rendering import render_result_figure

        image = np.zeros((6, 8))
        image[2, 3] = 10.0
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=image,
            render_data={
                "horizontal_projection": image.sum(axis=0),
                "vertical_projection": image.sum(axis=1),
            },
        )
        fig = render_result_figure(BeamAnalyzer, result)
        ax = fig.axes[0]
        assert len(ax.images) == 1
        assert ax.lines, "BeamAnalyzer's projection overlays must reach the seam's axes"

    @pytest.mark.parametrize(
        "class_path",
        [
            "image_analysis.analyzers.standard_analyzer.StandardAnalyzer",
            "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
            "image_analysis.analyzers.Undulator.hi_res_mag_cam_analyzer.HiResMagCamAnalyzer",
            "image_analysis.analyzers.magspec_manual_calib_analyzer.MagSpecManualCalibAnalyzer",
        ],
    )
    def test_2d_renderers_honour_the_ax_contract(self, class_path):
        """Every 2D render_image draws INTO the axes it is given, without pyplot.

        The load-bearing contract of the render seam (ImageAnalysis
        CLAUDE.md "Ephemeral runs"): a renderer that ignored ``ax`` would
        return an empty seam figure AND leak a pyplot-registered figure
        per request on a server thread.
        """
        import importlib

        import matplotlib.pyplot as plt

        from image_analysis.tools.rendering import render_result_figure

        module_path, class_name = class_path.rsplit(".", 1)
        renderer = getattr(importlib.import_module(module_path), class_name)
        plt.close("all")
        result = ImageAnalyzerResult(data_type="2d", processed_image=np.ones((5, 7)))
        fig = render_result_figure(renderer, result, cmap="viridis")
        assert len(fig.axes[0].images) == 1
        assert plt.get_fignums() == []

    def test_1d_renderer_honours_the_ax_contract(self):
        import matplotlib.pyplot as plt

        from image_analysis.analyzers.standard_1d_analyzer import Standard1DAnalyzer
        from image_analysis.tools.rendering import render_result_figure

        plt.close("all")
        line = np.column_stack([np.arange(10.0), np.arange(10.0) ** 2])
        result = ImageAnalyzerResult(data_type="1d", line_data=line)
        fig = render_result_figure(
            Standard1DAnalyzer.__new__(Standard1DAnalyzer), result
        )
        assert fig.axes[0].lines
        assert plt.get_fignums() == []

    def test_unrenderable_result_is_a_render_error(self):
        from image_analysis.analyzers.standard_analyzer import StandardAnalyzer
        from image_analysis.tools.rendering import RenderError, render_result_figure

        result = ImageAnalyzerResult(data_type="scalars_only", scalars={"x": 1.0})
        with pytest.raises(RenderError, match="data_type='2d'"):
            render_result_figure(StandardAnalyzer, result)

    @pytest.mark.parametrize(
        "image",
        [np.zeros((0, 5)), np.full((4, 4), np.nan), np.ones((4, 4)), np.zeros((4, 4))],
        ids=["empty", "all-nan", "constant", "zeros"],
    )
    def test_window_limits_degrade_never_raise(self, image):
        import warnings

        from image_analysis.tools.rendering import window_limits

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert window_limits(image, (1, 99)) == {}
