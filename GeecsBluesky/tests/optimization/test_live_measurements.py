"""Timestamp joins and reductions use owned live arrays, never scan files."""

from types import SimpleNamespace
import builtins

import numpy as np
import pytest
import yaml
from geecs_analysis.compat.v2 import compile_v2
from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET
from geecs_schemas import OptimizerConfig
from geecs_schemas.analysis import AnalysisDiagnostic
from geecs_schemas.optimizer_config import DiagnosticMeasurement

from geecs_bluesky.exceptions import GeecsDeviceDownError
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.optimization.live_frames import LiveFrameSource
from geecs_bluesky.optimization.measurements import (
    CompiledMeasurement,
    CompiledMeasurements,
    compile_measurements,
)


def test_timestamp_join_owns_frames_and_refuses_ambiguous_matches():
    source = LiveFrameSource("test", keep=2)

    class Frame(np.ndarray):
        timestamp = 100.0

    frame = np.ones((2, 2)).view(Frame)
    source._on_update(frame)
    frame[:] = 7
    source.wait_connected(0)
    assert source.frame_at(100.0001).sum() == 4
    assert source.frame_at(100.1) is None
    frame.timestamp = 100.0005
    source._on_update(frame)
    assert source.frame_at(100.0002) is None
    assert source.await_frames([100.1], 0) == {}
    source.close()


def test_source_never_connects():
    with pytest.raises(GeecsDeviceDownError):
        LiveFrameSource("missing").wait_connected(0)


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_real_beam_analysis_join_counts_and_minimum(mode):
    diagnostic = AnalysisDiagnostic.model_validate(
        {
            "schema_version": 2,
            "name": "Camera",
            "image": {"type": "camera"},
            "analyzer": {"kind": "beam"},
        }
    )
    spec = DiagnosticMeasurement(diagnostic="Camera", frames=mode, min_shots=2)
    measurement = CompiledMeasurement(
        "cam",
        spec,
        "Camera",
        "stamp",
        diagnostic.analyzer.emitted_scalars(),
        compile_v2(diagnostic),
    )
    seen = []
    axis = np.arange(64)
    frame = 100 * np.exp(-((axis[:, None] - 32) ** 2 + (axis[None, :] - 30) ** 2) / 100)

    def frames(stamps, timeout):
        seen.extend(stamps)
        return {stamp: frame for stamp in stamps if stamp != 103}

    compiled = CompiledMeasurements(
        (measurement,),
        {},
        {"Camera": SimpleNamespace(await_frames=frames)},
        frozenset({"Camera"}),
        (),
    )
    rows = [{"stamp": stamp + LABVIEW_EPOCH_OFFSET} for stamp in (101, 102, 103)]
    result = compiled.evaluate_bin(rows, compiled.frames_for(rows))
    assert seen == [101, 102, 103]
    assert result.valid_shots == {"cam": 2}
    assert np.isfinite(result.outputs["cam.image_total"])
    result = compiled.evaluate_bin(rows[:1], compiled.frames_for(rows[:1]))
    assert result.valid_shots == {"cam": 1}
    assert np.isnan(result.outputs["cam.image_total"])


def test_missing_scalar_propagates_through_min_expression():
    config = OptimizerConfig(
        vocs={
            "variables": {"Motor:Current": [-1, 1]},
            "objectives": {"score": "MINIMIZE"},
        },
        measurements={"signal": {"signal": "Motor:Current", "min_shots": 2}},
        derived={"score": "min(1, signal)"},
        generator={"name": "random"},
    )
    compiled = compile_measurements(
        config,
        namespace=SimpleNamespace(resolve=lambda name: SimpleNamespace(name="signal")),
        resolver=None,
        shots_per_step=2,
    )
    result = compiled.evaluate_bin([{"signal": 1}], {})
    assert np.isnan(result.outputs["score"])


@pytest.fixture
def diagnostic_setup(tmp_path, monkeypatch):
    from tests.test_namespace import row

    path = tmp_path / "analyzers" / "Camera.yaml"
    path.parent.mkdir()
    document = {
        "name": "Camera",
        "image": {
            "type": "camera",
            "pipeline": ["thresholding"],
            "thresholding": {"method": "constant", "value": 5, "mode": "to_zero"},
        },
        "analyzer": {"kind": "beam", "enabled_stats": ["image_total"]},
    }
    path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(
        ConfigsRepoResolver, "analysis_config_dir", property(lambda _: tmp_path)
    )
    resolver = ConfigsRepoResolver("Test", tmp_path)
    namespace = SimpleNamespace(
        resolve=lambda _: SimpleNamespace(acq_timestamp=SimpleNamespace(name="stamp")),
        experiment="Test",
        roster=SimpleNamespace(
            variables={"Camera": [row("image", choices="image")]}, types={}
        ),
    )
    sources = []

    def source_factory(*args, **kwargs):
        sources.append(args)
        return SimpleNamespace()

    def compile(mode="per_shot", **measurement):
        cfg = OptimizerConfig(
            vocs={
                "variables": {"Motor:Current": [-1, 1]},
                "objectives": {"cam.image_total": "MAXIMIZE"},
            },
            measurements={
                "cam": {
                    "diagnostic": "Camera",
                    "frames": mode,
                    "min_shots": 2,
                    **measurement,
                }
            },
            generator={"name": "random"},
        )
        return compile_measurements(
            cfg,
            namespace=namespace,
            resolver=resolver,
            shots_per_step=2,
            source_factory=source_factory,
        )

    return compile, resolver, path, document, sources


@pytest.mark.parametrize("mode,expected", [("per_shot", 256.0), ("per_bin", 0.0)])
def test_compiled_recipe_survives_source_removal_and_preserves_reduction(
    diagnostic_setup, monkeypatch, mode, expected
):
    compile, resolver, path, _, sources = diagnostic_setup
    compiled = compile(mode)
    assert compiled.output_names == ("cam.image_total",)
    assert compiled.required_devices == frozenset({"Camera"})
    assert sources == [("test:camera:image",)]
    assert resolver.resolve_diagnostic("Camera").source_id == "Camera"
    path.unlink()
    frames = {101.0: np.zeros((8, 8)), 102.0: np.zeros((8, 8))}
    frames[101.0][1::2] = 8
    frames[102.0][::2] = 8
    # Each per-shot thresholded total is 256; mean raw pixels are all 4,
    # below the threshold. This distinguishes average-before-analysis.
    original = {stamp: frame.copy() for stamp, frame in frames.items()}
    rows = [{"stamp": stamp + LABVIEW_EPOCH_OFFSET} for stamp in (101, 102, 102, 103)]
    observed = []
    from geecs_analysis.compat import v2

    analyze = v2.analyze_v2

    def capture(frame, recipe, *, shot):
        observed.append(shot)
        return analyze(frame, recipe, shot=shot)

    monkeypatch.setattr(v2, "analyze_v2", capture)

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not reopen configuration or compile")

    monkeypatch.setattr(resolver, "resolve_diagnostic", forbidden)
    monkeypatch.setattr(v2, "compile_v2", forbidden)
    result = compiled.evaluate_bin(rows, {"Camera": frames})
    assert result.outputs == {"cam.image_total": expected}
    assert result.valid_shots == {"cam": 2}  # duplicate and absent frames don't count
    assert [shot.acq_timestamp for shot in observed] == (
        [101, 102] if mode == "per_shot" else [None]
    )
    assert all(
        shot.device == "Camera" and shot.shot_number is None for shot in observed
    )
    for stamp, frame in frames.items():
        np.testing.assert_array_equal(frame, original[stamp])


@pytest.mark.parametrize("mode", ["per_shot", "per_bin"])
def test_failed_and_nonfinite_frames_do_not_meet_minimum(diagnostic_setup, mode):
    compile, _, _, _, _ = diagnostic_setup
    compiled = compile(mode, overrides={"image": {"pipeline": []}})
    rows = [{"stamp": stamp + LABVIEW_EPOCH_OFFSET} for stamp in (101, 102)]
    result = compiled.evaluate_bin(
        rows, {"Camera": {101: np.full((8, 8), 8), 102: np.full((8, 8), np.nan)}}
    )
    assert result.valid_shots == {"cam": 1 if mode == "per_shot" else 0}
    assert np.isnan(result.outputs["cam.image_total"])


def test_overrides_discover_compiled_scalars_and_never_import_legacy(
    diagnostic_setup, monkeypatch
):
    compile, resolver, path, _, _ = diagnostic_setup
    original_import = builtins.__import__

    def import_without_legacy(name, *args, **kwargs):
        if name.startswith("image_analysis"):
            raise AssertionError("optimizer must use geecs-analysis directly")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_legacy)
    before = path.read_bytes()
    compiled = compile(
        overrides={"analyzer": {"enabled_stats": ["image_total", "image_peak_value"]}}
    )
    assert compiled.output_names == ("cam.image_peak_value", "cam.image_total")
    assert path.read_bytes() == before
    assert resolver.diagnostic_device("Camera") == "Camera"
    result = compiled.evaluate_bin(
        [{"stamp": stamp + LABVIEW_EPOCH_OFFSET} for stamp in (101, 102)],
        {"Camera": {stamp: np.full((8, 8), 8) for stamp in (101, 102)}},
    )
    assert result.outputs == {"cam.image_total": 512, "cam.image_peak_value": 8}


def test_unsupported_recipe_is_refused_before_frame_source_creation(diagnostic_setup):
    compile, _, _, _, sources = diagnostic_setup
    with pytest.raises(GeecsConfigurationError, match="not ported: transforms"):
        compile(
            overrides={
                "image": {
                    "pipeline": ["transforms"],
                    "transforms": {"rotation_angle": 45},
                }
            }
        )
    assert sources == []
