"""Timestamp joins and reductions use owned live arrays, never scan files."""

from types import SimpleNamespace

import numpy as np
import pytest
from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET
from geecs_schemas import OptimizerConfig
from geecs_schemas.analysis import AnalysisDiagnostic
from geecs_schemas.optimizer_config import DiagnosticMeasurement

from geecs_bluesky.exceptions import GeecsDeviceDownError
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
        diagnostic,
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
