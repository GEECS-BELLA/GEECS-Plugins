"""Streaming groups preserve per-shot/per-bin numerical and failure semantics."""

import numpy as np
import weakref
import pytest
from geecs_data_utils.frames import ShotMeta
from geecs_schemas.analysis import AnalysisDiagnostic

from geecs_analysis.compat.v2 import analyze_v2, compile_v2
from geecs_analysis.compat.v2_run import ShotGroup, run_units


def recipe(**image):
    return compile_v2(
        AnalysisDiagnostic.model_validate(
            {
                "name": "Camera",
                "analyzer": {"kind": "standard"},
                "image": {"type": "camera", **image},
            }
        )
    )


def test_average_before_nonlinear_processing_is_not_average_of_processed_shots():
    compiled = recipe(
        pipeline=["thresholding"],
        thresholding={
            "method": "constant",
            "value": 15,
            "mode": "to_zero",
        },
    )
    arrays = {
        1: np.array([[0, 20]], dtype=np.uint16),
        2: np.array([[20, 0]], dtype=np.uint16),
    }
    shots = list(
        run_units(
            compiled, [ShotGroup(1, (1,)), ShotGroup(2, (2,))], arrays.__getitem__
        )
    )
    (binned,) = run_units(
        compiled,
        [ShotGroup(7, (1, 2))],
        arrays.__getitem__,
        average_before_analysis=True,
    )
    np.testing.assert_array_equal(binned.measurement.frame.data, [[0, 0]])
    np.testing.assert_array_equal(
        np.mean([r.measurement.frame.data for r in shots], axis=0), [[10, 10]]
    )
    assert binned.measurement.frame.shot is None
    assert shots[0].measurement.frame.shot == ShotMeta("Camera", 1)
    assert binned.loaded_shots == (1, 2) and binned.group.key == 7


def test_loading_is_lazy_ordered_and_one_group_at_a_time():
    called = []
    buffer = np.zeros((2, 2), dtype=np.uint16)

    def load(shot):
        called.append(shot)
        buffer[:] = shot
        return buffer  # A source may reuse a native read buffer.

    outcomes = run_units(
        recipe(),
        [ShotGroup(7, (3, 1)), ShotGroup(8, (2,))],
        load,
        average_before_analysis=True,
    )
    assert not called
    first = next(outcomes)
    assert called == [3, 1]
    np.testing.assert_array_equal(first.measurement.frame.data, np.full((2, 2), 2))
    second = next(outcomes)
    assert called == [3, 1, 2]
    assert second.group.key == 8
    assert first.loaded_shots == (3, 1)
    with pytest.raises(StopIteration):
        next(outcomes)


def test_failed_member_does_not_change_scalar_write_membership_and_later_groups_continue():
    def load(shot):
        if shot in {2, 3}:
            raise OSError(f"missing {shot}")
        return np.full((2, 2), shot)

    results = list(
        run_units(
            recipe(),
            [ShotGroup(0, (1, 2)), ShotGroup(1, (3,)), ShotGroup(2, (4,))],
            load,
            average_before_analysis=True,
        )
    )
    assert results[0].group.shots == (1, 2)
    assert results[0].loaded_shots == (1,)
    assert results[0].load_failures[0].shot == 2
    assert results[0].load_failures[0].message == "missing 2"
    assert results[0].measurement is not None and results[0].error is None
    assert results[1].measurement is None
    assert results[1].error == "No loadable inputs in group"
    assert results[2].measurement is not None


def test_bad_geometry_is_an_explicit_outcome_and_does_not_block_later_groups():
    arrays = {1: np.ones((2, 2)), 2: np.ones((3, 3)), 3: np.ones((2, 2))}
    results = list(
        run_units(
            recipe(),
            [ShotGroup(1, (1, 2)), ShotGroup(2, (3,))],
            arrays.__getitem__,
            average_before_analysis=True,
        )
    )
    assert results[0].measurement is None and results[0].error
    assert results[0].loaded_shots == (1, 2)
    assert not results[0].load_failures
    assert results[1].measurement is not None


def test_trace_raw_average_preserves_float32_precision_before_scaling():
    doc = AnalysisDiagnostic.model_validate(
        {
            "name": "Spectrum",
            "analyzer": {"kind": "line"},
            "image": {
                "type": "line",
                "data_loading": {"data_type": "npy"},
                "x_scale_factor": 1000,
                "storage_dtype": "float32",
            },
        }
    )
    compiled = compile_v2(doc)
    arrays = [
        np.column_stack(
            (
                np.linspace(0.05, 0.15, 31),
                np.random.default_rng(seed).uniform(1, 10, 31),
            )
        ).astype(np.float32)
        for seed in range(3)
    ]
    expected = analyze_v2(np.mean(arrays, axis=0), compiled)
    (actual,) = run_units(
        compiled,
        [ShotGroup(1, (1, 2, 3))],
        lambda shot: arrays[shot - 1],
        average_before_analysis=True,
    )
    np.testing.assert_array_equal(
        actual.measurement.frame.as_trace(), expected.frame.as_trace()
    )
    assert all(np.isfinite(value) for value in expected.scalars.values())
    assert dict(actual.measurement.scalars) == dict(expected.scalars)


def test_background_binding_failure_precedes_every_source_read():
    doc = AnalysisDiagnostic.model_validate(
        {
            "name": "Camera",
            "analyzer": {"kind": "standard"},
            "image": {
                "type": "camera",
                "pipeline": ["background"],
                "background": {"method": "from_file", "file_path": "dark.npy"},
            },
        }
    )
    compiled = compile_v2(doc, allow_file_backgrounds=True)

    def load(shot):
        raise AssertionError("preflight failed to stop source reads")

    with pytest.raises(ValueError, match="Missing frame input"):
        list(run_units(compiled, [ShotGroup(1, (1,))], load))


def test_per_shot_provenance_and_non_array_source_refusal():
    identity = ShotMeta("device", 1, 123.5)
    arrays = {1: np.ones((2, 2)), 2: None}
    results = list(
        run_units(
            recipe(),
            [ShotGroup(1, (1,)), ShotGroup(2, (2,))],
            arrays.__getitem__,
            shot_metadata={1: identity},
        )
    )
    assert results[0].measurement.frame.shot is identity
    assert results[1].measurement is None
    assert "native ndarray" in results[1].load_failures[0].message


@pytest.mark.parametrize("shots", [(), (1, 1), (0,), (True,), (1.5,)])
def test_invalid_membership_is_rejected(shots):
    with pytest.raises(ValueError):
        ShotGroup(1, shots)


def test_per_shot_mode_refuses_multi_member_groups_without_loading():
    called = []
    with pytest.raises(ValueError, match="single-member"):
        list(run_units(recipe(), [ShotGroup(1, (1, 2))], called.append))
    assert not called


def test_raw_source_arrays_are_released_before_yielding():
    references = []

    def load(shot):
        data = np.full((3, 3), shot)
        references.append(weakref.ref(data))
        return data

    outcomes = run_units(
        recipe(), [ShotGroup(1, (1, 2))], load, average_before_analysis=True
    )
    result = next(outcomes)
    assert result.measurement is not None
    assert all(reference() is None for reference in references)


def test_bad_identity_is_refused_before_reading_source():
    called = []
    with pytest.raises(ValueError, match="Shot metadata"):
        list(
            run_units(
                recipe(),
                [ShotGroup(1, (1,))],
                called.append,
                shot_metadata={1: ShotMeta("Camera", 2)},
            )
        )
    assert not called


def test_processing_failure_does_not_retry_or_discard_later_shots():
    arrays = {1: np.ones(3), 2: np.ones((2, 2))}
    results = list(
        run_units(
            recipe(), [ShotGroup(1, (1,)), ShotGroup(2, (2,))], arrays.__getitem__
        )
    )
    assert (
        results[0].measurement is None
        and "Camera input must be HxW" in results[0].error
    )
    assert results[0].loaded_shots == (1,) and not results[0].load_failures
    assert results[1].measurement is not None


class _Tracked(np.ndarray):
    """A raw frame, and every array numpy derives from it, weakly tracked."""

    live: list = []

    def __array_finalize__(self, obj):
        _Tracked.live.append(weakref.ref(self))


def _trace_recipe():
    return compile_v2(
        AnalysisDiagnostic.model_validate(
            {
                "name": "Spectrum",
                "analyzer": {"kind": "line"},
                "image": {
                    "type": "line",
                    "data_loading": {"data_type": "npy"},
                    "storage_dtype": "float32",
                },
            }
        )
    )


@pytest.mark.parametrize("kind", ["uint16 camera", "float32 trace"])
def test_raw_bin_fold_equals_the_stacked_mean_bit_for_bit(kind, monkeypatch):
    """The running sum is numpy's reduction order along the first axis.

    So the quotient equals ``np.mean`` over the stacked copies exactly, in
    the dtype the stacked mean returns, with a bad load in the middle left
    out of both. This holds for frames of more than one element: a stack of
    1x1 frames reduces along its only (contiguous) axis, where numpy sums
    pairwise, and a float fold can then differ in the last bit.
    """
    import geecs_analysis.compat.v2_run as v2_run

    rng = np.random.default_rng(1025)
    if kind == "uint16 camera":
        compiled = recipe()
        frames = [rng.integers(0, 4096, (12, 20), dtype=np.uint16) for _ in range(40)]
    else:
        compiled = _trace_recipe()
        x = np.linspace(0.05, 0.15, 31)
        frames = [
            np.column_stack((x, rng.uniform(1, 10, 31))).astype(np.float32)
            for _ in range(64)
        ]
    bad = len(frames) // 2
    buffer = np.empty_like(frames[0])

    def load(shot):
        if shot == bad:
            raise OSError("truncated")
        buffer[:] = frames[shot - 1]
        return buffer  # A reused read buffer, as a streaming source has.

    seen = []
    real = v2_run.analyze_v2

    def capture(raw, *args, **kwargs):
        seen.append(raw.copy())
        return real(raw, *args, **kwargs)

    monkeypatch.setattr(v2_run, "analyze_v2", capture)
    shots = tuple(range(1, len(frames) + 1))
    (result,) = run_units(
        compiled, [ShotGroup(1, shots)], load, average_before_analysis=True
    )
    good = [frames[n - 1].copy() for n in shots if n != bad]
    expected = np.mean(good, axis=0)
    assert result.error is None and result.measurement is not None
    assert result.group.shots == shots
    assert result.loaded_shots == tuple(n for n in shots if n != bad)
    assert [f.shot for f in result.load_failures] == [bad]
    (raw,) = seen
    assert raw.dtype == expected.dtype
    assert raw.tobytes() == expected.tobytes()


def test_a_long_raw_bin_average_keeps_at_most_two_raw_frames_alive():
    """A 300-shot bin folds each frame and drops it; a list would hold all.

    The loader returns fresh frames of a tracked subclass, so a retained
    frame or any copy of one stays visible; before each load the count of
    earlier frames (and copies) still alive is recorded.
    """
    import gc

    _Tracked.live.clear()
    alive = []

    def load(shot):
        gc.collect()
        alive.append(sum(ref() is not None for ref in _Tracked.live))
        return np.full((64, 64), shot, dtype=np.uint16).view(_Tracked)

    shots = tuple(range(1, 301))
    (result,) = run_units(
        recipe(), [ShotGroup(1, shots)], load, average_before_analysis=True
    )
    assert result.error is None and result.loaded_shots == shots
    np.testing.assert_array_equal(
        result.measurement.frame.data, np.full((64, 64), 150.5)
    )
    assert len(alive) == 300
    assert max(alive) <= 2, f"up to {max(alive)} raw frames alive of 300"


def test_a_broadcastable_member_is_still_an_incompatible_shape():
    """A (1, N) frame would broadcast into an (M, N) sum; the stack refused it."""
    arrays = {1: np.ones((2, 3)), 2: np.ones((1, 3)), 3: np.ones((2, 3))}
    (result,) = run_units(
        recipe(),
        [ShotGroup(1, (1, 2, 3))],
        arrays.__getitem__,
        average_before_analysis=True,
    )
    assert result.measurement is None and "Incompatible raw shapes" in result.error
    assert result.loaded_shots == (1, 2, 3) and not result.load_failures


@pytest.mark.parametrize("dtypes", [(np.uint8, np.int32), (np.int16, np.uint16)])
def test_mixed_integer_members_average_as_the_promoted_stack(dtypes, monkeypatch):
    """Integer samples all sum in float64, so a mid-bin switch folds exactly."""
    import geecs_analysis.compat.v2_run as v2_run

    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 200, (4, 5)).astype(dtypes[n >= 5]) for n in range(9)]
    seen = []
    real = v2_run.analyze_v2

    def capture(raw, *args, **kwargs):
        seen.append(raw.copy())
        return real(raw, *args, **kwargs)

    monkeypatch.setattr(v2_run, "analyze_v2", capture)
    (result,) = run_units(
        recipe(),
        [ShotGroup(1, tuple(range(1, 10)))],
        lambda shot: frames[shot - 1],
        average_before_analysis=True,
    )
    expected = np.mean(frames, axis=0)
    assert result.error is None
    assert seen[0].dtype == expected.dtype
    assert seen[0].tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtypes", [(np.float32, np.float64), (np.uint16, np.float32)])
def test_a_member_that_would_promote_the_sum_is_an_incompatible_dtype(dtypes):
    """The stack sums every member in the promoted dtype; a fold cannot.

    Several members are folded before the switch, so the sum already
    carries rounding in the narrower dtype (a single folded member would
    be exact and hide it).
    """
    rng = np.random.default_rng(3)
    frames = [rng.uniform(0, 200, (8, 8)).astype(dtypes[n >= 5]) for n in range(7)]
    (result,) = run_units(
        recipe(),
        [ShotGroup(1, tuple(range(1, 8)))],
        lambda shot: frames[shot - 1],
        average_before_analysis=True,
    )
    assert result.measurement is None and "Incompatible raw dtypes" in result.error
    assert result.loaded_shots == tuple(range(1, 8)) and not result.load_failures
