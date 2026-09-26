"""Pooled execution yields the serial sequence, opens the loader once per worker, and logs through."""

from __future__ import annotations

import importlib
import logging
import os
import textwrap

import numpy as np
import pytest
from geecs_schemas.analysis import AnalysisDiagnostic

from geecs_analysis.compat.v2 import compile_v2
from geecs_analysis.compat.v2_run import ShotGroup, run_units

# A spawned worker imports the loader's module by name, so the loader lives
# in a real module on sys.path (the test module itself is not importable
# under pytest's importlib mode). The parent's sys.path travels to spawned
# children.
HELPER = textwrap.dedent(
    '''
    """A picklable shot source that records each process's open/close."""
    import logging
    import os
    from pathlib import Path

    import numpy as np

    log = logging.getLogger("pool_helper")


    class Source:
        def __init__(self, shape, record_dir):
            self.shape = shape
            self.record_dir = str(record_dir)
            self.opened = False

        def _record(self, event):
            with open(Path(self.record_dir) / str(os.getpid()), "a") as f:
                f.write(event + "\\n")

        def __enter__(self):
            self._record("open")
            self.opened = True
            return self

        def __exit__(self, *exc):
            self._record("close")
            self.opened = False
            return False

        def __call__(self, shot):
            assert self.opened, "read before the source was entered"
            if shot == 5:
                raise OSError(f"missing {shot}")
            if shot == 3:
                log.warning("shot %d looks odd", shot)
            return np.full(self.shape, shot, dtype=np.uint16)
    '''
)


@pytest.fixture
def source(tmp_path, monkeypatch):
    (tmp_path / "pool_helper.py").write_text(HELPER)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module("pool_helper")
    records = tmp_path / "records"
    records.mkdir()
    yield module.Source((4, 6), records), records
    monkeypatch.delitem(importlib.sys.modules, "pool_helper", raising=False)


def recipe():
    return compile_v2(
        AnalysisDiagnostic.model_validate(
            {
                "name": "Camera",
                "analyzer": {"kind": "beam"},
                "image": {
                    "type": "camera",
                    "pipeline": ["roi"],
                    "roi": {"x_min": 1, "x_max": 5, "y_min": 0, "y_max": 4},
                },
            }
        )
    )


def events(records):
    return {p.name: p.read_text().split() for p in records.iterdir()}


def same_scalars(a, b):
    assert list(a) == list(b)
    for key in a:
        assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key])), key


def same_outcomes(pooled, serial):
    assert len(pooled) == len(serial)
    for a, b in zip(pooled, serial, strict=True):
        assert (a.group, a.loaded_shots, a.load_failures, a.error) == (
            b.group,
            b.loaded_shots,
            b.load_failures,
            b.error,
        )
        assert (a.measurement is None) == (b.measurement is None)
        if a.measurement is not None:
            same_scalars(a.measurement.scalars, b.measurement.scalars)
            np.testing.assert_array_equal(
                a.measurement.frame.data, b.measurement.frame.data
            )
            assert a.measurement.frame.shot == b.measurement.frame.shot
            assert a.measurement.notes == b.measurement.notes
            assert [o.id for o in a.measurement.overlays] == [
                o.id for o in b.measurement.overlays
            ]


@pytest.mark.parametrize("average", [False, True])
def test_pool_yields_the_serial_sequence_and_opens_each_worker_once(source, average):
    loader, records = source
    compiled = recipe()
    if average:
        groups = [ShotGroup(k, (2 * k - 1, 2 * k)) for k in range(1, 8)]
    else:
        groups = [ShotGroup(n, (n,)) for n in range(1, 13)]
    serial = list(run_units(compiled, groups, loader, average_before_analysis=average))
    assert events(records) == {str(os.getpid()): ["open", "close"]}
    for path in records.iterdir():
        path.unlink()
    pooled = list(
        run_units(compiled, groups, loader, average_before_analysis=average, workers=2)
    )
    same_outcomes(pooled, serial)
    assert [o.group.key for o in pooled] == [g.key for g in groups]
    # The failed shot is an outcome in both modes, never an exception.
    failed = [o for o in pooled if o.load_failures]
    assert [f.shot for o in failed for f in o.load_failures] == [5]
    workers = events(records)
    assert str(os.getpid()) not in workers, "the parent read nothing itself"
    assert 1 <= len(workers) <= 2
    assert all(seen == ["open", "close"] for seen in workers.values())


def test_worker_log_records_reach_the_parent_logger(source, caplog):
    loader, _ = source
    groups = [ShotGroup(n, (n,)) for n in range(1, 5)]
    with caplog.at_level(logging.WARNING, logger="pool_helper"):
        outcomes = list(run_units(recipe(), groups, loader, workers=2))
    assert len(outcomes) == 4
    records = [r for r in caplog.records if r.name == "pool_helper"]
    assert [r.getMessage() for r in records] == ["shot 3 looks odd"]
    assert records[0].levelno == logging.WARNING
    assert records[0].process != os.getpid()


def test_closing_a_pooled_run_early_shuts_the_pool_down(source):
    loader, records = source
    groups = (ShotGroup(n, (n,)) for n in range(1, 41))
    outcomes = run_units(recipe(), groups, loader, workers=2)
    first = next(outcomes)
    assert first.group.key == 1
    outcomes.close()
    # Every worker that opened the source also closed it: the pool exited
    # through its atexit handlers, and the untouched groups were dropped.
    seen = events(records)
    assert seen and all(value == ["open", "close"] for value in seen.values())
    assert not any(records.parent.glob("dummy"))


def test_pool_refuses_multi_member_groups_in_per_shot_mode(source):
    loader, records = source
    with pytest.raises(ValueError, match="single-member"):
        list(run_units(recipe(), [ShotGroup(1, (1, 2))], loader, workers=2))
    assert all(value == ["open", "close"] for value in events(records).values())


def test_serial_default_creates_no_pool(monkeypatch):
    import concurrent.futures

    def no_pool(*args, **kwargs):
        raise AssertionError("workers=1 must not build a pool")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", no_pool)
    arrays = {1: np.ones((3, 3)), 2: np.ones((3, 3))}
    outcomes = list(
        run_units(
            recipe(), [ShotGroup(1, (1,)), ShotGroup(2, (2,))], arrays.__getitem__
        )
    )
    assert [o.group.key for o in outcomes] == [1, 2]
