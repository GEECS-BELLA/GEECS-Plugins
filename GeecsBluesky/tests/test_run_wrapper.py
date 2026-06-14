"""Tests for geecs_run_wrapper — run-bookkeeping (metadata + native saving)."""

from __future__ import annotations

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import RunEngine
from ophyd_async.core import AsyncStatus

from geecs_bluesky.plans.run_wrapper import geecs_run_wrapper


def _tiny_run():
    """A minimal plan that opens and closes one run with intrinsic md."""

    @bpp.run_decorator(md={"plan_name": "tiny", "geecs_event_schema": 1})
    def inner():
        yield from bps.null()

    yield from inner()


def _run_capture_start(plan) -> dict:
    start = {}
    RE = RunEngine()
    RE.subscribe(lambda name, doc: start.update(doc) if name == "start" else None)
    RE(plan)
    return start


class _RecordingSignal:
    """Minimal Movable that records the values set on it."""

    parent = None  # bps.mv inspects .parent for coupled-device handling

    def __init__(self, name: str) -> None:
        self.name = name
        self.sets: list = []

    def set(self, value) -> AsyncStatus:
        self.sets.append(value)

        async def _noop() -> None:
            pass

        return AsyncStatus(_noop())


class _FakeSavingDetector:
    def __init__(self, name: str) -> None:
        self.name = name
        self._geecs_device_name = name.upper()
        self.localsavingpath = _RecordingSignal(f"{name}-localsavingpath")
        self.save = _RecordingSignal(f"{name}-save")


def test_wrapper_injects_scan_metadata_and_scan_id() -> None:
    start = _run_capture_start(
        geecs_run_wrapper(
            _tiny_run(),
            experiment="Undulator",
            scan_number=7,
            scan_folder="/data/Scan007",
            extra_md={"operator": "skb", "scan_mode": "noscan"},
        )
    )
    assert start["scan_number"] == 7
    assert start["scan_id"] == 7  # Bluesky display field = GEECS scan number
    assert start["scan_folder"] == "/data/Scan007"
    assert start["experiment"] == "Undulator"
    assert start["operator"] == "skb"
    assert start["scan_mode"] == "noscan"
    # plan-intrinsic md survives
    assert start["plan_name"] == "tiny"
    assert start["geecs_event_schema"] == 1


def test_wrapper_without_scan_number_omits_it() -> None:
    start = _run_capture_start(geecs_run_wrapper(_tiny_run(), experiment="Undulator"))
    assert "scan_number" not in start
    assert start["experiment"] == "Undulator"
    # scan_id is still present (the RunEngine's own counter) — we just don't
    # override it to a GEECS scan number when none was claimed.


def test_wrapper_brackets_native_saving(tmp_path) -> None:
    det = _FakeSavingDetector("topcam")
    save_dir = tmp_path / "Scan007" / "UC_TopCam"
    start = _run_capture_start(
        geecs_run_wrapper(
            _tiny_run(),
            scan_number=7,
            saving_detectors=[(det, str(save_dir))],
        )
    )
    # save path set on, then off (finalize) — and the dir was created
    assert det.localsavingpath.sets == [str(save_dir)]
    assert det.save.sets == ["on", "off"]
    assert save_dir.is_dir()
    assert start["nonscalar_save_paths"] == {"TOPCAM": str(save_dir)}
