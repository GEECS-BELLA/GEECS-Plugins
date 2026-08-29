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


class _FakeHeaderedDevice:
    """Minimal stand-in carrying a ``_column_headers`` map."""

    def __init__(self, name: str, column_headers: dict[str, str]) -> None:
        self.name = name
        self._column_headers = column_headers


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


def test_wrapper_uses_device_command_path_separately(tmp_path) -> None:
    """Native-save metadata path and device-visible path may differ."""
    det = _FakeSavingDetector("topcam")
    save_dir = tmp_path / "Scan007" / "UC_TopCam"
    device_path = r"Z:\data\Undulator\Y2026\06-Jun\26_0623\scans\Scan007\UC_TopCam"
    start = _run_capture_start(
        geecs_run_wrapper(
            _tiny_run(),
            scan_number=7,
            saving_detectors=[(det, str(save_dir), device_path)],
        )
    )

    assert det.localsavingpath.sets == [device_path]
    assert det.save.sets == ["on", "off"]
    assert save_dir.is_dir()
    assert start["nonscalar_save_paths"] == {"TOPCAM": str(save_dir)}


def test_wrapper_injects_merged_scalar_headers() -> None:
    dev_a = _FakeHeaderedDevice(
        "wavemeter", {"wavemeter-wavelength_nm": "UC_Wavemeter Wavelength (nm)"}
    )
    dev_b = _FakeHeaderedDevice(
        "jet_x", {"jet_x-position": "U_ESP_JetXYZ Position.Axis 1"}
    )
    start = _run_capture_start(
        geecs_run_wrapper(_tiny_run(), scan_number=3, devices=[dev_a, dev_b])
    )
    assert start["geecs_scalar_headers"] == {
        "wavemeter-wavelength_nm": "UC_Wavemeter Wavelength (nm)",
        "jet_x-position": "U_ESP_JetXYZ Position.Axis 1",
    }


def test_wrapper_omits_scalar_headers_when_no_devices() -> None:
    start = _run_capture_start(geecs_run_wrapper(_tiny_run(), scan_number=3))
    assert "geecs_scalar_headers" not in start


def test_wrapper_defer_save_on_leaves_enable_to_the_plan(tmp_path) -> None:
    """defer_save_on=True: no eager save-on; finalize save-off + md kept."""
    from geecs_bluesky.plans.run_wrapper import save_enable_plan

    det = _FakeSavingDetector("topcam")
    save_dir = tmp_path / "Scan007" / "UC_TopCam"

    def _inner_with_windowed_save():
        # The step plans yield save_enable_plan at the trigger-stopped point.
        yield from save_enable_plan([(det, str(save_dir))])
        yield from _tiny_run()

    start = _run_capture_start(
        geecs_run_wrapper(
            _inner_with_windowed_save(),
            scan_number=7,
            saving_detectors=[(det, str(save_dir))],
            defer_save_on=True,
        )
    )
    # Exactly one on (from the inner plan) and one off (finalize) — the
    # wrapper itself added no eager enable.
    assert det.save.sets == ["on", "off"]
    assert det.localsavingpath.sets == [str(save_dir)]
    assert save_dir.is_dir()
    # The save-path metadata is unaffected by the deferral.
    assert start["nonscalar_save_paths"] == {"TOPCAM": str(save_dir)}


def test_wrapper_defer_without_inner_enable_still_cleans_up(tmp_path) -> None:
    """Even if the inner plan never enabled saving, finalize still turns it off."""
    det = _FakeSavingDetector("topcam")
    save_dir = tmp_path / "Scan007" / "UC_TopCam"
    _run_capture_start(
        geecs_run_wrapper(
            _tiny_run(),
            scan_number=7,
            saving_detectors=[(det, str(save_dir))],
            defer_save_on=True,
        )
    )
    assert det.save.sets == ["off"]  # idempotent, harmless
    assert det.localsavingpath.sets == []


def test_save_enable_plan_alone(tmp_path) -> None:
    """The extracted stub creates the dir and writes path + save='on'."""
    from bluesky import RunEngine

    from geecs_bluesky.plans.run_wrapper import save_enable_plan

    det = _FakeSavingDetector("topcam")
    save_dir = tmp_path / "Scan008" / "UC_TopCam"
    RunEngine()(save_enable_plan([(det, str(save_dir))]))
    assert det.localsavingpath.sets == [str(save_dir)]
    assert det.save.sets == ["on"]
    assert save_dir.is_dir()


def test_wrapper_creates_capture_device_dirs_before_start_doc(tmp_path) -> None:
    """md capture_devices → engine-side mkdir, pre-start-doc, exist_ok."""
    (tmp_path / "UC_Existing").mkdir()
    seen_at_start: dict = {}

    def _spy(name, doc):
        if name == "start":
            # Dirs must already exist when the start doc is emitted (the
            # capture daemon's writers find them on the first frame).
            seen_at_start["dirs"] = [
                (tmp_path / d).is_dir() for d in ("UC_CamA", "UC_Existing")
            ]

    RE = RunEngine()
    RE.subscribe(_spy)
    RE(
        geecs_run_wrapper(
            _tiny_run(),
            experiment="Undulator",
            scan_number=9,
            scan_folder=str(tmp_path),
            extra_md={
                "capture_devices": ["UC_CamA", "UC_Existing"],
                "native_image_save": False,
            },
        )
    )
    assert seen_at_start["dirs"] == [True, True]
    assert (tmp_path / "UC_CamA").is_dir()


def test_wrapper_no_capture_dirs_without_scan_folder() -> None:
    """No scan folder claimed → no dir creation attempted (no crash)."""
    start = _run_capture_start(
        geecs_run_wrapper(
            _tiny_run(),
            experiment="Undulator",
            extra_md={"capture_devices": ["UC_CamA"]},
        )
    )
    assert start["capture_devices"] == ["UC_CamA"]


class _FakeCaptureOwnedDetector:
    """A save_control_only device: only the save control child exists."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._save_control_only = True
        self.save = _RecordingSignal(f"{name}-save")


def test_wrapper_commands_save_off_for_capture_owned_devices(tmp_path) -> None:
    """Toggle-off cameras get an eager save='off' write (codex #697 C1)."""
    cam = _FakeCaptureOwnedDetector("uc_cam")
    saver = _FakeSavingDetector("topcam")
    RE = RunEngine()
    RE(
        geecs_run_wrapper(
            _tiny_run(),
            experiment="Undulator",
            scan_number=11,
            scan_folder=str(tmp_path),
            saving_detectors=[(saver, str(tmp_path / "TOPCAM"))],
            devices=[cam, saver],
        )
    )
    assert cam.save.sets == ["off"]
    # The native saver's bracket is untouched: on at start, off at finalize.
    assert saver.save.sets == ["on", "off"]


def test_wrapper_no_off_write_without_marked_devices() -> None:
    """Plain devices never receive stray save writes."""
    plain = _FakeHeaderedDevice("plain", {"k": "v"})
    start = _run_capture_start(
        geecs_run_wrapper(_tiny_run(), experiment="Undulator", devices=[plain])
    )
    assert start["experiment"] == "Undulator"  # ran cleanly, nothing to assert off
