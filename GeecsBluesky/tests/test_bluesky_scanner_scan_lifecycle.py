"""Pinning tests for BlueskyScanner scan-lifecycle fixes (PR #449 review).

Finding A — a free-run reference (pacemaker) that fails to connect must not
let a non-Triggerable contributor silently inherit pacemaker duty: the next
synchronous device is promoted to the reference role, and with nothing to
promote the scanner aborts loudly.  Belt-and-suspenders: the free-run plan
itself rejects a non-Triggerable reference.

Finding B — a stop requested before the plan reaches the RunEngine aborts
the scan cleanly (no claim, no plan execution), and a timed-out scan-thread
join keeps the scanner reporting active instead of clearing the handle.

Finding C — early-exit paths validate *before* claiming the scan folder, so
they never leave a silently claimed ``ScanNNN/`` behind; optimization
settables join the cleanup list as soon as they connect, and post-claim
failures log the claimed-but-aborted state loudly.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner

# ---------------------------------------------------------------------------
# Fakes (no network / DB / CA)
# ---------------------------------------------------------------------------


class _FakeTriggerable:
    """Reference-capable fake — has a real ``trigger()`` like CaGenericDetector."""

    def __init__(self, name: str) -> None:
        self.name = name

    def trigger(self) -> None:
        raise NotImplementedError("not exercised by these tests")


class _FakeReadable:
    """Non-Triggerable fake — like CaTimestampedReadable / CaSnapshotReadable."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Records factory calls; raises for devices listed in *fail*."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = set(fail or ())
        self.calls: list[tuple[str, str]] = []
        self.settables: list[tuple[str, str]] = []
        self.scan_kwargs: dict | None = None
        self.optimize_kwargs: dict | None = None
        self.shot_control_called_with: object = "unset"

    def _maybe_fail(self, device: str) -> None:
        if device in self.fail:
            raise RuntimeError(f"connect failed: {device}")

    def detector(self, device, variables, *, save_images=False, name=None):
        self.calls.append(("detector", device))
        self._maybe_fail(device)
        return _FakeTriggerable(name or device)

    def contributor(self, device, variables, *, save_images=False, name=None):
        self.calls.append(("contributor", device))
        self._maybe_fail(device)
        return _FakeReadable(name or device)

    def snapshot(self, device, variables, *, name=None):
        self.calls.append(("snapshot", device))
        self._maybe_fail(device)
        return _FakeReadable(name or device)

    def settable(self, device, variable, *, name=None):
        self.settables.append((device, variable))
        self._maybe_fail(device)
        return _FakeReadable(name or f"{device}_{variable}")

    def shot_control(self, config):
        self.shot_control_called_with = config

    def scan(self, **kwargs):
        self.scan_kwargs = kwargs
        return "uid"

    def optimize(self, **kwargs):
        self.optimize_kwargs = kwargs
        return "uid", []


def _make_scanner(
    session: _FakeSession,
    devices_config: dict | None = None,
    mode: str = "free_run_time_sync",
) -> BlueskyScanner:
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._session = session
    scanner._experiment_dir = "TestExp"
    scanner._devices_config = devices_config or {}
    scanner._acquisition_mode = mode
    scanner._shot_control = None
    scanner._shots_per_step = 2
    scanner._detectors = []
    scanner._motor = None
    scanner._device_lock = threading.Lock()
    scanner._on_event = None
    scanner._current_state = None
    scanner._total_shots = 0
    scanner._completed_shots = 0
    scanner._abort_requested = False
    scanner._optimization_loader = None
    scanner._RE = SimpleNamespace(
        state="idle", abort=lambda reason=None: None, _loop=None
    )
    return scanner


def _noscan_config() -> SimpleNamespace:
    return SimpleNamespace(
        scan_mode="noscan",
        device_var=None,
        start=0.0,
        end=0.0,
        step=0.0,
        wait_time=1.0,
        additional_description="",
        background=False,
    )


def _patch_claim(monkeypatch, claims: list, result=(None, None)) -> None:
    """Record every claim attempt made through either claim entry point."""
    monkeypatch.setattr(
        bluesky_scanner,
        "claim_scan_number",
        lambda experiment: claims.append(experiment) or result,
    )
    monkeypatch.setattr(
        bluesky_scanner,
        "claim_scan",
        lambda experiment: claims.append(experiment) or result,
    )


# ---------------------------------------------------------------------------
# Finding A — reference connect failure must reclassify or abort loudly
# ---------------------------------------------------------------------------


def test_reference_connect_failure_promotes_next_sync_device(caplog) -> None:
    """The next synchronous device becomes the Triggerable pacemaker."""
    session = _FakeSession(fail={"U_RefCam"})
    scanner = _make_scanner(
        session,
        {
            "U_RefCam": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
            "U_Stage": {"synchronous": False, "variable_list": ["Pos"]},
        },
    )

    with caplog.at_level(logging.WARNING):
        detectors = scanner._build_session_devices()

    # Promoted device leads the list (session.scan uses detectors[0] as the
    # pacemaker) and was built Triggerable via the detector factory.
    assert detectors[0].name == "u_cam2"
    assert isinstance(detectors[0], _FakeTriggerable)
    assert ("detector", "U_Cam2") in session.calls
    assert ("contributor", "U_Cam2") not in session.calls
    assert "Promoting U_Cam2 to free-run reference" in caplog.text
    assert [d.name for d in detectors] == ["u_cam2", "u_stage"]


def test_reference_promotion_cascades_past_further_failures() -> None:
    """If the promoted device also fails, promotion moves to the next one."""
    session = _FakeSession(fail={"U_RefCam", "U_Cam2"})
    scanner = _make_scanner(
        session,
        {
            "U_RefCam": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
            "U_Cam3": {"synchronous": True, "variable_list": ["Val"]},
        },
    )

    detectors = scanner._build_session_devices()

    assert detectors[0].name == "u_cam3"
    assert isinstance(detectors[0], _FakeTriggerable)
    assert ("detector", "U_Cam3") in session.calls


def test_reference_failure_without_promotable_device_aborts_loudly() -> None:
    """No Triggerable pacemaker available -> raise instead of acquiring garbage."""
    session = _FakeSession(fail={"U_RefCam"})
    scanner = _make_scanner(
        session,
        {
            "U_RefCam": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Stage": {"synchronous": False, "variable_list": ["Pos"]},
        },
    )

    with pytest.raises(GeecsConfigurationError, match="pacemaker"):
        scanner._build_session_devices()

    # No acquisition was started, and the device that did connect is tracked
    # so the scan thread's cleanup disconnects it.
    assert session.scan_kwargs is None
    assert [d.name for d in scanner._detectors] == ["u_stage"]


def test_free_run_plan_rejects_non_triggerable_reference() -> None:
    """Belt-and-suspenders: the plan itself refuses a trigger-less pacemaker."""
    plan = geecs_free_run_step_scan(
        motor=None,
        positions=[None],
        reference=_FakeReadable("cam"),
        detectors=[],
        shots_per_step=1,
    )
    with pytest.raises(TypeError, match="Triggerable"):
        next(plan)


def test_free_run_plan_still_requires_shot_id_tracker() -> None:
    """The Triggerable guard precedes (not replaces) the shot-id guard."""
    plan = geecs_free_run_step_scan(
        motor=None,
        positions=[None],
        reference=_FakeTriggerable("ref"),
        detectors=[],
        shots_per_step=1,
    )
    with pytest.raises(ValueError, match="configure_shot_id"):
        next(plan)


# ---------------------------------------------------------------------------
# Finding B — stop before the RunEngine runs; timed-out thread join
# ---------------------------------------------------------------------------


def test_stop_during_device_connect_prevents_plan_execution(
    monkeypatch, caplog
) -> None:
    """A stop clicked while devices connect aborts before RE(plan) runs."""
    session = _FakeSession()
    scanner = _make_scanner(
        session, {"U_RefCam": {"synchronous": True, "variable_list": ["Sig"]}}
    )
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)

    original_detector = session.detector

    def detector_with_stop(device, variables, **kwargs):
        det = original_detector(device, variables, **kwargs)
        scanner._abort_requested = True  # the stop button, mid-connect
        return det

    session.detector = detector_with_stop
    claims: list = []
    _patch_claim(monkeypatch, claims)

    with caplog.at_level(logging.WARNING):
        scanner._run_scan(_noscan_config())

    assert session.scan_kwargs is None, "plan must never reach the RunEngine"
    assert claims == [], "no scan folder may be claimed for an aborted scan"
    assert scanner.current_state == "aborted"
    assert "stop requested before acquisition" in caplog.text


def test_timed_out_join_keeps_scanner_active(monkeypatch, caplog) -> None:
    """A join timeout must not clear the handle while the thread still runs."""
    scanner = _make_scanner(_FakeSession())
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    monkeypatch.setattr(bluesky_scanner, "_THREAD_JOIN_TIMEOUT", 0.05)
    scanner._RE = SimpleNamespace(state="running", abort=lambda reason=None: None)

    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True, name="stuck-scan")
    thread.start()
    scanner._scan_thread = thread
    try:
        with caplog.at_level(logging.ERROR):
            scanner.stop_scanning_thread()

        assert scanner._scan_thread is thread, "handle kept after timed-out join"
        assert scanner.is_scanning_active() is True
        assert "did not stop" in caplog.text
    finally:
        release.set()
        thread.join(timeout=5)

    # Once the thread actually exits, a subsequent stop clears the handle.
    scanner._RE = SimpleNamespace(state="idle", abort=lambda reason=None: None)
    scanner.stop_scanning_thread()
    assert scanner._scan_thread is None
    assert scanner.is_scanning_active() is False


# ---------------------------------------------------------------------------
# Finding C — validate before claiming; clean up on early exits
# ---------------------------------------------------------------------------


def test_statistics_with_no_detectors_claims_no_folder(monkeypatch, caplog) -> None:
    """The no-usable-detectors early exit must not leave a claimed folder."""
    session = _FakeSession()
    scanner = _make_scanner(session, devices_config={})
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(monkeypatch, claims)

    with caplog.at_level(logging.INFO):
        scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert claims == []
    assert session.scan_kwargs is None
    assert "nothing to collect" in caplog.text


def test_strict_mode_without_shot_control_raises_before_claim(monkeypatch) -> None:
    """The strict-mode raise path fires before any scan folder is claimed."""
    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {"U_RefCam": {"synchronous": True, "variable_list": ["Sig"]}},
        mode="strict_shot_control",
    )
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(monkeypatch, claims)

    with pytest.raises(GeecsConfigurationError, match="strict_shot_control"):
        scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert claims == []


def test_optimization_without_detectors_claims_nothing_and_no_settables(
    monkeypatch, caplog
) -> None:
    """Empty detector list exits before connecting settables or claiming."""
    session = _FakeSession()
    scanner = _make_scanner(session, devices_config={})
    scanner._optimization_loader = lambda path: SimpleNamespace(
        variable_names=["U_S1H:Current"]
    )
    claims: list = []
    _patch_claim(monkeypatch, claims)

    scan_config = SimpleNamespace(
        optimizer_config_path="/cfg/opt.yaml",
        start=0.0,
        end=1.0,
        step=1.0,
        wait_time=1.0,
        additional_description="",
    )
    with caplog.at_level(logging.ERROR):
        scanner._run_optimization(scan_config)

    assert session.settables == [], "no CA monitors may be opened on this path"
    assert claims == []
    assert session.optimize_kwargs is None
    assert "aborting before claiming" in caplog.text


def test_optimization_postclaim_failure_tracks_settables_and_logs(
    monkeypatch, caplog
) -> None:
    """A failure after the claim leaves settables cleanable and a loud record."""

    class _ExplodingBridge:
        variable_names = ["U_S1H:Current"]

        def bind(self, **kwargs):
            raise RuntimeError("bind exploded")

    session = _FakeSession()
    scanner = _make_scanner(
        session, {"U_RefCam": {"synchronous": True, "variable_list": ["Sig"]}}
    )
    scanner._optimization_loader = lambda path: _ExplodingBridge()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(
        monkeypatch, claims, result=(SimpleNamespace(number=7), "/tmp/Scan007")
    )

    scan_config = SimpleNamespace(
        optimizer_config_path="/cfg/opt.yaml",
        start=0.0,
        end=1.0,
        step=1.0,
        wait_time=1.0,
        additional_description="",
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="bind exploded"):
            scanner._run_optimization(scan_config)

    # The connected settable joined self._detectors before the failure, so
    # the scan thread's cleanup will disconnect its persistent CA monitor.
    tracked = [d.name for d in scanner._detectors]
    assert "u_s1h_current" in tracked
    # The claimed-but-aborted state is surfaced loudly, never silently.
    assert "claimed" in caplog.text
    assert "left in place" in caplog.text


# ---------------------------------------------------------------------------
# Empty variable_list — synchronous devices still build (image-only cameras)
# ---------------------------------------------------------------------------


def test_variableless_sync_reference_is_built_not_skipped() -> None:
    """A save-images-only camera is a valid free-run reference.

    acq_timestamp is always created as a dedicated child of the CA detector,
    so an empty variable_list must not disqualify a synchronous device —
    the legacy scanner force-appends acq_timestamp for the same reason.
    Regression: this config used to be skipped at DEBUG and then aborted
    with a misleading "failed to connect" error (live-check 2026-07-06,
    single healthy UC_Amp4_IR_input, laser-off).
    """
    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {
            "U_Cam": {
                "synchronous": True,
                "variable_list": [],
                "save_nonscalar_data": True,
            },
        },
    )

    detectors = scanner._build_session_devices()

    assert session.calls == [("detector", "U_Cam")]
    assert len(detectors) == 1
    assert isinstance(detectors[0], _FakeTriggerable)


def test_variableless_sync_contributor_is_built() -> None:
    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {
            "U_RefCam": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {
                "synchronous": True,
                "variable_list": [],
                "save_nonscalar_data": True,
            },
        },
    )

    detectors = scanner._build_session_devices()

    assert ("contributor", "U_Cam2") in session.calls
    assert len(detectors) == 2


def test_variableless_snapshot_is_skipped_with_warning(caplog) -> None:
    """An async device with no variables records nothing — skip loudly."""
    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {
            "U_RefCam": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Idle": {"synchronous": False, "variable_list": []},
        },
    )

    with caplog.at_level(logging.WARNING):
        detectors = scanner._build_session_devices()

    assert ("snapshot", "U_Idle") not in session.calls
    assert len(detectors) == 1
    assert any(
        "U_Idle" in r.message and "empty variable_list" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Analyzer-device auto-provisioning (legacy device_requirements parity)
# ---------------------------------------------------------------------------


class _FakeBridge:
    """Duck-typed optimization bridge (mirrors SessionOptimizationBridge)."""

    variable_names = ["U_S1H:Current"]
    on_finish = "hold"

    def __init__(self, device_requirements: dict | None = None) -> None:
        if device_requirements is not None:
            self.device_requirements = device_requirements
        self.bound_kwargs: dict | None = None

    def bind(self, **kwargs):
        self.bound_kwargs = kwargs
        return (lambda bin_data: 0.0), self


def _opt_scan_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer_config_path="/cfg/opt.yaml",
        start=0.0,
        end=1.0,
        step=1.0,
        wait_time=1.0,
        additional_description="",
    )


def _run_optimization_with_bridge(monkeypatch, scanner, bridge) -> None:
    scanner._optimization_loader = lambda path: bridge
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    _patch_claim(monkeypatch, [], result=(None, None))
    scanner._run_optimization(_opt_scan_config())


def test_optimization_auto_provisions_required_analyzer_device(
    monkeypatch, caplog
) -> None:
    """A required device absent from the GUI list is built as a sync detector."""
    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {"U_Ref": {"synchronous": True, "variable_list": ["Sig"]}},
        mode="strict_shot_control",
    )
    scanner._shot_control = object()  # strict mode only checks presence here
    bridge = _FakeBridge(
        {
            "Devices": {
                "UC_ObjCam": {
                    "add_all_variables": False,
                    "save_nonscalar_data": True,
                    "synchronous": True,
                    "variable_list": ["acq_timestamp"],
                }
            }
        }
    )

    with caplog.at_level(logging.INFO):
        _run_optimization_with_bridge(monkeypatch, scanner, bridge)

    # Merged before _build_session_devices: synchronous → detector factory.
    assert ("detector", "UC_ObjCam") in session.calls
    cfg = scanner._devices_config["UC_ObjCam"]
    assert cfg["synchronous"] is True
    assert cfg["save_nonscalar_data"] is True
    assert cfg["variable_list"] == ["acq_timestamp"]
    detector_names = [d.name for d in session.optimize_kwargs["detectors"]]
    assert "uc_objcam" in detector_names
    assert "auto-provisioned" in caplog.text


def test_optimization_merges_required_variables_into_gui_device(
    monkeypatch, caplog
) -> None:
    """An overlapping device keeps its GUI config and gains missing variables."""
    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {
            "UC_ObjCam": {
                "synchronous": True,
                "save_nonscalar_data": False,
                "variable_list": ["MeanCounts"],
            }
        },
        mode="strict_shot_control",
    )
    scanner._shot_control = object()
    bridge = _FakeBridge(
        {
            "Devices": {
                "UC_ObjCam": {
                    "synchronous": True,
                    "save_nonscalar_data": True,
                    "variable_list": ["MeanCounts", "acq_timestamp"],
                }
            }
        }
    )

    with caplog.at_level(logging.INFO):
        _run_optimization_with_bridge(monkeypatch, scanner, bridge)

    cfg = scanner._devices_config["UC_ObjCam"]
    # Union of variables; GUI settings (save flag) preserved.
    assert cfg["variable_list"] == ["MeanCounts", "acq_timestamp"]
    assert cfg["save_nonscalar_data"] is False
    assert session.calls.count(("detector", "UC_ObjCam")) == 1
    assert "GUI settings preserved" in caplog.text


def test_optimization_bridge_without_requirements_is_unchanged(monkeypatch) -> None:
    """A bridge without a device_requirements attribute changes nothing."""
    session = _FakeSession()
    gui_devices = {"U_Ref": {"synchronous": True, "variable_list": ["Sig"]}}
    scanner = _make_scanner(
        session,
        {name: dict(cfg) for name, cfg in gui_devices.items()},
        mode="strict_shot_control",
    )
    scanner._shot_control = object()
    bridge = _FakeBridge()  # no device_requirements attribute at all

    _run_optimization_with_bridge(monkeypatch, scanner, bridge)

    assert scanner._devices_config == gui_devices
    assert [c for c in session.calls if c[0] == "detector"] == [("detector", "U_Ref")]


def test_reference_abort_message_names_the_failing_devices() -> None:
    """The loud abort says WHY each device failed, not just that it did."""
    session = _FakeSession(fail={"U_RefCam"})
    scanner = _make_scanner(
        session,
        {"U_RefCam": {"synchronous": True, "variable_list": ["Sig"]}},
    )

    with pytest.raises(GeecsConfigurationError) as excinfo:
        scanner._build_session_devices()

    assert "U_RefCam" in str(excinfo.value)
    assert "connect failed" in str(excinfo.value)
