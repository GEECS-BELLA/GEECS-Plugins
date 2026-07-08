"""Tests for BlueskyScanner GUI progress events and the pre-flight checks.

Item 1 — ``ScanStepEvent`` emission: every Bluesky event document emits a
shot-level progress event through ``on_event`` (same channel and thread
semantics as the lifecycle events), clamped at ``total_shots`` so the free-run
tail-flush overcount stays cosmetic.

Item 2 — pre-flight liveness (both modes) + free-run staleness: before the
scan folder is claimed, every synchronous device's gateway ``CONNECTED`` PV
(``connected_status``) is read — a device reporting Disconnected raises an
operator dialog through the legacy ``ScanDialogEvent``/``DialogRequest``
channel (drop-and-continue vs abort; abort-only for a disconnected free-run
reference; fail-open when the PV is unreadable).  In free-run mode a second
stage checks ``acq_timestamp`` freshness for the trigger-must-be-free-running
requirement: all CONNECTED but all stale → the trigger-off dialog; the
residual CONNECTED-but-stale contributor keeps the drop dialog.  Strict runs
the liveness stage only (frames are not needed pre-scan).  Headless /
unanswered → today's proceed-and-fail-loudly default.

See ``Planning/gui_stewardship/00_overview.md`` §4–5.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from geecs_bluesky import preflight
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner

# ---------------------------------------------------------------------------
# Fakes (no network / DB / CA / Qt)
# ---------------------------------------------------------------------------


@dataclass
class _FakeStepEvent:
    """Stands in for geecs_scanner.engine ScanStepEvent."""

    step_index: int = 0
    total_steps: int = 0
    shots_completed: int = 0
    phase: str = "started"


@dataclass
class _FakeDialogEvent:
    """Stands in for geecs_scanner.engine ScanDialogEvent."""

    request: object = None


@dataclass
class _FakeDialogRequest:
    """Stands in for geecs_scanner.engine.dialog_request.DialogRequest."""

    exc: Exception
    context: str | None = None
    title: str | None = None
    continue_label: str | None = None
    abort_label: str | None = None
    response_event: threading.Event = field(default_factory=threading.Event)
    abort: list[bool] = field(default_factory=lambda: [False])


class _FakeConnectedSignal:
    """Stands in for the ``connected_status`` ``epics_signal_r`` (str read).

    ``value`` may be an Exception instance to simulate an unreadable
    ``CONNECTED`` PV (old gateway without status PVs → fail-open).
    """

    def __init__(self, value: str | Exception = "Connected") -> None:
        self.value = value

    async def get_value(self) -> str:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class _FakeSyncDevice:
    """Sync device fake: ``_last_acq`` cache + gateway ``connected_status``."""

    def __init__(
        self,
        device: str,
        name: str,
        last_acq: float | None,
        connected: str | Exception = "Connected",
    ) -> None:
        self.name = name  # ophyd (safe) name
        self._geecs_device_name = device
        self._last_acq = last_acq
        self.connected_status = _FakeConnectedSignal(connected)

    def trigger(self) -> None:  # reference-capable, like CaGenericDetector
        raise NotImplementedError("not exercised by these tests")


class _FakeSnapshotDevice:
    """Async snapshot fake: deliberately has NO ``_last_acq`` attribute."""

    def __init__(self, device: str, name: str) -> None:
        self.name = name
        self._geecs_device_name = device


def _labview_now() -> float:
    return time.time() + bluesky_scanner._LABVIEW_EPOCH_OFFSET


def _fresh() -> float:
    return _labview_now() - 1.0


def _stale() -> float:
    return _labview_now() - 60.0


class _FakeSession:
    """Session fake whose factories attach ``_last_acq`` + ``connected_status``.

    ``connected`` maps device name → ``CONNECTED`` reading: ``"Connected"``
    (default), ``"Disconnected"``, or an Exception for an unreadable PV.
    """

    def __init__(
        self,
        last_acq: dict[str, float | None] | None = None,
        connected: dict[str, str | Exception] | None = None,
    ) -> None:
        self._last_acq = last_acq or {}
        self._connected = connected or {}
        self.scan_kwargs: dict | None = None

    def _cache_for(self, device: str) -> float | None:
        if device in self._last_acq:
            return self._last_acq[device]
        return _fresh()

    def _connected_for(self, device: str) -> str | Exception:
        return self._connected.get(device, "Connected")

    def detector(self, device, variables, *, save_images=False, name=None):
        return _FakeSyncDevice(
            device, name or device, self._cache_for(device), self._connected_for(device)
        )

    def contributor(self, device, variables, *, save_images=False, name=None):
        return _FakeSyncDevice(
            device, name or device, self._cache_for(device), self._connected_for(device)
        )

    def snapshot(self, device, variables, *, name=None):
        return _FakeSnapshotDevice(device, name or device)

    def shot_control(self, config):
        pass

    def scan(self, **kwargs):
        self.scan_kwargs = kwargs
        return "uid"


# The liveness read dispatches the signal coroutine to the RE loop via
# run_coroutine_threadsafe (the scanner's connect/disconnect pattern), so the
# fake RE needs a real event loop running in a background thread.
_LOOP: asyncio.AbstractEventLoop | None = None


def _bg_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    if _LOOP is None:
        _LOOP = asyncio.new_event_loop()
        threading.Thread(target=_LOOP.run_forever, daemon=True).start()
    return _LOOP


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
    scanner._total_steps = 0
    scanner._completed_shots = 0
    scanner._abort_requested = False
    scanner._optimization_loader = None
    scanner._RE = SimpleNamespace(
        state="idle", abort=lambda reason=None: None, _loop=_bg_loop()
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


def _patch_dialog_channel(monkeypatch) -> None:
    """Route dialog emission through the fakes (env-independent tests)."""
    monkeypatch.setattr(bluesky_scanner, "ScanDialogEvent", _FakeDialogEvent)
    monkeypatch.setattr(bluesky_scanner, "DialogRequest", _FakeDialogRequest)
    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.0)


def _patch_claim(monkeypatch, claims: list, result=(None, None)) -> None:
    monkeypatch.setattr(
        bluesky_scanner,
        "claim_scan_number",
        lambda experiment: claims.append(experiment) or result,
    )


def _dialog_consumer(events: list, claims: list, answers: dict, *, abort: bool | None):
    """Return an on_event callback answering dialog requests inline.

    ``abort=None`` leaves the request unanswered (timeout path).  Records the
    claim count observed at dialog time so tests can assert nothing was
    claimed before the operator was asked.
    """

    def on_event(event) -> None:
        events.append(event)
        if isinstance(event, _FakeDialogEvent):
            answers["claims_at_dialog"] = list(claims)
            answers["request"] = event.request
            if abort is not None:
                event.request.abort[0] = abort
                event.request.response_event.set()

    return on_event


# ---------------------------------------------------------------------------
# Item 1 — step/progress events from event documents
# ---------------------------------------------------------------------------


def test_event_documents_emit_step_progress(monkeypatch) -> None:
    """One ScanStepEvent per event document, with running shot counts."""
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", _FakeStepEvent)
    events: list = []
    scanner = _make_scanner(_FakeSession())
    scanner._on_event = events.append
    scanner._total_shots = 4
    scanner._total_steps = 2

    scanner._on_document("start", {"uid": "abc"})
    for bin_number in (1, 1, 2, 2):
        scanner._on_document("event", {"data": {"bin_number": bin_number}})

    assert scanner._last_run_uid == "abc"
    step_events = [e for e in events if isinstance(e, _FakeStepEvent)]
    assert [e.shots_completed for e in step_events] == [1, 2, 3, 4]
    assert [e.step_index for e in step_events] == [0, 0, 1, 1]
    assert all(e.total_steps == 2 for e in step_events)
    assert all(e.phase == "completed" for e in step_events)


def test_progress_clamps_at_total_shots_for_tail_flush(monkeypatch) -> None:
    """The free-run tail-flush extra event document must not exceed 100 %."""
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", _FakeStepEvent)
    events: list = []
    scanner = _make_scanner(_FakeSession())
    scanner._on_event = events.append
    scanner._total_shots = 2
    scanner._total_steps = 1

    scanner._on_document("event", {"data": {"bin_number": 1}})
    scanner._on_document("event", {"data": {"bin_number": 1}})
    # Tail flush: one extra event on the flush stream (no bin_number here).
    scanner._on_document("event", {"data": {}})

    assert [e.shots_completed for e in events] == [1, 2, 2]
    assert events[-1].step_index == 0  # clamped inside [0, total_steps)
    # The raw counter still overcounts (known, cosmetic) but the estimate
    # exposed to pollers is clamped too.
    assert scanner._completed_shots == 3
    assert scanner.estimate_current_completion() == 1.0


def test_progress_without_callback_or_event_type_is_silent(monkeypatch) -> None:
    """No consumer or no geecs_scanner installed → counting still works."""
    scanner = _make_scanner(_FakeSession())
    scanner._total_shots = 2
    scanner._on_document("event", {"data": {"bin_number": 1}})  # _on_event None

    events: list = []
    scanner._on_event = events.append
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", None)
    scanner._on_document("event", {"data": {"bin_number": 1}})

    assert events == []
    assert scanner._completed_shots == 2


def test_progress_survives_raising_callback(monkeypatch) -> None:
    """A misbehaving on_event consumer must not break document handling."""
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", _FakeStepEvent)

    def bad_callback(event) -> None:
        raise RuntimeError("GUI went away")

    scanner = _make_scanner(_FakeSession())
    scanner._on_event = bad_callback
    scanner._total_shots = 2
    scanner._on_document("event", {"data": {"bin_number": 1}})
    assert scanner._completed_shots == 1


# ---------------------------------------------------------------------------
# Item 2 — pre-flight liveness: connected + fresh devices are untouched
# ---------------------------------------------------------------------------


def test_fresh_devices_no_dialog_no_behavior_change(monkeypatch) -> None:
    """All sync devices CONNECTED and fresh → no dialog, every device kept."""
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession()  # every device defaults to Connected + fresh
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
            "U_Stage": {"synchronous": False, "variable_list": ["Pos"]},
        },
    )
    events: list = []
    scanner._on_event = events.append

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Ref",
        "U_Cam2",
        "U_Stage",
    ]
    assert claims == ["TestExp"]


# ---------------------------------------------------------------------------
# Item 2 — liveness: a DISCONNECTED device raises the dialog in either mode
# ---------------------------------------------------------------------------


def test_disconnected_contributor_free_run_drop_removes_and_disconnects(
    monkeypatch,
) -> None:
    """Free-run + gateway says a contributor is down → drop dialog, pre-claim.

    The device's frames may even be recent (a just-crashed device) — the
    CONNECTED PV alone is authoritative, so the dialog fires regardless of
    the staleness heuristic.
    """
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(connected={"U_Cam2": "Disconnected"})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
            "U_Stage": {"synchronous": False, "variable_list": ["Pos"]},
        },
    )
    disconnected: list[str] = []
    scanner._disconnect_device = lambda dev: disconnected.append(dev._geecs_device_name)
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=False)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    # Nothing was claimed before the operator was asked.
    assert answers["claims_at_dialog"] == []
    request = answers["request"]
    assert "U_Cam2" in str(request.exc)
    assert "DISCONNECTED" in str(request.exc)  # liveness, not staleness, wording
    assert "stale" not in str(request.exc).lower()
    assert "drop" in request.continue_label.lower()
    # The dead device was disconnected and removed; the scan proceeded.
    assert disconnected == ["U_Cam2"]
    assert session.scan_kwargs is not None
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Ref",
        "U_Stage",
    ]
    assert [d._geecs_device_name for d in scanner._detectors] == ["U_Ref", "U_Stage"]
    assert claims == ["TestExp"]


def test_disconnected_device_strict_drop_works(monkeypatch) -> None:
    """Strict + gateway says a device is down → same drop dialog, pre-claim.

    Live motivation (2026-07-07, Scan006): an OFF camera's data PVs still
    CA-connected fine (the gateway serves everything in the DB), so the scan
    ran into 3 wasted refires and a post-claim abort.  The CONNECTED PV now
    catches it before the claim, mode-independently.
    """
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(connected={"U_Cam2": "Disconnected"})
    scanner = _make_scanner(
        session,
        {
            "U_Cam1": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
        mode="strict_shot_control",
    )
    scanner._shot_control = object()
    answers: dict = {}
    events: list = []
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=False)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert answers["claims_at_dialog"] == []  # asked before any claim
    assert "DISCONNECTED" in str(answers["request"].exc)
    assert session.scan_kwargs is not None
    names = [d._geecs_device_name for d in session.scan_kwargs["detectors"]]
    assert names == ["U_Cam1"]  # dead device dropped


def test_disconnected_device_strict_abort_is_pre_claim(monkeypatch) -> None:
    """Strict + disconnected device + operator abort → nothing claimed."""
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(connected={"U_Cam2": "Disconnected"})
    scanner = _make_scanner(
        session,
        {
            "U_Cam1": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
        mode="strict_shot_control",
    )
    scanner._shot_control = object()
    answers: dict = {}
    events: list = []
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=True)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert claims == []
    assert session.scan_kwargs is None
    assert scanner._abort_requested


def test_disconnected_reference_free_run_is_abort_only(monkeypatch) -> None:
    """A DISCONNECTED pacemaker offers abort vs clearly-labeled try-anyway."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(connected={"U_Ref": "Disconnected"})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=True)

    scanner._run_scan(_noscan_config())

    request = answers["request"]
    assert "reference" in str(request.exc).lower()
    assert "U_Ref" in str(request.exc)
    assert "DISCONNECTED" in str(request.exc)
    # No drop offer — the second option is a clearly-labeled try-anyway.
    assert "anyway" in request.continue_label.lower()
    assert claims == []
    assert scanner.current_state == "aborted"


# ---------------------------------------------------------------------------
# Item 2 — liveness fail-open: an unreadable CONNECTED PV never blocks a scan
# ---------------------------------------------------------------------------


def test_connected_unreadable_fails_open(monkeypatch) -> None:
    """CONNECTED read raising (old gateway) → no dialog, no crash, proceed."""
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(
        connected={
            "U_Ref": RuntimeError("no CONNECTED PV on this gateway"),
            "U_Cam2": RuntimeError("no CONNECTED PV on this gateway"),
        }
    )
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    scanner._on_event = events.append

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None
    assert len(session.scan_kwargs["detectors"]) == 2
    assert claims == ["TestExp"]


def test_connected_read_timeout_fails_open(monkeypatch) -> None:
    """A hanging CONNECTED read (dead IOC link) times out and reads as live."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "_LIVENESS_READ_TIMEOUT_S", 0.05)
    claims: list = []
    _patch_claim(monkeypatch, claims)

    class _HangingSignal:
        async def get_value(self) -> str:
            await asyncio.sleep(30.0)
            return "Connected"

    session = _FakeSession()
    scanner = _make_scanner(
        session,
        {"U_Ref": {"synchronous": True, "variable_list": ["Sig"]}},
    )
    events: list = []
    scanner._on_event = events.append

    original_detector = session.detector

    def hanging_detector(device, variables, **kwargs):
        det = original_detector(device, variables, **kwargs)
        det.connected_status = _HangingSignal()
        return det

    session.detector = hanging_detector

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None
    assert claims == ["TestExp"]


# ---------------------------------------------------------------------------
# Item 2 — strict mode is liveness-only: staleness never dialogs there
# ---------------------------------------------------------------------------


def test_strict_all_stale_proceeds_silently(monkeypatch) -> None:
    """Strict + all CONNECTED + ALL frames stale → no dialog, proceed.

    Frames are not needed before a strict scan: the trigger legitimately
    sits OFF (ARMED starts it), leaving every cache stale.  CONNECTED said
    the devices are live, and that is the whole strict pre-flight.
    """
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Cam": None})
    scanner = _make_scanner(
        session,
        {"U_Cam": {"synchronous": True, "variable_list": ["Sig"]}},
        mode="strict_shot_control",
    )
    scanner._shot_control = object()
    events: list = []
    scanner._on_event = events.append

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None


def test_strict_differential_stale_proceeds_silently(monkeypatch) -> None:
    """Strict + CONNECTED devices with mixed frame ages → no dialog.

    The previous differential-staleness heuristic is gone: CONNECTED is the
    authoritative liveness signal, so a connected-but-frameless device before
    a strict scan (trigger off, camera armed late, …) is not diagnosable as
    dead and must not dialog.
    """
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Cam2": _stale()})
    scanner = _make_scanner(
        session,
        {
            "U_Cam1": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
        mode="strict_shot_control",
    )
    scanner._shot_control = object()
    events: list = []
    scanner._on_event = events.append

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None
    names = [d._geecs_device_name for d in session.scan_kwargs["detectors"]]
    assert names == ["U_Cam1", "U_Cam2"]  # nothing dropped
    assert claims == ["TestExp"]


# ---------------------------------------------------------------------------
# Item 2 — free-run residual case: CONNECTED-but-stale contributor
# ---------------------------------------------------------------------------


def test_stale_connected_contributor_keeps_drop_dialog(monkeypatch) -> None:
    """CONNECTED contributor with no frames + fresh reference → drop dialog.

    The residual staleness case: the fresh reference proves the trigger is
    running, so a frameless-but-CONNECTED contributor is a per-device
    acquisition problem (camera acquisition stopped while its TCP stream is
    up) — the drop-or-abort dialog is kept for it, with not-acquiring (not
    dead-device) wording.
    """
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Cam2": _stale()})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
            "U_Stage": {"synchronous": False, "variable_list": ["Pos"]},
        },
    )
    disconnected: list[str] = []
    scanner._disconnect_device = lambda dev: disconnected.append(dev._geecs_device_name)
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=False)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    # Nothing was claimed before the operator was asked.
    assert answers["claims_at_dialog"] == []
    request = answers["request"]
    assert "U_Cam2" in str(request.exc)
    assert "s ago" in str(request.exc)  # says how stale
    assert "CONNECTED" in str(request.exc)  # liveness already vouched for it
    assert request.continue_label is not None
    assert "drop" in request.continue_label.lower()
    # The stale device was disconnected and removed; the scan proceeded.
    assert disconnected == ["U_Cam2"]
    assert session.scan_kwargs is not None
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Ref",
        "U_Stage",
    ]
    assert [d._geecs_device_name for d in scanner._detectors] == ["U_Ref", "U_Stage"]
    assert claims == ["TestExp"]


def test_stale_contributor_abort_is_pre_claim(monkeypatch) -> None:
    """Operator answers abort → clean pre-claim abort, devices disconnected."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Cam2": None})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    disconnected: list[str] = []
    scanner._disconnect_device = lambda dev: disconnected.append(dev._geecs_device_name)
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=True)

    scanner._run_scan(_noscan_config())

    assert claims == [], "no scan folder may be claimed for an aborted scan"
    assert session.scan_kwargs is None
    assert scanner.current_state == "aborted"
    # The scan thread's cleanup disconnected everything that had connected.
    assert set(disconnected) == {"U_Ref", "U_Cam2"}


# ---------------------------------------------------------------------------
# Item 2 — CONNECTED-but-stale reference: abort-only v1
# ---------------------------------------------------------------------------


def test_stale_reference_dialog_is_abort_only_v1(monkeypatch) -> None:
    """A CONNECTED but frameless pacemaker offers abort vs labeled retry."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Ref": _stale()})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=True)

    scanner._run_scan(_noscan_config())

    request = answers["request"]
    assert "reference" in str(request.exc).lower()
    assert "U_Ref" in str(request.exc)
    # No drop offer — the second option is a clearly-labeled try-anyway.
    assert "anyway" in request.continue_label.lower()
    assert claims == []
    assert scanner.current_state == "aborted"


def test_stale_reference_try_anyway_proceeds_with_full_list(monkeypatch) -> None:
    """Choosing try-anyway preserves today's behavior (t0 sync fails loudly)."""
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Ref": _stale()})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=False)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert session.scan_kwargs is not None
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Ref",
        "U_Cam2",
    ]


# ---------------------------------------------------------------------------
# Item 2 — all sync devices stale: the trigger is probably off
# ---------------------------------------------------------------------------


def test_all_stale_dialog_blames_the_trigger(monkeypatch) -> None:
    """All CONNECTED but no fresh frames anywhere → trigger-off wording.

    Now unambiguous: the liveness stage already confirmed every device's TCP
    stream is up, so a total absence of frames can only be the trigger.
    """
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Ref": None, "U_Cam2": _stale()})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=True)

    scanner._run_scan(_noscan_config())

    request = answers["request"]
    assert "trigger appears to be off" in str(request.exc)
    assert "CONNECTED" in str(request.exc)
    assert claims == []
    assert scanner.current_state == "aborted"


# ---------------------------------------------------------------------------
# Item 2 — headless / no answer: today's behavior is preserved
# ---------------------------------------------------------------------------


def test_headless_disconnected_device_proceeds_unchanged(monkeypatch) -> None:
    """on_event=None → no dialog machinery, scan proceeds with all devices.

    Even a device the gateway reports DISCONNECTED must not block a headless
    scan — it proceeds and fails loudly downstream (t0 sync / device-down
    abort in the plan).
    """
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(connected={"U_Cam2": "Disconnected"})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    scanner._on_event = None

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert session.scan_kwargs is not None
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Ref",
        "U_Cam2",
    ]
    assert claims == ["TestExp"]


def test_unanswered_dialog_times_out_and_proceeds(monkeypatch) -> None:
    """No response within the (shortened) timeout → proceed unchanged."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "_PREFLIGHT_DIALOG_TIMEOUT_S", 0.05)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(connected={"U_Cam2": "Disconnected"})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, claims, answers, abort=None)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None
    assert [d._geecs_device_name for d in session.scan_kwargs["detectors"]] == [
        "U_Ref",
        "U_Cam2",
    ]
    assert claims == ["TestExp"]


def test_missing_dialog_types_proceed_unchanged(monkeypatch) -> None:
    """geecs_scanner not importable (standalone install) → default behavior."""
    monkeypatch.setattr(bluesky_scanner, "ScanDialogEvent", None)
    monkeypatch.setattr(bluesky_scanner, "DialogRequest", None)
    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.0)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Cam2": None})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    scanner._on_event = events.append

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert session.scan_kwargs is not None
    assert len(session.scan_kwargs["detectors"]) == 2


# ---------------------------------------------------------------------------
# Item 2 — the stale re-check gives a just-connected monitor a second look
# ---------------------------------------------------------------------------


def test_recheck_clears_transiently_stale_device(monkeypatch) -> None:
    """A device whose first frame lands during the grace wait is not flagged."""
    _patch_dialog_channel(monkeypatch)
    claims: list = []
    _patch_claim(monkeypatch, claims)
    session = _FakeSession(last_acq={"U_Cam2": None})
    scanner = _make_scanner(
        session,
        {
            "U_Ref": {"synchronous": True, "variable_list": ["Sig"]},
            "U_Cam2": {"synchronous": True, "variable_list": ["Val"]},
        },
    )
    events: list = []
    scanner._on_event = events.append

    # The staleness-recheck sleep lives in geecs_bluesky.preflight now
    # (the pipeline module); patching its `time` module is the same
    # global-time patch as before the move.
    original_sleep = preflight.time.sleep

    def frame_arrives_during_grace(seconds: float) -> None:
        for det in scanner._detectors:
            if det._geecs_device_name == "U_Cam2":
                det._last_acq = _fresh()
        original_sleep(0)

    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.01)
    monkeypatch.setattr(preflight.time, "sleep", frame_arrives_during_grace)

    scanner._execute_scan(_noscan_config(), motor=None, positions=[None])

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert session.scan_kwargs is not None
    assert len(session.scan_kwargs["detectors"]) == 2
