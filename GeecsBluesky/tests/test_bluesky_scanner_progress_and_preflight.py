"""Tests for BlueskyScanner GUI progress events and the pre-flight checks.

Item 1 — ``ScanStepEvent`` emission: every Bluesky event document emits a
shot-level progress event through ``on_event`` (same channel and thread
semantics as the lifecycle events), clamped at ``total_shots`` so the free-run
tail-flush overcount stays cosmetic.

Item 2 — pre-flight liveness (both modes) + free-run staleness, driven
through :meth:`BlueskyScanner._delegated_preflight` — the runner hook the
delegated ScanRequest path calls with the assembled detector list, pre-claim
(the ordering itself is pinned in the scan-request seam suite).  Every
synchronous device's gateway ``CONNECTED`` PV (``connected_status``) is
read — a device reporting Disconnected raises an operator dialog through the
legacy ``ScanDialogEvent``/``DialogRequest`` channel (drop-and-continue vs
abort; abort-only for a disconnected free-run reference; fail-open when the
PV is unreadable).  In free-run mode a second stage checks ``acq_timestamp``
freshness for the trigger-must-be-free-running requirement: all CONNECTED
but all stale → the trigger-off dialog; the residual CONNECTED-but-stale
contributor keeps the drop dialog.  Strict runs the liveness stage only
(frames are not needed pre-scan).  Headless / unanswered → today's
proceed-and-fail-loudly default.  Dropped devices are **not** disconnected
by the hook — the runner's ``finally`` owns disconnection of everything it
created.

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
        last_acq: float | None,
        connected: str | Exception = "Connected",
    ) -> None:
        self.name = device.lower()  # ophyd (safe) name
        self._geecs_device_name = device
        self._last_acq = last_acq
        self.connected_status = _FakeConnectedSignal(connected)

    def trigger(self) -> None:  # reference-capable, like CaGenericDetector
        raise NotImplementedError("not exercised by these tests")


class _FakeSnapshotDevice:
    """Async snapshot fake: deliberately has NO ``_last_acq`` attribute."""

    def __init__(self, device: str) -> None:
        self.name = device.lower()
        self._geecs_device_name = device


def _labview_now() -> float:
    return time.time() + bluesky_scanner._LABVIEW_EPOCH_OFFSET


def _fresh() -> float:
    return _labview_now() - 1.0


def _stale() -> float:
    return _labview_now() - 60.0


def _sync(
    device: str,
    *,
    last_acq: float | None = None,
    fresh: bool = True,
    connected: str | Exception = "Connected",
) -> _FakeSyncDevice:
    """Build a sync-device fake, fresh by default."""
    if last_acq is None and fresh:
        last_acq = _fresh()
    return _FakeSyncDevice(device, last_acq, connected)


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


def _make_scanner() -> BlueskyScanner:
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._session = SimpleNamespace()
    scanner._experiment_dir = "TestExp"
    scanner._on_event = None
    scanner._current_state = None
    scanner._total_shots = 0
    scanner._total_steps = 0
    scanner._completed_shots = 0
    scanner._scan_number = None
    scanner._abort_requested = False
    scanner._optimization_loader = None
    scanner._scan_request = None
    scanner._request_resolver = None
    scanner._last_run_uid = None
    scanner._active_run_uid = None
    scanner._active_descriptor_uids = set()
    scanner._RE = SimpleNamespace(
        state="idle", abort=lambda reason=None: None, _loop=_bg_loop()
    )
    return scanner


def _begin_owned_run(
    scanner: BlueskyScanner, uid: str = "run-1", descriptor: str = "desc-1"
) -> str:
    """Claim a run as this scanner's own (state RUNNING + start/descriptor).

    Mirrors a real scan: the bridge sets RUNNING before the plan reaches the
    RunEngine, then the run's start and descriptor documents flow through
    ``_on_document``.  Returns the descriptor uid for the event documents.
    """
    scanner._current_state = scanner._scan_state("RUNNING")
    scanner._on_document("start", {"uid": uid})
    scanner._on_document("descriptor", {"uid": descriptor, "run_start": uid})
    return descriptor


def _patch_dialog_channel(monkeypatch) -> None:
    """Route dialog emission through the fakes (env-independent tests)."""
    monkeypatch.setattr(bluesky_scanner, "ScanDialogEvent", _FakeDialogEvent)
    monkeypatch.setattr(bluesky_scanner, "DialogRequest", _FakeDialogRequest)
    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.0)


def _dialog_consumer(events: list, answers: dict, *, abort: bool | None):
    """Return an on_event callback answering dialog requests inline.

    ``abort=None`` leaves the request unanswered (timeout path).
    """

    def on_event(event) -> None:
        events.append(event)
        if isinstance(event, _FakeDialogEvent):
            answers["request"] = event.request
            if abort is not None:
                event.request.abort[0] = abort
                event.request.response_event.set()

    return on_event


def _names(detectors: list | None) -> list[str] | None:
    if detectors is None:
        return None
    return [d._geecs_device_name for d in detectors]


# ---------------------------------------------------------------------------
# Item 1 — step/progress events from event documents
# ---------------------------------------------------------------------------


def test_event_documents_emit_step_progress(monkeypatch) -> None:
    """One ScanStepEvent per event document, with running shot counts."""
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", _FakeStepEvent)
    events: list = []
    scanner = _make_scanner()
    scanner._on_event = events.append
    scanner._total_shots = 4
    scanner._total_steps = 2

    desc = _begin_owned_run(scanner, uid="abc")
    for bin_number in (1, 1, 2, 2):
        scanner._on_document(
            "event", {"descriptor": desc, "data": {"bin_number": bin_number}}
        )

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
    scanner = _make_scanner()
    scanner._on_event = events.append
    scanner._total_shots = 2
    scanner._total_steps = 1

    desc = _begin_owned_run(scanner)
    scanner._on_document("event", {"descriptor": desc, "data": {"bin_number": 1}})
    scanner._on_document("event", {"descriptor": desc, "data": {"bin_number": 1}})
    # Tail flush: one extra event on the flush stream (no bin_number here).
    scanner._on_document("event", {"descriptor": desc, "data": {}})

    assert [e.shots_completed for e in events] == [1, 2, 2]
    assert events[-1].step_index == 0  # clamped inside [0, total_steps)
    # The raw counter still overcounts (known, cosmetic) but the estimate
    # exposed to pollers is clamped too.
    assert scanner._completed_shots == 3
    assert scanner.estimate_current_completion() == 1.0


def test_progress_without_callback_or_event_type_is_silent(monkeypatch) -> None:
    """No consumer or no geecs_scanner installed → counting still works."""
    scanner = _make_scanner()
    scanner._total_shots = 2
    desc = _begin_owned_run(scanner)
    scanner._on_document(
        "event", {"descriptor": desc, "data": {"bin_number": 1}}
    )  # _on_event None

    events: list = []
    scanner._on_event = events.append
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", None)
    scanner._on_document("event", {"descriptor": desc, "data": {"bin_number": 1}})

    assert events == []
    assert scanner._completed_shots == 2


def test_progress_survives_raising_callback(monkeypatch) -> None:
    """A misbehaving on_event consumer must not break document handling."""
    monkeypatch.setattr(bluesky_scanner, "ScanStepEvent", _FakeStepEvent)

    def bad_callback(event) -> None:
        raise RuntimeError("GUI went away")

    scanner = _make_scanner()
    scanner._on_event = bad_callback
    scanner._total_shots = 2
    desc = _begin_owned_run(scanner)
    scanner._on_document("event", {"descriptor": desc, "data": {"bin_number": 1}})
    assert scanner._completed_shots == 1


# ---------------------------------------------------------------------------
# Item 2 — pre-flight liveness: connected + fresh devices are untouched
# ---------------------------------------------------------------------------


def test_fresh_devices_no_dialog_no_behavior_change(monkeypatch) -> None:
    """All sync devices CONNECTED and fresh → no dialog, every device kept."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    detectors = [_sync("U_Ref"), _sync("U_Cam2"), _FakeSnapshotDevice("U_Stage")]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Ref", "U_Cam2", "U_Stage"]
    assert scanner._abort_requested is False


# ---------------------------------------------------------------------------
# Item 2 — liveness: a DISCONNECTED device raises the dialog in either mode
# ---------------------------------------------------------------------------


def test_disconnected_contributor_free_run_drop_removes_it(monkeypatch) -> None:
    """Free-run + gateway says a contributor is down → drop dialog.

    The device's frames may even be recent (a just-crashed device) — the
    CONNECTED PV alone is authoritative, so the dialog fires regardless of
    the staleness heuristic.  The dropped device stays connected: the
    delegated runner's ``finally`` owns disconnection.
    """
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=False)
    detectors = [
        _sync("U_Ref"),
        _sync("U_Cam2", connected="Disconnected"),
        _FakeSnapshotDevice("U_Stage"),
    ]

    checked = scanner._delegated_preflight(detectors, strict=False)

    request = answers["request"]
    assert "U_Cam2" in str(request.exc)
    assert "DISCONNECTED" in str(request.exc)  # liveness, not staleness, wording
    assert "stale" not in str(request.exc).lower()
    assert "drop" in request.continue_label.lower()
    # The dead device was removed; the rest of the list survives.
    assert _names(checked) == ["U_Ref", "U_Stage"]
    assert scanner._abort_requested is False


def test_disconnected_device_strict_drop_works(monkeypatch) -> None:
    """Strict + gateway says a device is down → same drop dialog.

    Live motivation (2026-07-07, Scan006): an OFF camera's data PVs still
    CA-connected fine (the gateway serves everything in the DB), so the scan
    ran into 3 wasted refires and a post-claim abort.  The CONNECTED PV now
    catches it before the claim, mode-independently.
    """
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=False)
    detectors = [_sync("U_Cam1"), _sync("U_Cam2", connected="Disconnected")]

    checked = scanner._delegated_preflight(detectors, strict=True)

    assert "DISCONNECTED" in str(answers["request"].exc)
    assert _names(checked) == ["U_Cam1"]  # dead device dropped


def test_disconnected_device_strict_abort_sets_flag(monkeypatch) -> None:
    """Strict + disconnected device + operator abort → None + abort flag.

    The runner treats ``None`` as an abort before any claim; the flag makes
    the scan thread's cleanup report ABORTED.
    """
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=True)
    detectors = [_sync("U_Cam1"), _sync("U_Cam2", connected="Disconnected")]

    checked = scanner._delegated_preflight(detectors, strict=True)

    assert checked is None
    assert scanner._abort_requested is True


def test_disconnected_reference_free_run_is_abort_only(monkeypatch) -> None:
    """A DISCONNECTED pacemaker offers abort vs clearly-labeled try-anyway."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=True)
    detectors = [_sync("U_Ref", connected="Disconnected"), _sync("U_Cam2")]

    checked = scanner._delegated_preflight(detectors, strict=False)

    request = answers["request"]
    assert "reference" in str(request.exc).lower()
    assert "U_Ref" in str(request.exc)
    assert "DISCONNECTED" in str(request.exc)
    # No drop offer — the second option is a clearly-labeled try-anyway.
    assert "anyway" in request.continue_label.lower()
    assert checked is None
    assert scanner._abort_requested is True


# ---------------------------------------------------------------------------
# Item 2 — liveness fail-open: an unreadable CONNECTED PV never blocks a scan
# ---------------------------------------------------------------------------


def test_connected_unreadable_fails_open(monkeypatch) -> None:
    """CONNECTED read raising (old gateway) → no dialog, no crash, proceed."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    unreadable = RuntimeError("no CONNECTED PV on this gateway")
    detectors = [
        _sync("U_Ref", connected=unreadable),
        _sync("U_Cam2", connected=unreadable),
    ]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Ref", "U_Cam2"]


def test_connected_read_timeout_fails_open(monkeypatch) -> None:
    """A hanging CONNECTED read (dead IOC link) times out and reads as live."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "_LIVENESS_READ_TIMEOUT_S", 0.05)

    class _HangingSignal:
        async def get_value(self) -> str:
            await asyncio.sleep(30.0)
            return "Connected"

    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    device = _sync("U_Ref")
    device.connected_status = _HangingSignal()

    checked = scanner._delegated_preflight([device], strict=False)

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Ref"]


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
    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    detectors = [_sync("U_Cam", last_acq=None, fresh=False)]

    checked = scanner._delegated_preflight(detectors, strict=True)

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Cam"]


def test_strict_differential_stale_proceeds_silently(monkeypatch) -> None:
    """Strict + CONNECTED devices with mixed frame ages → no dialog.

    The previous differential-staleness heuristic is gone: CONNECTED is the
    authoritative liveness signal, so a connected-but-frameless device before
    a strict scan (trigger off, camera armed late, …) is not diagnosable as
    dead and must not dialog.
    """
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    detectors = [_sync("U_Cam1"), _sync("U_Cam2", last_acq=_stale())]

    checked = scanner._delegated_preflight(detectors, strict=True)

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Cam1", "U_Cam2"]  # nothing dropped


# ---------------------------------------------------------------------------
# Item 2 — free-run residual case: CONNECTED-but-stale contributor
# ---------------------------------------------------------------------------


def test_stale_connected_contributor_keeps_drop_dialog(monkeypatch) -> None:
    """CONNECTED contributor with no frames + fresh reference → drop dialog.

    The residual staleness case: the fresh reference proves the trigger is
    running, so a frameless-but-CONNECTED contributor is a per-device
    acquisition problem (camera acquisition stopped while its TCP stream is
    up) — the drop-or-abort dialog is kept for it, with not-acquiring (not
    dead-device) wording.  The dropped device stays connected for the
    runner's own cleanup.
    """
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=False)
    detectors = [
        _sync("U_Ref"),
        _sync("U_Cam2", last_acq=_stale()),
        _FakeSnapshotDevice("U_Stage"),
    ]

    checked = scanner._delegated_preflight(detectors, strict=False)

    request = answers["request"]
    assert "U_Cam2" in str(request.exc)
    assert "s ago" in str(request.exc)  # says how stale
    assert "CONNECTED" in str(request.exc)  # liveness already vouched for it
    assert request.continue_label is not None
    assert "drop" in request.continue_label.lower()
    assert _names(checked) == ["U_Ref", "U_Stage"]


def test_stale_contributor_abort_returns_none(monkeypatch) -> None:
    """Operator answers abort → None + the abort flag, pre-claim."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=True)
    detectors = [_sync("U_Ref"), _sync("U_Cam2", last_acq=None, fresh=False)]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert checked is None
    assert scanner._abort_requested is True


# ---------------------------------------------------------------------------
# Item 2 — CONNECTED-but-stale reference: abort-only v1
# ---------------------------------------------------------------------------


def test_stale_reference_dialog_is_abort_only_v1(monkeypatch) -> None:
    """A CONNECTED but frameless pacemaker offers abort vs labeled retry."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=True)
    detectors = [_sync("U_Ref", last_acq=_stale()), _sync("U_Cam2")]

    checked = scanner._delegated_preflight(detectors, strict=False)

    request = answers["request"]
    assert "reference" in str(request.exc).lower()
    assert "U_Ref" in str(request.exc)
    # No drop offer — the second option is a clearly-labeled try-anyway.
    assert "anyway" in request.continue_label.lower()
    assert checked is None
    assert scanner._abort_requested is True


def test_stale_reference_try_anyway_proceeds_with_full_list(monkeypatch) -> None:
    """Choosing try-anyway preserves today's behavior (t0 sync fails loudly)."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=False)
    detectors = [_sync("U_Ref", last_acq=_stale()), _sync("U_Cam2")]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert _names(checked) == ["U_Ref", "U_Cam2"]


# ---------------------------------------------------------------------------
# Item 2 — all sync devices stale: the trigger is probably off
# ---------------------------------------------------------------------------


def test_all_stale_dialog_blames_the_trigger(monkeypatch) -> None:
    """All CONNECTED but no fresh frames anywhere → trigger-off wording.

    Now unambiguous: the liveness stage already confirmed every device's TCP
    stream is up, so a total absence of frames can only be the trigger.
    """
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=True)
    detectors = [
        _sync("U_Ref", last_acq=None, fresh=False),
        _sync("U_Cam2", last_acq=_stale()),
    ]

    checked = scanner._delegated_preflight(detectors, strict=False)

    request = answers["request"]
    assert "trigger appears to be off" in str(request.exc)
    assert "CONNECTED" in str(request.exc)
    assert checked is None
    assert scanner._abort_requested is True


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
    scanner = _make_scanner()
    scanner._on_event = None
    detectors = [_sync("U_Ref"), _sync("U_Cam2", connected="Disconnected")]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert _names(checked) == ["U_Ref", "U_Cam2"]
    assert scanner._abort_requested is False


def test_unanswered_dialog_times_out_and_proceeds(monkeypatch) -> None:
    """No response within the (shortened) timeout → proceed unchanged."""
    _patch_dialog_channel(monkeypatch)
    monkeypatch.setattr(bluesky_scanner, "_PREFLIGHT_DIALOG_TIMEOUT_S", 0.05)
    scanner = _make_scanner()
    events: list = []
    answers: dict = {}
    scanner._on_event = _dialog_consumer(events, answers, abort=None)
    detectors = [_sync("U_Ref"), _sync("U_Cam2", connected="Disconnected")]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Ref", "U_Cam2"]


def test_missing_dialog_types_proceed_unchanged(monkeypatch) -> None:
    """geecs_scanner not importable (standalone install) → default behavior."""
    monkeypatch.setattr(bluesky_scanner, "ScanDialogEvent", None)
    monkeypatch.setattr(bluesky_scanner, "DialogRequest", None)
    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.0)
    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    detectors = [_sync("U_Ref"), _sync("U_Cam2", last_acq=None, fresh=False)]

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert _names(checked) == ["U_Ref", "U_Cam2"]


# ---------------------------------------------------------------------------
# Item 2 — the stale re-check gives a just-connected monitor a second look
# ---------------------------------------------------------------------------


def test_recheck_clears_transiently_stale_device(monkeypatch) -> None:
    """A device whose first frame lands during the grace wait is not flagged."""
    _patch_dialog_channel(monkeypatch)
    scanner = _make_scanner()
    events: list = []
    scanner._on_event = events.append
    detectors = [_sync("U_Ref"), _sync("U_Cam2", last_acq=None, fresh=False)]

    # The staleness-recheck sleep lives in geecs_bluesky.preflight (the
    # pipeline module); patching its `time` module is the global-time patch.
    original_sleep = preflight.time.sleep

    def frame_arrives_during_grace(seconds: float) -> None:
        for det in detectors:
            if det._geecs_device_name == "U_Cam2":
                det._last_acq = _fresh()
        original_sleep(0)

    monkeypatch.setattr(bluesky_scanner, "_STALE_RECHECK_WAIT_S", 0.01)
    monkeypatch.setattr(preflight.time, "sleep", frame_arrives_during_grace)

    checked = scanner._delegated_preflight(detectors, strict=False)

    assert not any(isinstance(e, _FakeDialogEvent) for e in events)
    assert _names(checked) == ["U_Ref", "U_Cam2"]
