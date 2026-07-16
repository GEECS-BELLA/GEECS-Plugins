"""Pinning tests for BlueskyScanner scan-thread lifecycle behavior.

- A stop requested before the request reaches the delegated runner aborts
  the scan cleanly (``run_scan_request`` is never invoked), and a
  timed-out scan-thread join keeps the scanner reporting active instead
  of clearing the handle.
- Operator-requested aborts are quiet: an exception unwinding out of the
  delegated run after Stop is one INFO line (no ERROR traceback), a quiet
  aborted-outcome return still reports ABORTED, and an aborted
  optimization skips the bridge's ``finish()`` bookkeeping (success runs
  it — also pinned end-to-end in the scan-request seam suite).
- Belt-and-suspenders plan guards: the free-run plan itself rejects a
  non-Triggerable reference and still requires the shot-id tracker.

The exec_config-path device-build/claim-ordering tests that used to live
here died with the legacy arm (G3): device building, pre-claim ordering,
and optimizer device-requirements merging are owned by
``scan_request_runner`` and pinned in its suites.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner
from geecs_schemas import ScanRequestMode

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


def _make_scanner(session: object | None = None) -> BlueskyScanner:
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._session = session if session is not None else SimpleNamespace()
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
    scanner._RE = SimpleNamespace(
        state="idle", abort=lambda reason=None: None, _loop=None
    )
    return scanner


def _noscan_request_sentinel() -> SimpleNamespace:
    """A stored-request stand-in for tests that stub run_scan_request.

    ``_run_delegated_request`` only reads ``.mode`` before handing the
    request to the (monkeypatched) runner, so a SimpleNamespace suffices.
    """
    return SimpleNamespace(mode=ScanRequestMode.NOSCAN)


# ---------------------------------------------------------------------------
# Stop before the delegated runner runs; timed-out thread join
# ---------------------------------------------------------------------------


def test_stop_before_run_prevents_delegation(monkeypatch, caplog) -> None:
    """A stop that lands before _run_scan starts never reaches the runner."""
    scanner = _make_scanner()
    scanner._scan_request = _noscan_request_sentinel()
    scanner._request_resolver = object()
    scanner._abort_requested = True  # the stop button, pre-acquisition
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    runs: list = []
    monkeypatch.setattr(
        bluesky_scanner, "run_scan_request", lambda *a, **kw: runs.append(a)
    )

    with caplog.at_level(logging.WARNING):
        scanner._run_scan()

    assert runs == [], "the delegated runner must never be invoked"
    assert scanner.current_state == "aborted"
    assert "stop requested before acquisition" in caplog.text


def test_run_scan_without_stored_request_aborts_loudly(monkeypatch, caplog) -> None:
    """A scan thread with no stored ScanRequest fails loudly, never silently."""
    scanner = _make_scanner()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)

    with caplog.at_level(logging.ERROR):
        scanner._run_scan()

    assert scanner.current_state == "aborted"
    assert "no stored ScanRequest" in caplog.text


def test_timed_out_join_keeps_scanner_active(monkeypatch, caplog) -> None:
    """A join timeout must not clear the handle while the thread still runs.

    The still-finishing case is INFO, not ERROR (issue #571): a stop during
    initialization normally outlives the short bookkeeping join, and the
    terminal lifecycle event — not this call — announces completion.
    """
    scanner = _make_scanner()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    monkeypatch.setattr(bluesky_scanner, "_STOP_JOIN_TIMEOUT", 0.05)
    scanner._RE = SimpleNamespace(state="running", abort=lambda reason=None: None)

    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True, name="stuck-scan")
    thread.start()
    scanner._scan_thread = thread
    scanner._scan_finished = False
    try:
        with caplog.at_level(logging.INFO):
            scanner.stop_scanning_thread()

        assert scanner._scan_thread is thread, "handle kept after timed-out join"
        assert scanner.is_scanning_active() is True
        assert "still finishing" in caplog.text
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == [], "a slow stop is expected, never an ERROR"
    finally:
        release.set()
        thread.join(timeout=5)

    # Once the thread actually exits, a subsequent stop clears the handle.
    scanner._RE = SimpleNamespace(state="idle", abort=lambda reason=None: None)
    scanner.stop_scanning_thread()
    assert scanner._scan_thread is None
    assert scanner.is_scanning_active() is False


def test_stop_returns_promptly_while_thread_is_stuck_in_init() -> None:
    """stop_scanning_thread is bounded by the short bookkeeping join.

    The GUI-freeze half of issue #571: the old 15 s join on the GUI thread
    pinwheeled the window while the scan thread sat in 20 s device
    connects.  The join budget is now a short constant, so even a
    worst-case call returns in ~_STOP_JOIN_TIMEOUT.
    """
    scanner = _make_scanner()
    scanner._RE = SimpleNamespace(state="idle", abort=lambda reason=None: None)

    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True, name="init-stuck")
    thread.start()
    scanner._scan_thread = thread
    scanner._scan_finished = False
    try:
        started = time.monotonic()
        scanner.stop_scanning_thread()
        elapsed = time.monotonic() - started
        assert elapsed < bluesky_scanner._STOP_JOIN_TIMEOUT + 1.0
        assert scanner._abort_requested is True
        assert scanner.is_scanning_active() is True
    finally:
        release.set()
        thread.join(timeout=5)


def test_delegated_request_supplies_should_abort_probe(monkeypatch) -> None:
    """The bridge hands the runner a live probe of its abort flag."""
    scanner = _make_scanner()
    scanner._scan_request = _noscan_request_sentinel()
    scanner._request_resolver = object()
    captured: dict = {}

    def capture(session, request, resolver, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(bluesky_scanner, "run_scan_request", capture)
    scanner._run_delegated_request()

    probe = captured["should_abort"]
    assert probe() is False
    scanner._abort_requested = True
    assert probe() is True, "the probe must read the flag live, not a copy"


# ---------------------------------------------------------------------------
# Plan guards (belt-and-suspenders, independent of the bridge)
# ---------------------------------------------------------------------------


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
# Operator-requested aborts are quiet — no ERROR tracebacks
# ---------------------------------------------------------------------------


def test_scan_thread_operator_abort_logs_info_not_error(monkeypatch, caplog) -> None:
    """An exception unwinding after Stop is one INFO line, not a traceback."""
    scanner = _make_scanner()
    scanner._scan_request = _noscan_request_sentinel()
    scanner._request_resolver = object()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)

    def run_stopped_mid_scan(*args, **kwargs):
        # The operator's Stop: the flag is set and something unwinds out of
        # the interrupted run (the shape of the 2026-07-15 field incident).
        scanner._abort_requested = True
        raise RuntimeError("unwinding after RE.abort()")

    monkeypatch.setattr(bluesky_scanner, "run_scan_request", run_stopped_mid_scan)

    with caplog.at_level(logging.INFO):
        scanner._run_scan()

    assert scanner.current_state == "aborted"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], "operator abort must not produce ERROR records"
    assert any(
        "exiting after operator abort" in r.getMessage() and r.levelno == logging.INFO
        for r in caplog.records
    )


def test_scan_thread_genuine_failure_keeps_error_traceback(monkeypatch, caplog) -> None:
    """Without an abort request, a raising scan thread still ERRORs loudly."""
    scanner = _make_scanner()
    scanner._scan_request = _noscan_request_sentinel()
    scanner._request_resolver = object()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)

    def run_blows_up(*args, **kwargs):
        raise RuntimeError("genuine failure")

    monkeypatch.setattr(bluesky_scanner, "run_scan_request", run_blows_up)

    with caplog.at_level(logging.INFO):
        scanner._run_scan()

    assert scanner.current_state == "aborted"
    tracebacks = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and "scan thread raised an exception" in r.getMessage()
    ]
    assert len(tracebacks) == 1 and tracebacks[0].exc_info is not None


def test_operator_abort_quiet_return_reports_aborted(monkeypatch, caplog) -> None:
    """The runner's quiet aborted-outcome return still reports ABORTED."""
    scanner = _make_scanner(SimpleNamespace(last_run_aborted=False))
    scanner._scan_request = _noscan_request_sentinel()
    scanner._request_resolver = object()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)

    def run_aborted_quietly(session, *args, **kwargs):
        scanner._abort_requested = True  # Stop arrived while the RE ran
        session.last_run_aborted = True  # session translated it quietly
        return "uid"

    monkeypatch.setattr(bluesky_scanner, "run_scan_request", run_aborted_quietly)

    with caplog.at_level(logging.INFO):
        scanner._run_scan()

    assert scanner.current_state == "aborted"
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


class _FinishTrackingBridge:
    """Loader-returned optimization bridge fake with finish() bookkeeping."""

    def __init__(self) -> None:
        self.finished = False

    def bind(self, **kwargs):
        return (lambda bin_data: 0.0), self

    def finish(self) -> None:
        self.finished = True


def _optimize_request_sentinel() -> SimpleNamespace:
    return SimpleNamespace(mode=ScanRequestMode.OPTIMIZE, optimization=object())


def test_optimization_operator_abort_skips_finish_and_stays_calm(
    monkeypatch, caplog
) -> None:
    """An aborted optimization skips bridge.finish() quietly."""
    session = SimpleNamespace(last_run_aborted=False)
    scanner = _make_scanner(session)
    bridge = _FinishTrackingBridge()
    scanner._optimization_loader = lambda spec: bridge
    scanner._scan_request = _optimize_request_sentinel()
    scanner._request_resolver = object()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)

    def optimize_aborted(inner_session, *args, **kwargs):
        scanner._abort_requested = True
        inner_session.last_run_aborted = True
        return "uid"

    monkeypatch.setattr(bluesky_scanner, "run_scan_request", optimize_aborted)

    with caplog.at_level(logging.INFO):
        scanner._run_scan()

    assert scanner.current_state == "aborted"
    assert bridge.finished is False, "finish() must not run after an abort"
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_optimization_normal_completion_still_runs_finish(monkeypatch) -> None:
    """The abort guard must not eat the successful-run finish() bookkeeping."""
    session = SimpleNamespace(last_run_aborted=False)
    scanner = _make_scanner(session)
    bridge = _FinishTrackingBridge()
    scanner._optimization_loader = lambda spec: bridge
    scanner._scan_request = _optimize_request_sentinel()
    scanner._request_resolver = object()
    monkeypatch.setattr(bluesky_scanner, "ScanState", None)
    monkeypatch.setattr(bluesky_scanner, "run_scan_request", lambda *a, **kw: "uid")

    scanner._run_scan()

    assert scanner.current_state == "done"
    assert bridge.finished is True
