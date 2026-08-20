"""Operator-requested aborts are quiet, intentional outcomes (0.36.0).

Field incident (2026-07-15): an operator's Stop during a delegated
optimization run cleaned up perfectly (finalize disarm, ``exit_status
'abort'``, on_finish restore) yet the logs showed three ERROR blocks —
``RunEngineInterrupted`` escaping ``session.scan``/``session.optimize``,
the bridge's generic "scan thread raised an exception" traceback, and the
claimed-folder ERROR.  These tests pin the translation:

- the session catches ``RunEngineInterrupted`` at both ``RE(...)`` call
  sites and, when the engine settled back to ``idle`` (an abort/stop —
  bluesky raises the same type for a pause, which leaves the engine
  ``paused``), returns the aborted outcome with ``last_run_aborted`` set
  and one INFO line instead of raising;
- a *pause* still propagates (the operator may resume);
- optimize's on_finish restore-to-initial runs on the quiet abort path
  without depending on the exception propagating;
- ``log_claimed_scan_failure(aborted=True)`` is a calm WARNING, and the
  genuine-failure ERROR is unchanged.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

pytest.importorskip("aioca")

from bluesky.utils import RunEngineInterrupted  # noqa: E402
from ophyd_async.core import callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.scan_log import log_claimed_scan_failure  # noqa: E402
from geecs_bluesky.session import GeecsSession  # noqa: E402
from tests.ca_mock_helpers import start_pacer  # noqa: E402


def _session() -> GeecsSession:
    return GeecsSession("TestExp", tiled=False, mock=True)


def _abort_when_running(run_engine, *, settle_s: float = 0.5) -> threading.Thread:
    """Fire ``RE.abort()`` from another thread once the plan is running.

    The operator's Stop button: ``stop_scanning_thread`` calls
    ``RE.abort(reason=...)`` from the GUI thread while ``RE(plan)`` blocks
    the scan thread.
    """

    def aborter() -> None:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if run_engine.state == "running":
                time.sleep(settle_s)  # let the plan get properly underway
                try:
                    run_engine.abort(reason="operator clicked stop")
                except Exception:
                    pass  # the plan may have finished inside the window
                return
            time.sleep(0.01)

    thread = threading.Thread(target=aborter, daemon=True, name="operator-stop")
    thread.start()
    return thread


def _package_errors(caplog) -> list[logging.LogRecord]:
    """ERROR-or-worse records from this package's loggers."""
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and r.name.startswith("geecs_bluesky")
    ]


# ---------------------------------------------------------------------------
# Session: real RunEngine, real abort from another thread (mock CA devices)
# ---------------------------------------------------------------------------


def test_scan_abort_mid_plan_returns_quietly(caplog) -> None:
    """RE.abort() mid-scan: scan() returns (no raise), one INFO, no ERROR."""
    s = _session()
    det = s.detector("U_Cam", ["Sig"])
    set_mock_value(det.acq_timestamp, 1000.0)
    pacer = start_pacer(s.RE, [(det, 1000.0)], initial_delay=0.4, interval=0.05)
    aborter = _abort_when_running(s.RE)
    try:
        with caplog.at_level(logging.INFO, logger="geecs_bluesky"):
            s.scan(
                detectors=[det],
                motor=None,
                positions=[None],
                shots_per_step=200,
                mode="free_run",
                save_data=False,
            )
    finally:
        pacer.cancel()
        aborter.join(timeout=15.0)
        s.disconnect(det)

    assert s.last_run_aborted is True
    quiet = [
        r
        for r in caplog.records
        if "aborted by operator" in r.getMessage() and r.name == "geecs_bluesky.session"
    ]
    assert len(quiet) == 1 and quiet[0].levelno == logging.INFO
    assert _package_errors(caplog) == [], (
        "an operator abort must not produce ERROR records from this package"
    )


def test_optimize_abort_restores_initial_and_returns_history(caplog) -> None:
    """RE.abort() mid-optimize: quiet return and on_finish restore still runs."""
    s = _session()
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    callback_on_mock_put(
        knob._setpoint, lambda v, **kw: set_mock_value(knob.readback, v)
    )
    set_mock_value(knob.readback, 5.0)  # pre-optimization value to restore
    set_mock_value(cam.acq_timestamp, 1000.0)

    class _EndlessSuggester:
        def suggest(self):
            return {"s1h": 0.25}  # never stops — only the abort ends the run

        def observe(self, inputs, objective, bin_data):
            pass

    pacer = start_pacer(s.RE, [(cam, 1000.0)], initial_delay=0.4, interval=0.05)
    aborter = _abort_when_running(s.RE, settle_s=0.8)
    try:
        with caplog.at_level(logging.INFO, logger="geecs_bluesky"):
            _uid, history = s.optimize(
                variables={"s1h": knob},
                detectors=[cam],
                objective=lambda bin_data: 1.0,
                suggester=_EndlessSuggester(),
                shots_per_iteration=2,
                max_iterations=1000,
                save_data=False,
                on_finish="initial",
            )
    finally:
        pacer.cancel()
        aborter.join(timeout=15.0)
        s.disconnect(cam, knob)

    assert s.last_run_aborted is True
    assert isinstance(history, list)  # completed iterations, returned not raised
    # The restore no longer rides the exception path: it ran on the quiet
    # abort branch, driving the knob back to its pre-optimization value.
    assert s._read_movable(knob) == pytest.approx(5.0)
    assert _package_errors(caplog) == []
    assert any(
        "aborted by operator" in r.getMessage()
        for r in caplog.records
        if r.name == "geecs_bluesky.session" and r.levelno == logging.INFO
    )


# ---------------------------------------------------------------------------
# Session: the abort-vs-pause distinction (deterministic fake-RE pins)
# ---------------------------------------------------------------------------


class _InterruptingRE:
    """Stands in for a RunEngine whose ``__call__`` was interrupted.

    Bluesky raises the same ``RunEngineInterrupted`` for a pause and for an
    abort/stop; the difference is the settled engine state — ``paused``
    (resumable) vs ``idle`` (the ``_run`` coroutine's cleanup finished and
    reset the state before the blocked ``RE(...)`` call raises).
    """

    def __init__(self, state: str) -> None:
        self.state = state

    def __call__(self, plan) -> None:
        plan.close()
        raise RunEngineInterrupted("boilerplate pause message")


def test_settled_interruption_returns_aborted_outcome(caplog) -> None:
    """Interrupted with the engine idle (abort/stop) → quiet aborted return."""
    s = _session()
    det = s.detector("U_Cam", ["Sig"])
    real_re = s.RE
    s.RE = _InterruptingRE("idle")  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.INFO, logger="geecs_bluesky"):
            uid = s.scan(
                detectors=[det],
                motor=None,
                positions=[None],
                shots_per_step=1,
                save_data=False,
            )
    finally:
        s.RE = real_re
        s.disconnect(det)

    assert uid is None
    assert s.last_run_aborted is True
    assert any("aborted by operator" in r.getMessage() for r in caplog.records)
    assert _package_errors(caplog) == []


def test_paused_interruption_still_raises() -> None:
    """Interrupted with the engine paused → propagate (the operator may resume)."""
    s = _session()
    det = s.detector("U_Cam", ["Sig"])
    real_re = s.RE
    s.RE = _InterruptingRE("paused")  # type: ignore[assignment]
    try:
        with pytest.raises(RunEngineInterrupted):
            s.scan(
                detectors=[det],
                motor=None,
                positions=[None],
                shots_per_step=1,
                save_data=False,
            )
    finally:
        s.RE = real_re
        s.disconnect(det)

    assert s.last_run_aborted is False


# ---------------------------------------------------------------------------
# scan_log: the claimed-folder note is calm on abort, loud on failure
# ---------------------------------------------------------------------------


def test_claimed_scan_abort_note_is_calm_warning(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="geecs_bluesky.scan_log"):
        log_claimed_scan_failure(
            7, "/data/scans/Scan007", label="Optimization scan", aborted=True
        )
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "aborted by operator" in message
    assert "never deleted" in message
    assert "partial data may be present" in message


def test_claimed_scan_failure_note_stays_error(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="geecs_bluesky.scan_log"):
        log_claimed_scan_failure(7, "/data/scans/Scan007")
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert "failed or aborted" in record.getMessage()


def test_upstream_request_abort_traceback_is_filtered(caplog) -> None:
    """Upstream bluesky's 'Run aborted' RequestAbort traceback is dropped by
    the session's log filter; a genuine exception's record passes through
    (the filter keys on the exception TYPE, never the message)."""
    from bluesky.utils import RequestAbort

    s = _session()
    re_logger = s.RE.log.logger

    with caplog.at_level(logging.DEBUG, logger=re_logger.name):
        try:
            raise RequestAbort()
        except RequestAbort:
            re_logger.exception("Run aborted")  # upstream's abort-path record
        try:
            raise RuntimeError("real failure")
        except RuntimeError:
            re_logger.exception("Run aborted")  # upstream's genuine-error record

    aborted = [r for r in caplog.records if r.getMessage() == "Run aborted"]
    assert len(aborted) == 1, "only the genuine-exception record may survive"
    assert aborted[0].exc_info[0] is RuntimeError


def test_live_abort_produces_no_upstream_traceback(caplog) -> None:
    """End-to-end: a real RE.abort() leaves zero RequestAbort tracebacks in
    the log stream (the field complaint), while the quiet INFO remains."""
    s = _session()
    det = s.detector("U_Cam", ["Sig"])
    set_mock_value(det.acq_timestamp, 1000.0)
    pacer = start_pacer(s.RE, [(det, 1000.0)], initial_delay=0.4, interval=0.05)
    aborter = _abort_when_running(s.RE)
    try:
        with caplog.at_level(logging.DEBUG):
            s.scan(
                detectors=[det],
                motor=None,
                positions=[None],
                shots_per_step=200,
                mode="free_run",
                save_data=False,
            )
    finally:
        pacer.cancel()
        aborter.join(timeout=15.0)
        s.disconnect(det)

    from bluesky.utils import RequestAbort

    leaked = [
        r
        for r in caplog.records
        if r.exc_info
        and r.exc_info[0] is not None
        and issubclass(r.exc_info[0], RequestAbort)
    ]
    assert leaked == [], "RequestAbort tracebacks must not reach the log"


def test_abort_log_filter_attaches_once_across_sessions() -> None:
    """The filter lands on the process-global bluesky logger — repeated
    session construction must not stack duplicates (review finding, #570)."""
    from geecs_bluesky.session import _RequestAbortLogFilter

    s1 = _session()
    s2 = _session()
    assert s1.RE.log.logger is s2.RE.log.logger  # global, by upstream design
    ours = [
        f for f in s1.RE.log.logger.filters if isinstance(f, _RequestAbortLogFilter)
    ]
    assert len(ours) == 1


# ---------------------------------------------------------------------------
# The idle-window stop gate (#571): a stop that lands after the caller's
# last pre-run check but before the engine reports 'running' must still
# take effect.  RE.abort() on an idle engine raises TransitionError, so the
# session re-reads should_abort as the plan's first instruction — which the
# engine only fetches after setting state 'running' (bluesky 1.15.0,
# run_engine.py::_run), giving a happens-before edge over any stopper that
# observed the engine idle.
# ---------------------------------------------------------------------------


def test_stop_gate_trips_before_plan_start(caplog) -> None:
    """should_abort True at plan start → zero messages, no run, quiet abort."""
    s = _session()
    det = s.detector("U_Cam", ["Sig"])
    set_mock_value(det.acq_timestamp, 1000.0)
    documents: list[str] = []
    token = s.RE.subscribe(lambda name, doc: documents.append(name))
    try:
        with caplog.at_level(logging.INFO, logger="geecs_bluesky"):
            uid = s.scan(
                detectors=[det],
                motor=None,
                positions=[None],
                shots_per_step=5,
                mode="free_run",
                save_data=False,
                should_abort=lambda: True,  # the stop landed in the window
            )
    finally:
        s.RE.unsubscribe(token)
        s.disconnect(det)

    assert uid is None
    assert s.last_run_aborted is True
    assert documents == [], "the gate must skip the plan before any run opens"
    notes = [
        r
        for r in caplog.records
        if "stopped before the plan started" in r.getMessage()
        and r.name == "geecs_bluesky.session"
    ]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO
    assert _package_errors(caplog) == []


def test_stop_gate_quiet_when_no_stop_requested() -> None:
    """A never-tripping probe leaves the scan completely unchanged."""
    s = _session()
    det = s.detector("U_Cam", ["Sig"])
    set_mock_value(det.acq_timestamp, 1000.0)
    documents: list[str] = []
    token = s.RE.subscribe(lambda name, doc: documents.append(name))
    pacer = start_pacer(s.RE, [(det, 1000.0)], initial_delay=0.4, interval=0.05)
    try:
        s.scan(
            detectors=[det],
            motor=None,
            positions=[None],
            shots_per_step=2,
            mode="free_run",
            save_data=False,
            should_abort=lambda: False,
        )
    finally:
        pacer.cancel()
        s.RE.unsubscribe(token)
        s.disconnect(det)

    assert s.last_run_aborted is False
    assert "start" in documents and "stop" in documents


def test_optimize_stop_gate_trips_before_plan_start(caplog) -> None:
    """The same gate guards optimize: (None, []) returned, nothing moved."""
    s = _session()
    cam = s.detector("UC_Cam", ["Sig"], name="cam")
    knob = s.settable("U_S1H", "Current", name="s1h")
    set_mock_value(knob.readback, 5.0)
    set_mock_value(cam.acq_timestamp, 1000.0)

    class _NeverSuggester:
        def suggest(self):
            raise AssertionError("the plan must never run")

        def observe(self, inputs, objective, bin_data):
            raise AssertionError("the plan must never run")

    try:
        with caplog.at_level(logging.INFO, logger="geecs_bluesky"):
            uid, history = s.optimize(
                variables={"s1h": knob},
                detectors=[cam],
                objective=lambda bin_data: 1.0,
                suggester=_NeverSuggester(),
                shots_per_iteration=2,
                max_iterations=3,
                save_data=False,
                on_finish="initial",
                should_abort=lambda: True,
            )
    finally:
        s.disconnect(cam, knob)

    assert uid is None
    assert history == []
    assert s.last_run_aborted is True
    assert s._read_movable(knob) == pytest.approx(5.0), "nothing moved, no restore"
    assert _package_errors(caplog) == []
