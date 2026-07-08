"""Tests for the OperatorChannel seam (NullOperator / EventStreamOperator).

The channel is the one seam through which the engine asks the operator
anything; behavior is pinned against the previous inline implementation in
``BlueskyScanner._request_operator_decision``: headless → default answer,
emit failure → default, timeout → default, otherwise the consumer's
abort-flag maps to "abort"/"continue".
"""

from __future__ import annotations

import threading

from geecs_bluesky.events import DialogRequest, ScanDialogEvent
from geecs_bluesky.operator_channel import (
    ANSWER_ABORT,
    ANSWER_CONTINUE,
    ANSWER_DEFAULT,
    EventStreamOperator,
    NullOperator,
    OperatorChannel,
    OperatorQuestion,
)


def _question(**overrides) -> OperatorQuestion:
    base = dict(
        message="Device U_Cam2 is down. Drop it or abort.",
        title="Disconnected Device(s)",
        continue_label="Drop && Continue",
        abort_label="Abort Scan",
        timeout=1.0,
    )
    base.update(overrides)
    return OperatorQuestion(**base)


def test_null_operator_returns_default() -> None:
    assert NullOperator().ask(_question()) == ANSWER_DEFAULT


def test_null_operator_honours_custom_default() -> None:
    assert NullOperator().ask(_question(default="continue")) == "continue"


def test_implementations_satisfy_the_protocol() -> None:
    assert isinstance(NullOperator(), OperatorChannel)
    assert isinstance(EventStreamOperator(lambda e: None), OperatorChannel)


def test_event_stream_operator_emits_dialog_request_verbatim() -> None:
    """The DialogRequest carries the question's wording and exception."""
    events: list = []

    def answer_continue(event) -> None:
        events.append(event)
        event.request.abort[0] = False
        event.request.response_event.set()

    exc = RuntimeError("U_Cam2 (gateway reports DISCONNECTED)")
    channel = EventStreamOperator(answer_continue)
    answer = channel.ask(_question(exc=exc, context="extra context"))

    assert answer == ANSWER_CONTINUE
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ScanDialogEvent)
    request = event.request
    assert isinstance(request, DialogRequest)
    assert request.exc is exc
    assert request.context == "extra context"
    assert request.title == "Disconnected Device(s)"
    assert request.continue_label == "Drop && Continue"
    assert request.abort_label == "Abort Scan"


def test_event_stream_operator_maps_abort() -> None:
    def answer_abort(event) -> None:
        event.request.abort[0] = True
        event.request.response_event.set()

    assert EventStreamOperator(answer_abort).ask(_question()) == ANSWER_ABORT


def test_event_stream_operator_without_exc_wraps_the_message() -> None:
    """A message-only question still yields a request the GUI can render."""
    captured: list = []

    def consume(event) -> None:
        captured.append(event.request)
        event.request.response_event.set()

    EventStreamOperator(consume).ask(_question(exc=None))
    assert str(captured[0].exc) == _question().message


def test_event_stream_operator_timeout_returns_default() -> None:
    """Nobody answers → the question's default, after the question timeout."""
    channel = EventStreamOperator(lambda event: None)
    assert channel.ask(_question(timeout=0.05)) == ANSWER_DEFAULT


def test_event_stream_operator_uses_channel_default_timeout() -> None:
    """A question without its own timeout uses the injected channel default."""
    channel = EventStreamOperator(lambda event: None, default_timeout=0.05)
    assert channel.ask(_question(timeout=None)) == ANSWER_DEFAULT


def test_event_stream_operator_raising_emit_returns_default() -> None:
    def broken(event) -> None:
        raise RuntimeError("GUI went away")

    assert EventStreamOperator(broken).ask(_question()) == ANSWER_DEFAULT


def test_event_stream_operator_answer_from_another_thread() -> None:
    """The wait works across the worker→consumer thread boundary."""
    channel_events: list = []

    def answer_later(event) -> None:
        channel_events.append(event)

        def respond() -> None:
            event.request.abort[0] = True
            event.request.response_event.set()

        threading.Timer(0.02, respond).start()

    channel = EventStreamOperator(answer_later)
    assert channel.ask(_question(timeout=2.0)) == ANSWER_ABORT
