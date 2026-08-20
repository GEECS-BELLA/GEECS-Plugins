"""OperatorChannel — the one seam through which the engine asks the operator.

The engine never contains dialog plumbing; it contains *questions* (vision doc
§2).  A question is an :class:`OperatorQuestion`; whoever constructed the
engine injects an :class:`OperatorChannel` that knows how to get it answered:

- :class:`EventStreamOperator` — reproduces the GUI path: wraps the question
  in a :class:`~geecs_bluesky.events.DialogRequest`, emits it inside a
  :class:`~geecs_bluesky.events.ScanDialogEvent` through the ``on_event``
  callback, and blocks on the request's ``response_event`` (with a timeout so
  an unattended scan never hangs on a dialog nobody will answer).
- :class:`NullOperator` — headless: logs the question and returns its
  default answer immediately.

Answers are plain strings so channels stay trivial to implement:
``"continue"`` / ``"abort"`` when the consumer answered, ``"default"`` when
nobody did (headless, emit failure, or timeout) — callers must then preserve
their fail-loud default behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from geecs_bluesky import events as _events

logger = logging.getLogger(__name__)

# Canonical answer strings (see module docstring).
ANSWER_CONTINUE = "continue"
ANSWER_ABORT = "abort"
ANSWER_DEFAULT = "default"

# How long EventStreamOperator waits for an answer when the question does not
# carry its own timeout.  On timeout the caller's default behavior applies —
# a headless or unattended scan must never hang on a dialog nobody answers.
DEFAULT_DIALOG_TIMEOUT_S = 30.0


@dataclass
class OperatorQuestion:
    """One question for the operator, with its two answers spelled out.

    Parameters
    ----------
    message :
        Operator-facing body text.  When ``exc`` is set, consumers render
        ``str(exc)`` (the two should agree; ``message`` exists so channels
        without an exception concept can still show the question).
    title :
        Dialog window title.
    continue_label :
        Text for the non-abort button (e.g. ``"Drop && Continue"``,
        ``"Try Anyway"``) — the question author owns the wording of what
        "continue" means.
    abort_label :
        Text for the abort button.
    default :
        Answer returned when nobody answers (headless / timeout).  The
        conventional value is :data:`ANSWER_DEFAULT`, telling the caller to
        preserve its fail-loud default behavior.
    timeout :
        Seconds to wait for an answer; ``None`` uses the channel's default.
    exc :
        Optional exception carried into the ``DialogRequest`` so consumers
        that render exceptions (the GUI dialog) see the original object.
    context :
        Optional extra information for the dialog body.
    """

    message: str
    title: Optional[str] = None
    continue_label: Optional[str] = None
    abort_label: Optional[str] = None
    default: str = ANSWER_DEFAULT
    timeout: Optional[float] = None
    exc: Optional[Exception] = None
    context: Optional[str] = None


@runtime_checkable
class OperatorChannel(Protocol):
    """Anything that can get an :class:`OperatorQuestion` answered."""

    def ask(self, question: OperatorQuestion) -> str:
        """Return ``"continue"``, ``"abort"``, or the question's default."""
        ...


class NullOperator:
    """Headless channel: log the question, return its default answer."""

    def ask(self, question: OperatorQuestion) -> str:
        """Return the question's default immediately (no consumer to ask).

        Parameters
        ----------
        question :
            The question that would have been shown.

        Returns
        -------
        str
            ``question.default``.
        """
        logger.debug(
            "No operator channel consumer for question %r — returning "
            "default answer %r",
            question.title or question.message,
            question.default,
        )
        return question.default


class EventStreamOperator:
    """Channel that asks through the typed event stream (today's GUI path).

    Wraps the question in a :class:`~geecs_bluesky.events.DialogRequest`,
    emits a :class:`~geecs_bluesky.events.ScanDialogEvent` via the injected
    callback, and waits on ``response_event``.  Behavior matches the
    scanner's legacy one-question seam exactly (the since-deleted
    ``BlueskyScanner._request_operator_decision``): an emit failure or
    an unanswered wait returns the question's default.

    Parameters
    ----------
    emit :
        The ``on_event`` callback events are delivered through.
    dialog_event_type, request_type :
        Injectable event/request classes (default: the real ones from
        :mod:`geecs_bluesky.events`).  Test seam — hermetic tests substitute
        lightweight fakes.
    default_timeout :
        Wait budget used when a question carries no ``timeout``.
    """

    def __init__(
        self,
        emit: Callable[[Any], None],
        *,
        dialog_event_type: Optional[type] = None,
        request_type: Optional[type] = None,
        default_timeout: float = DEFAULT_DIALOG_TIMEOUT_S,
    ) -> None:
        self._emit = emit
        self._dialog_event_type = dialog_event_type or _events.ScanDialogEvent
        self._request_type = request_type or _events.DialogRequest
        self._default_timeout = default_timeout

    def ask(self, question: OperatorQuestion) -> str:
        """Emit the question as a dialog event and wait for the answer.

        Parameters
        ----------
        question :
            The question to ask.

        Returns
        -------
        str
            ``"continue"`` or ``"abort"`` when the consumer answered;
            ``question.default`` when the emit failed or no answer arrived
            within the timeout — callers must then preserve their fail-loud
            default behavior.
        """
        request = self._request_type(
            exc=question.exc
            if question.exc is not None
            else RuntimeError(question.message),
            context=question.context,
            title=question.title,
            continue_label=question.continue_label,
            abort_label=question.abort_label,
        )
        try:
            self._emit(self._dialog_event_type(request=request))
        except Exception:
            logger.debug("on_event callback raised; ignoring", exc_info=True)
            return question.default
        timeout = (
            question.timeout if question.timeout is not None else self._default_timeout
        )
        if not request.response_event.wait(timeout=timeout):
            logger.warning(
                "Operator dialog %r got no response within %.0f s — "
                "proceeding with default behavior",
                question.title,
                timeout,
            )
            return question.default
        return ANSWER_ABORT if request.abort[0] else ANSWER_CONTINUE
