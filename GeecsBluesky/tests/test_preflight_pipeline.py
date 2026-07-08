"""Tests for the pre-flight pipeline runner mechanics.

The two production checks' *semantics* are pinned end to end by
``test_bluesky_scanner_progress_and_preflight.py`` (unchanged); these tests
pin the pipeline plumbing itself: check ordering, abort short-circuiting,
``skip_remaining``, and the ask→answer routing through the OperatorChannel.
"""

from __future__ import annotations

from geecs_bluesky.operator_channel import OperatorQuestion
from geecs_bluesky.preflight import (
    Aborted,
    Ask,
    Passed,
    PreflightContext,
    run_preflight,
)


class _ScriptedChannel:
    """Answers questions from a scripted list and records them."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.questions: list[OperatorQuestion] = []

    def ask(self, question: OperatorQuestion) -> str:
        self.questions.append(question)
        return self.answers.pop(0)


def _ctx(detectors: list | None = None) -> PreflightContext:
    return PreflightContext(
        detectors=detectors if detectors is not None else ["a", "b"],
        strict=False,
        read_liveness=lambda device: True,
        drop_devices=lambda detectors, ids: detectors,
        device_label=str,
    )


def _passing_check(calls: list, label: str):
    def check(ctx: PreflightContext) -> Passed:
        calls.append(label)
        return Passed()

    return check


def test_all_checks_pass_returns_detectors_in_order() -> None:
    calls: list = []
    ctx = _ctx()
    result = run_preflight(
        [_passing_check(calls, "one"), _passing_check(calls, "two")],
        ctx,
        _ScriptedChannel([]),
    )
    assert result == ["a", "b"]
    assert calls == ["one", "two"]


def test_abort_outcome_short_circuits() -> None:
    calls: list = []

    def aborting(ctx: PreflightContext) -> Aborted:
        return Aborted(reason="config invalid")

    result = run_preflight(
        [aborting, _passing_check(calls, "never")], _ctx(), _ScriptedChannel([])
    )
    assert result is None
    assert calls == []


def test_skip_remaining_stops_the_pipeline_but_passes() -> None:
    calls: list = []

    def opt_in(ctx: PreflightContext) -> Passed:
        return Passed(skip_remaining=True)

    result = run_preflight(
        [opt_in, _passing_check(calls, "never")], _ctx(), _ScriptedChannel([])
    )
    assert result == ["a", "b"]
    assert calls == []


def _asking_check(outcomes: dict):
    """One Ask check whose handlers record which branch ran."""

    def check(ctx: PreflightContext) -> Ask:
        def on_continue():
            outcomes["branch"] = "continue"
            ctx.detectors = ["a"]  # e.g. a drop
            return Passed()

        def on_default():
            outcomes["branch"] = "default"
            return Passed(skip_remaining=True)

        return Ask(
            question=ctx.question(
                RuntimeError("something is wrong"),
                title="Something Wrong",
                continue_label="Fix && Continue",
            ),
            on_continue=on_continue,
            on_default=on_default,
            abort_reason="operator aborted",
        )

    return check


def test_ask_routes_continue_through_the_channel() -> None:
    outcomes: dict = {}
    channel = _ScriptedChannel(["continue"])
    result = run_preflight([_asking_check(outcomes)], _ctx(), channel)
    assert outcomes["branch"] == "continue"
    assert result == ["a"]  # the handler's mutation is what the runner returns
    assert channel.questions[0].title == "Something Wrong"
    assert channel.questions[0].continue_label == "Fix && Continue"


def test_ask_abort_answer_aborts_the_run() -> None:
    outcomes: dict = {}
    result = run_preflight(
        [_asking_check(outcomes)], _ctx(), _ScriptedChannel(["abort"])
    )
    assert result is None
    assert "branch" not in outcomes


def test_ask_default_answer_runs_default_handler() -> None:
    outcomes: dict = {}
    calls: list = []
    result = run_preflight(
        [_asking_check(outcomes), _passing_check(calls, "later")],
        _ctx(),
        _ScriptedChannel(["default"]),
    )
    assert outcomes["branch"] == "default"
    assert result == ["a", "b"]  # proceed unchanged
    assert calls == []  # default handler asked to skip the rest


def test_context_question_carries_the_dialog_timeout() -> None:
    ctx = _ctx()
    ctx.dialog_timeout = 12.5
    question = ctx.question(RuntimeError("boom"), title="T", continue_label="C")
    assert question.timeout == 12.5
    assert question.abort_label == "Abort Scan"
    assert question.message == "boom"
