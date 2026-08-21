"""Tests for the pre-flight pipeline runner mechanics.

The production check's *semantics* are pinned in
``test_preflight_unserved.py``; these tests pin the pipeline plumbing
itself: check ordering, abort short-circuiting, ``skip_remaining``, and
the headless Ask contract (queueserver decision 3, issue #649): engine-side
there is no operator to answer, so an :class:`Ask` takes its
``on_default`` branch with one WARNING naming the question — the
continue/abort answering engine lives client-side, pre-submit.
"""

from __future__ import annotations

import logging

from geecs_bluesky.preflight import (
    Aborted,
    Ask,
    OperatorQuestion,
    Passed,
    PreflightContext,
    run_preflight,
)


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
    )
    assert result == ["a", "b"]
    assert calls == ["one", "two"]


def test_abort_outcome_short_circuits() -> None:
    calls: list = []

    def aborting(ctx: PreflightContext) -> Aborted:
        return Aborted(reason="config invalid")

    result = run_preflight([aborting, _passing_check(calls, "never")], _ctx())
    assert result is None
    assert calls == []


def test_skip_remaining_stops_the_pipeline_but_passes() -> None:
    calls: list = []

    def opt_in(ctx: PreflightContext) -> Passed:
        return Passed(skip_remaining=True)

    result = run_preflight([opt_in, _passing_check(calls, "never")], _ctx())
    assert result == ["a", "b"]
    assert calls == []


def _asking_check(outcomes: dict, *, default_outcome=None):
    """One Ask check whose handlers record which branch ran."""

    def check(ctx: PreflightContext) -> Ask:
        def on_continue():
            outcomes["branch"] = "continue"
            ctx.detectors = ["a"]  # e.g. a drop
            return Passed()

        def on_default():
            outcomes["branch"] = "default"
            return default_outcome if default_outcome is not None else Passed()

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


def test_ask_takes_its_default_branch_with_one_warning(caplog) -> None:
    """Headless: an Ask never routes to on_continue — on_default runs, and
    one WARNING names the question so the choice is visible in scan.log."""
    outcomes: dict = {}
    calls: list = []
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.preflight"):
        result = run_preflight(
            [_asking_check(outcomes), _passing_check(calls, "later")],
            _ctx(),
        )
    assert outcomes["branch"] == "default"
    assert result == ["a", "b"]  # on_continue's mutation never happened
    assert calls == ["later"]  # a plain Passed default continues the pipeline
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "something is wrong" in warnings[0].getMessage()


def test_ask_default_skip_remaining_is_honored() -> None:
    """The on_default outcome is treated exactly like a direct check result."""
    outcomes: dict = {}
    calls: list = []
    result = run_preflight(
        [
            _asking_check(outcomes, default_outcome=Passed(skip_remaining=True)),
            _passing_check(calls, "never"),
        ],
        _ctx(),
    )
    assert outcomes["branch"] == "default"
    assert result == ["a", "b"]
    assert calls == []  # the default handler asked to skip the rest


def test_ask_default_returning_aborted_aborts_the_run() -> None:
    """A check whose fail-loud default is abort still aborts headless."""
    outcomes: dict = {}
    result = run_preflight(
        [_asking_check(outcomes, default_outcome=Aborted(reason="fail loud"))],
        _ctx(),
    )
    assert outcomes["branch"] == "default"
    assert result is None


def test_context_question_carries_the_dialog_timeout() -> None:
    ctx = _ctx()
    ctx.dialog_timeout = 12.5
    question = ctx.question(RuntimeError("boom"), title="T", continue_label="C")
    assert isinstance(question, OperatorQuestion)
    assert question.timeout == 12.5
    assert question.abort_label == "Abort Scan"
    assert question.message == "boom"
