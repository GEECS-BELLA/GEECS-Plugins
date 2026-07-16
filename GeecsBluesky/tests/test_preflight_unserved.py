"""Tests for the unserved-variables pre-flight check (the 2026-07-15 incident).

A save set naming variables the gateway does not serve — real DB variables
that are neither ``get='yes'`` in ``expt_device_variable`` nor settable
(live case: ``UC_TopView`` ``2ndmomW0x``/``2ndmomW0y``) — used to die 20 s
into detector connect with an ophyd ``NotConnectedError`` traceback.  These
tests pin the replacement behavior on the ScanRequest paths: ONE pre-claim
operator question naming every unserved variable; continue (and the headless
default, with a WARNING) drops exactly those variables from the devices
config — a device whose every listed variable is unserved is dropped whole —
and records them in run metadata; abort stops the run pre-claim (no scan
number burned); a DB failure degrades to pass with one warning.  The check
runs on noscan/step *and* optimize.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from geecs_bluesky.operator_channel import OperatorQuestion
from geecs_bluesky.preflight import run_unserved_variables_check
from geecs_bluesky.scan_request_runner import run_scan_request
from geecs_schemas import SaveSet, SaveSetEntry, ScanRequest

# ---------------------------------------------------------------------------
# Fakes: session recording variable lists, one-save-set resolver, channel
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Fake session recording each device build's variable list (no CA/RE)."""

    def __init__(self) -> None:
        self.device_calls: list[tuple[str, str, list[str]]] = []
        self.scan_kwargs: dict | None = None
        self.optimize_kwargs: dict | None = None
        self.disconnected: list = []

    def _make(self, kind: str, device: str, variables: list[str]):
        self.device_calls.append((kind, device, list(variables)))
        return SimpleNamespace(_geecs_device_name=device, kind=kind)

    def detector(self, device, variables, *, save_images=False, name=None):
        return self._make("detector", device, variables)

    def contributor(self, device, variables, *, save_images=False, name=None):
        return self._make("contributor", device, variables)

    def snapshot(self, device, variables, *, name=None):
        return self._make("snapshot", device, variables)

    def settable(self, device, variable, *, name=None):
        return SimpleNamespace(_geecs_device_name=f"{device}:{variable}")

    def shot_control(self, config) -> None:
        pass

    def scan(self, **kwargs):
        self.scan_kwargs = kwargs
        return "uid-scan"

    def optimize(self, **kwargs):
        self.optimize_kwargs = kwargs
        return "uid-opt", []

    def disconnect(self, *devices) -> None:
        self.disconnected.extend(devices)


class _SaveSetResolver:
    """Minimal resolver: named save sets only (no defaults, no actions)."""

    def __init__(self, save_sets: dict[str, SaveSet]) -> None:
        self._save_sets = save_sets

    def resolve_save_set(self, name: str) -> SaveSet:
        return self._save_sets[name]


class _ScriptedChannel:
    """Answers questions from a scripted list and records them."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.questions: list[OperatorQuestion] = []

    def ask(self, question: OperatorQuestion) -> str:
        self.questions.append(question)
        return self.answers.pop(0)


# The incident shape: UC_TopView's subscribed set is centroidx/y + counts;
# 2ndmomW0x/2ndmomW0y are real DB variables but not get='yes' → no PVs.
_SERVED = {
    "UC_TopView": {
        "centroidx",
        "centroidy",
        "MaxCounts",
        "MeanCounts",
        "save",  # settable control surface counts as served
        "localsavingpath",
    },
}


def _topview_save_set(extra_entries: list[SaveSetEntry] | None = None) -> SaveSet:
    entries = [
        SaveSetEntry(
            device="UC_TopView",
            scalars=["centroidx", "2ndmomW0x", "2ndmomW0y"],
            db_scalars=False,
        )
    ]
    return SaveSet(name="TopView", entries=entries + list(extra_entries or []))


def _noscan_request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="strict",
        save_sets=["TopView"],
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def _install_served(monkeypatch, served: dict[str, set[str]] | None) -> None:
    """Route the runner's served-set provider to an in-memory map (or None)."""
    import geecs_bluesky.scan_request_runner as runner

    provider = SimpleNamespace(served_by_device=lambda: served)
    monkeypatch.setattr(runner, "make_served_set_provider", lambda session: provider)


# ---------------------------------------------------------------------------
# The check through run_scan_request (noscan/step path)
# ---------------------------------------------------------------------------


def test_all_served_asks_nothing_and_keeps_the_config(monkeypatch) -> None:
    _install_served(
        monkeypatch, {"UC_TopView": {"centroidx", "2ndmomW0x", "2ndmomW0y"}}
    )
    session = _RecordingSession()
    channel = _ScriptedChannel([])
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    uid = run_scan_request(
        session, _noscan_request(), resolver, operator_channel=channel
    )

    assert uid == "uid-scan"
    assert channel.questions == []
    assert session.device_calls == [
        ("detector", "UC_TopView", ["centroidx", "2ndmomW0x", "2ndmomW0y"])
    ]
    assert "dropped_unserved_variables" not in session.scan_kwargs["md"]


def test_unserved_variables_ask_one_question_with_the_pinned_text(
    monkeypatch,
) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    channel = _ScriptedChannel(["continue"])
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    run_scan_request(session, _noscan_request(), resolver, operator_channel=channel)

    assert len(channel.questions) == 1
    question = channel.questions[0]
    assert question.message.startswith(
        "UC_TopView:2ndmomW0x, UC_TopView:2ndmomW0y are not set to 'get' in "
        "expt_device_variable, so the gateway does not serve them."
    )
    assert question.message.endswith("Continue without these variables?")
    assert question.title == "Unserved Save-Set Variable(s)"
    assert question.continue_label == "Continue Without Them"


def test_continue_drops_exactly_the_unserved_variables(monkeypatch) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    uid = run_scan_request(
        session,
        _noscan_request(),
        resolver,
        operator_channel=_ScriptedChannel(["continue"]),
    )

    assert uid == "uid-scan"
    # The detector is built from the reduced list — the unserved variables
    # never reach a device (that is what prevented the connect timeout).
    assert session.device_calls == [("detector", "UC_TopView", ["centroidx"])]
    md = session.scan_kwargs["md"]
    assert md["dropped_unserved_variables"] == {
        "UC_TopView": ["2ndmomW0x", "2ndmomW0y"]
    }
    assert "dropped_unserved_devices" not in md


def test_fully_unserved_device_is_dropped_whole(monkeypatch) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    ghost = SaveSetEntry(device="U_Ghost", scalars=["foo"], db_scalars=False)
    resolver = _SaveSetResolver({"TopView": _topview_save_set([ghost])})
    channel = _ScriptedChannel(["continue"])

    run_scan_request(session, _noscan_request(), resolver, operator_channel=channel)

    # One dialog covers both the partial and the whole-device drop.
    assert len(channel.questions) == 1
    assert (
        "Every listed variable of U_Ghost is unserved, so continuing drops "
        "the device(s) entirely." in channel.questions[0].message
    )
    built = [device for _kind, device, _vars in session.device_calls]
    assert built == ["UC_TopView"]  # U_Ghost never built
    md = session.scan_kwargs["md"]
    assert md["dropped_unserved_variables"]["U_Ghost"] == ["foo"]
    assert md["dropped_unserved_devices"] == ["U_Ghost"]


def test_abort_answer_aborts_pre_claim_before_any_device_is_built(
    monkeypatch,
) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    uid = run_scan_request(
        session,
        _noscan_request(),
        resolver,
        operator_channel=_ScriptedChannel(["abort"]),
    )

    assert uid is None
    # Pre-claim AND pre-device-build: the claim lives inside session.scan,
    # which never ran, and no detector was created either.
    assert session.scan_kwargs is None
    assert session.device_calls == []


def test_db_failure_degrades_to_pass_with_one_warning(monkeypatch, caplog) -> None:
    _install_served(monkeypatch, None)  # served set unknown (DB unreachable)
    session = _RecordingSession()
    channel = _ScriptedChannel([])
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    with caplog.at_level(logging.WARNING):
        uid = run_scan_request(
            session, _noscan_request(), resolver, operator_channel=channel
        )

    assert uid == "uid-scan"
    assert channel.questions == []  # never blocks a scan on a DB blip
    assert session.device_calls == [
        ("detector", "UC_TopView", ["centroidx", "2ndmomW0x", "2ndmomW0y"])
    ]
    warnings = [
        r
        for r in caplog.records
        if "served set could not be determined" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_headless_default_continues_and_drops_with_a_warning(
    monkeypatch, caplog
) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    with caplog.at_level(logging.WARNING):
        # No operator_channel at all — the headless NullOperator default.
        uid = run_scan_request(session, _noscan_request(), resolver)

    assert uid == "uid-scan"
    assert session.device_calls == [("detector", "UC_TopView", ["centroidx"])]
    md = session.scan_kwargs["md"]
    assert md["dropped_unserved_variables"] == {
        "UC_TopView": ["2ndmomW0x", "2ndmomW0y"]
    }
    assert any(
        "no operator answer — continuing without them" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# The check on the optimize path (runs pre-claim there too)
# ---------------------------------------------------------------------------


def _optimize_request(**overrides) -> ScanRequest:
    base = dict(
        mode="optimize",
        shots_per_step=3,
        acquisition="free_run",
        save_sets=["TopView"],
        optimization={
            "variables": {"U_S1H:Current": [-2.0, 2.0]},
            "objectives": {"counts": "MAXIMIZE"},
            "evaluator": {"module": "m", "class": "C"},
            "generator": {"name": "bayes_default"},
            "max_iterations": 4,
        },
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def test_optimize_path_runs_the_check_and_drops_on_continue(monkeypatch) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})
    channel = _ScriptedChannel(["continue"])

    uid = run_scan_request(
        session,
        _optimize_request(),
        resolver,
        objective=lambda rows: 0.0,
        suggester=lambda history: None,
        operator_channel=channel,
    )

    assert uid == "uid-opt"
    assert len(channel.questions) == 1
    # free_run: the first synchronous device is still the reference detector.
    assert session.device_calls == [("detector", "UC_TopView", ["centroidx"])]
    md = session.optimize_kwargs["md"]
    assert md["dropped_unserved_variables"] == {
        "UC_TopView": ["2ndmomW0x", "2ndmomW0y"]
    }


def test_optimize_path_abort_is_pre_claim(monkeypatch) -> None:
    _install_served(monkeypatch, _SERVED)
    import geecs_bluesky.scan_request_runner as runner

    claims: list = []
    monkeypatch.setattr(
        runner,
        "claim_scan",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    uid = run_scan_request(
        session,
        _optimize_request(),
        resolver,
        optimization_binder=lambda **kwargs: (lambda rows: 0.0, lambda history: None),
        operator_channel=_ScriptedChannel(["abort"]),
    )

    assert uid is None
    assert session.optimize_kwargs is None
    assert claims == []  # the binder path claims pre-bind — never reached
    assert session.device_calls == []


# ---------------------------------------------------------------------------
# Helper-level edge: no provider at all → the check is skipped entirely
# ---------------------------------------------------------------------------


def test_no_provider_skips_the_check() -> None:
    config = {"U_Cam": {"variable_list": ["x"], "synchronous": True}}
    effective, dropped, dropped_devices = run_unserved_variables_check(
        config, None, None
    )
    assert effective is config
    assert dropped == {}
    assert dropped_devices == []
