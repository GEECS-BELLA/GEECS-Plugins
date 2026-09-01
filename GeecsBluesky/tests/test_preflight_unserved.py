"""Tests for the unserved-variables pre-flight check (the 2026-07-15 incident).

A save set naming variables the gateway does not serve — real DB variables
that are neither ``get='yes'`` in ``expt_device_variable`` nor settable
(live case: ``UC_TopView`` ``2ndmomW0x``/``2ndmomW0y``) — used to die 20 s
into detector connect with an ophyd ``NotConnectedError`` traceback.  These
tests pin the replacement behavior on the ScanRequest paths.  Engine-side
the check runs headless (queueserver decision 3, issue #649 — the operator
is asked client-side pre-submit): its :class:`Ask` names every unserved
variable, and the headless default (with a WARNING) drops exactly those
variables from the devices config — a device whose every listed variable
is unserved is dropped whole — recorded in run metadata; a DB failure
degrades to pass with one warning.  The check runs on noscan/step *and*
optimize.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from geecs_bluesky.preflight import (
    Ask,
    PreflightContext,
    UnservedVariablesCheck,
    run_unserved_variables_check,
)
from geecs_schemas import SaveSet, SaveSetEntry, ScanRequest
from tests.test_scan_request_runner import run_request

# ---------------------------------------------------------------------------
# Fakes: session recording variable lists, one-save-set resolver
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Fake session recording each device build's variable list (no CA/RE).

    Exposes the plan preamble's session seams (see the runner suite's
    ``_FakeSession``); ``build_claimed_scan_plan`` records its kwargs as
    :attr:`scan_kwargs` and returns a plan that yields nothing.
    """

    experiment = ""
    rep_rate_hz = 1.0
    _mock = True

    def __init__(self) -> None:
        self.device_calls: list[tuple[str, str, list[str]]] = []
        self.scan_kwargs: dict | None = None
        self.disconnected: list = []

    def _refuse_if_manual_move(self, verb: str) -> None:
        pass

    def build_claimed_scan_plan(self, **kwargs):
        self.scan_kwargs = kwargs
        return _immediately("uid-scan")

    def _make(self, kind: str, device: str, variables: list[str]):
        self.device_calls.append((kind, device, list(variables)))
        return SimpleNamespace(_geecs_device_name=device, kind=kind)

    def detector(
        self,
        device,
        variables,
        *,
        save_images=False,
        save_control_only=False,
        name=None,
    ):
        return self._make("detector", device, variables)

    def contributor(
        self,
        device,
        variables,
        *,
        save_images=False,
        save_control_only=False,
        name=None,
    ):
        return self._make("contributor", device, variables)

    def snapshot(self, device, variables, *, save_control_only=False, name=None):
        return self._make("snapshot", device, variables)

    def settable(self, device, variable, *, name=None):
        return SimpleNamespace(_geecs_device_name=f"{device}:{variable}")

    def disconnect(self, *devices) -> None:
        self.disconnected.extend(devices)


def _immediately(value):
    return value
    yield  # pragma: no cover — makes this a generator


class _SaveSetResolver:
    """Minimal resolver: named save sets only (no defaults, no actions)."""

    def __init__(self, save_sets: dict[str, SaveSet]) -> None:
        self._save_sets = save_sets

    def resolve_save_set(self, name: str) -> SaveSet:
        return self._save_sets[name]


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
    # free_run: the resolver serves no trigger profile, and the preamble
    # refuses a strict request without shot control before this check runs.
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="free_run",
        save_sets=["TopView"],
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


def _install_served(monkeypatch, served: dict[str, set[str]] | None) -> None:
    """Route the runner's served-set provider to an in-memory map (or None)."""
    import geecs_bluesky.scan_request_runner as runner

    provider = SimpleNamespace(served_by_device=lambda: served)
    monkeypatch.setattr(runner, "make_served_set_provider", lambda session: provider)


def _check_ctx() -> PreflightContext:
    return PreflightContext(
        detectors=[],
        strict=False,
        read_liveness=lambda device: True,
        drop_devices=lambda detectors, ids: detectors,
        device_label=str,
    )


# ---------------------------------------------------------------------------
# The check through the scan plan's preamble (noscan/step path)
# ---------------------------------------------------------------------------


def test_all_served_asks_nothing_and_keeps_the_config(monkeypatch) -> None:
    _install_served(
        monkeypatch, {"UC_TopView": {"centroidx", "2ndmomW0x", "2ndmomW0y"}}
    )
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    uid = run_request(session, _noscan_request(), resolver)

    assert uid == "uid-scan"
    assert session.device_calls == [
        ("detector", "UC_TopView", ["centroidx", "2ndmomW0x", "2ndmomW0y"])
    ]
    assert "dropped_unserved_variables" not in session.scan_kwargs["md"]


def test_unserved_variables_raise_one_ask_with_the_pinned_text() -> None:
    """The Ask's wording is the console modal's body (rendered client-side
    pre-submit) — pinned at the check level since the engine answers no
    questions anymore."""
    check = UnservedVariablesCheck(
        devices_config={
            "UC_TopView": {
                "variable_list": ["centroidx", "2ndmomW0x", "2ndmomW0y"],
                "synchronous": True,
            },
            "U_Ghost": {"variable_list": ["foo"], "synchronous": True},
        },
        served_by_device=lambda: _SERVED,
    )
    result = check(_check_ctx())

    assert isinstance(result, Ask)
    question = result.question
    assert question.message.startswith(
        "UC_TopView:2ndmomW0x, UC_TopView:2ndmomW0y, U_Ghost:foo are not set "
        "to 'get' in expt_device_variable, so the gateway does not serve them."
    )
    # One question covers both the partial and the whole-device drop.
    assert (
        "Every listed variable of U_Ghost is unserved, so continuing drops "
        "the device(s) entirely." in question.message
    )
    assert question.message.endswith("Continue without these variables?")
    assert question.title == "Unserved Save-Set Variable(s)"
    assert question.continue_label == "Continue Without Them"


def test_fully_unserved_device_is_dropped_whole(monkeypatch) -> None:
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    ghost = SaveSetEntry(device="U_Ghost", scalars=["foo"], db_scalars=False)
    resolver = _SaveSetResolver({"TopView": _topview_save_set([ghost])})

    run_request(session, _noscan_request(), resolver)

    built = [device for _kind, device, _vars in session.device_calls]
    assert built == ["UC_TopView"]  # U_Ghost never built
    md = session.scan_kwargs["md"]
    assert md["dropped_unserved_variables"]["U_Ghost"] == ["foo"]
    assert md["dropped_unserved_devices"] == ["U_Ghost"]


def test_db_failure_degrades_to_pass_with_one_warning(monkeypatch, caplog) -> None:
    _install_served(monkeypatch, None)  # served set unknown (DB unreachable)
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    with caplog.at_level(logging.WARNING):
        uid = run_request(session, _noscan_request(), resolver)

    assert uid == "uid-scan"
    # Never blocks a scan on a DB blip: nothing dropped, full list built.
    assert session.device_calls == [
        ("detector", "UC_TopView", ["centroidx", "2ndmomW0x", "2ndmomW0y"])
    ]
    assert "dropped_unserved_variables" not in session.scan_kwargs["md"]
    warnings = [
        r
        for r in caplog.records
        if "served set could not be determined" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_headless_default_continues_and_drops_with_a_warning(
    monkeypatch, caplog
) -> None:
    """THE engine-side contract: nobody to ask, so the default applies —
    continue-and-drop, one WARNING, provenance in run metadata."""
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    resolver = _SaveSetResolver({"TopView": _topview_save_set()})

    with caplog.at_level(logging.WARNING):
        uid = run_request(session, _noscan_request(), resolver)

    assert uid == "uid-scan"
    # The detector is built from the reduced list — the unserved variables
    # never reach a device (that is what prevented the connect timeout).
    assert session.device_calls == [("detector", "UC_TopView", ["centroidx"])]
    md = session.scan_kwargs["md"]
    assert md["dropped_unserved_variables"] == {
        "UC_TopView": ["2ndmomW0x", "2ndmomW0y"]
    }
    assert "dropped_unserved_devices" not in md
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


def test_no_provider_skips_the_check() -> None:
    config = {"U_Cam": {"variable_list": ["x"], "synchronous": True}}
    effective, dropped, dropped_devices = run_unserved_variables_check(config, None)
    assert effective is config
    assert dropped == {}
    assert dropped_devices == []


def test_gateway_synthesized_variables_are_always_served(monkeypatch) -> None:
    """acq_timestamp/systimestamp/CONNECTED are gateway-synthesized for every
    device (no expt_device_variable row exists) — they must never draw the
    unserved question. Field regression 2026-07-16: the optimizer's
    auto-provisioned ``acq_timestamp`` request produced a false
    whole-device-drop question for UC_TopView.
    """
    _install_served(monkeypatch, _SERVED)
    session = _RecordingSession()
    save_set = SaveSet(
        name="TopView",
        entries=[
            SaveSetEntry(
                device="UC_TopView",
                scalars=["acq_timestamp", "systimestamp", "CONNECTED"],
                db_scalars=False,
            )
        ],
    )
    resolver = _SaveSetResolver({"TopView": save_set})

    uid = run_request(session, _noscan_request(), resolver)

    assert uid == "uid-scan"
    # No drop: synthesized vars are served, full list reaches the device.
    assert session.device_calls == [
        ("detector", "UC_TopView", ["acq_timestamp", "systimestamp", "CONNECTED"])
    ]
    assert "dropped_unserved_variables" not in session.scan_kwargs["md"]
