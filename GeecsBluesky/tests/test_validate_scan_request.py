"""validate_scan_request — THE one fail-fast definition (issue #529).

Pins the contract that submission-time validation (the GUI bridge's
``reinitialize``) and execution-time validation (``run_scan_request``'s
first phase) are the same function, so they cannot drift:

* every resolvable category refuses loudly through the one function;
* ``run_scan_request`` fails a bad request *before* touching any session
  state (no ``shot_control`` attach on a doomed request);
* the bridge routes ``reinitialize`` through the same function object.
"""

from __future__ import annotations

import pytest

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.scan_request_runner import (
    run_scan_request,
    validate_scan_request,
)
from geecs_bluesky.scanner_bridge import bluesky_scanner
from geecs_schemas import (
    ActionPlan,
    PseudoScanVariable,
    SaveSet,
    SaveSetEntry,
    ScanRequest,
    TriggerProfile,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubResolver:
    """Name → schema-model dict resolver (unknown names refuse loudly)."""

    def __init__(
        self,
        *,
        save_sets=None,
        trigger_profiles=None,
        variables=None,
        plans=None,
        defaults=None,
    ) -> None:
        self.save_sets = save_sets or {}
        self.trigger_profiles = trigger_profiles or {}
        self.variables = variables or {}
        self.plans = plans or {}
        self.defaults = defaults

    def resolve_save_set(self, name):
        try:
            return self.save_sets[name]
        except KeyError:
            raise GeecsConfigurationError(f"save set {name!r} not found") from None

    def resolve_trigger_profile(self, name):
        try:
            return self.trigger_profiles[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"trigger profile {name!r} not found"
            ) from None

    def resolve_scan_variable(self, name):
        try:
            return self.variables[name]
        except KeyError:
            raise GeecsConfigurationError(f"scan variable {name!r} not found") from None

    def resolve_action_plan(self, name):
        try:
            return self.plans[name]
        except KeyError:
            raise GeecsConfigurationError(f"action plan {name!r} not found") from None

    def resolve_experiment_defaults(self):
        return self.defaults


class _RefusingSession:
    """A session on which ANY touch is a test failure (fail-fast pin)."""

    def __getattr__(self, name):
        raise AssertionError(
            f"run_scan_request touched session.{name} before validation "
            "passed — phase 1 must fail first (issue #529)"
        )


_SAVE_SET = SaveSet(
    name="baseline",
    entries=[SaveSetEntry(device="U_Ref", scalars=["Sig"])],
)


def _resolver(**overrides) -> _StubResolver:
    base = dict(save_sets={"baseline": _SAVE_SET})
    base.update(overrides)
    return _StubResolver(**base)


def _request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=1,
        acquisition="free_run",
        save_sets=["baseline"],
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


# ---------------------------------------------------------------------------
# Each resolvable category refuses through the one function
# ---------------------------------------------------------------------------


def test_unknown_action_plan_refused() -> None:
    request = _request(actions={"setup": ["nope"]})
    with pytest.raises(GeecsConfigurationError, match="nope"):
        validate_scan_request(request, _resolver())


def test_unknown_save_set_refused() -> None:
    with pytest.raises(GeecsConfigurationError, match="ghost"):
        validate_scan_request(_request(save_sets=["ghost"]), _resolver())


def test_no_save_sets_refused_for_non_optimize() -> None:
    with pytest.raises(GeecsConfigurationError, match="at least one save"):
        validate_scan_request(_request(save_sets=[]), _resolver())


def test_no_save_sets_allowed_for_optimize() -> None:
    # The optimizer's device_requirements are auto-provisioned at execution
    # time; an empty *effective* set still refuses pre-claim there.
    request = _request(
        mode="optimize",
        save_sets=[],
        optimization={
            "variables": {"U_S1H:Current": [-2.0, 2.0]},
            "objectives": {"counts": "MAXIMIZE"},
            "evaluator": {"module": "m", "class": "C"},
            "generator": {"name": "bayes_default"},
        },
    )
    validated, applied = validate_scan_request(request, _resolver())
    assert validated.mode.value == "optimize"
    assert applied == {}


def test_unknown_trigger_profile_refused() -> None:
    with pytest.raises(GeecsConfigurationError, match="trigger profile"):
        validate_scan_request(_request(trigger_profile="HTU"), _resolver())


def test_unknown_trigger_variant_refused() -> None:
    profile = TriggerProfile(
        name="HTU",
        states={
            "SCAN": [{"device": "U_DG", "variable": "Trigger.Source", "value": "Ext"}]
        },
    )
    request = _request(trigger_profile="HTU", trigger_variant="no-such-variant")
    with pytest.raises(GeecsConfigurationError, match="variant"):
        validate_scan_request(request, _resolver(trigger_profiles={"HTU": profile}))


def test_unresolvable_step_axis_refused() -> None:
    request = _request(
        mode="step",
        axes=[{"variable": "ghost_axis", "positions": {"values": [0.0, 1.0]}}],
    )
    with pytest.raises(GeecsConfigurationError, match="ghost_axis"):
        validate_scan_request(request, _resolver())


def test_pseudo_step_axis_refused() -> None:
    pseudo = PseudoScanVariable(
        kind="pseudo",
        targets=[{"target": "U_X:Pos", "forward": "combo"}],
        mode="absolute",
    )
    request = _request(
        mode="step",
        axes=[{"variable": "combo", "positions": {"values": [0.0, 1.0]}}],
    )
    with pytest.raises(NotImplementedError, match="pseudo"):
        validate_scan_request(request, _resolver(variables={"combo": pseudo}))


def test_optimize_vocs_bare_name_must_resolve() -> None:
    request = _request(
        mode="optimize",
        save_sets=[],
        optimization={
            "variables": {"ghost_var": [0.0, 1.0]},
            "objectives": {"counts": "MAXIMIZE"},
            "evaluator": {"module": "m", "class": "C"},
            "generator": {"name": "bayes_default"},
        },
    )
    with pytest.raises(GeecsConfigurationError, match="ghost_var"):
        validate_scan_request(request, _resolver())


def test_optimize_vocs_device_variable_passes_through() -> None:
    # 'Device:Variable' strings bypass the catalog, matching the runner's
    # execution-time dispatch.
    request = _request(
        mode="optimize",
        save_sets=[],
        optimization={
            "variables": {"U_S1H:Current": [-2.0, 2.0]},
            "objectives": {"counts": "MAXIMIZE"},
            "evaluator": {"module": "m", "class": "C"},
            "generator": {"name": "bayes_default"},
        },
    )
    validated, _applied = validate_scan_request(request, _resolver())
    assert validated.optimization is not None


def test_returns_post_defaults_copy_with_provenance() -> None:
    profile = TriggerProfile(
        name="HTU",
        states={
            "SCAN": [{"device": "U_DG", "variable": "Trigger.Source", "value": "Ext"}]
        },
    )
    resolver = _resolver(
        trigger_profiles={"HTU": profile},
        defaults={"trigger_profile": "HTU"},
    )
    request = _request()  # no trigger_profile of its own
    validated, applied = validate_scan_request(request, resolver)
    assert validated.trigger_profile == "HTU"
    assert applied == {"trigger_profile": "HTU"}
    # The input request is untouched (post-defaults COPY, never in place).
    assert request.trigger_profile is None


def test_valid_plan_names_resolve_clean() -> None:
    plan = ActionPlan.model_validate({"steps": [{"do": "wait", "seconds": 0.01}]})
    request = _request(actions={"setup": ["prep"]})
    validated, applied = validate_scan_request(request, _resolver(plans={"prep": plan}))
    assert validated.actions.setup == ["prep"]
    assert applied == {}


# ---------------------------------------------------------------------------
# The two callers route through the one definition
# ---------------------------------------------------------------------------


def test_runner_fails_fast_before_touching_the_session() -> None:
    """Phase 1 refuses a bad request before ANY session attribute is used."""
    with pytest.raises(GeecsConfigurationError, match="ghost"):
        run_scan_request(_RefusingSession(), _request(save_sets=["ghost"]), _resolver())


def test_bridge_reinitialize_calls_the_shared_validator(monkeypatch) -> None:
    """reinitialize IS validate_scan_request (structural no-drift pin)."""
    calls: list[tuple] = []

    def _spy(request, resolver):
        calls.append((request, resolver))
        return request, {}

    monkeypatch.setattr(bluesky_scanner, "validate_scan_request", _spy)
    scanner = bluesky_scanner.BlueskyScanner.__new__(bluesky_scanner.BlueskyScanner)
    scanner._optimization_loader = None
    scanner._experiment_dir = "TestExp"
    scanner._scan_request = None
    scanner._request_resolver = None
    scanner._completed_shots = 0
    scanner._total_shots = 0
    scanner._total_steps = 0
    scanner._scan_number = None

    resolver = _resolver()
    request = _request()
    assert scanner.reinitialize(request, resolver=resolver) is True
    assert calls == [(request, resolver)]
