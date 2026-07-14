"""On-demand ActionPlan execution (G-actions v1) — hermetic.

Pins the engine half of the G-actions contract:

- ``GeecsSession.run_action`` resolves a named plan (fail-fast, nested
  ``run`` references included), prefetches/connects every signal pre-run,
  and executes the compiled steps in order on the session RunEngine.
- The v1 during-scan behavior is refusal with the **exact** message
  ``"scan in progress — action not started"`` — the GUI surfaces it
  verbatim, so the string is pinned character-for-character here (session
  and bridge).
- ``GeecsSession.describe_action`` is the pure dry-run: flattened step
  summaries, zero signal creation, zero connects.
- ``BlueskyScanner.run_action`` / ``describe_action`` are thin delegations
  (the GUI contract mirrored by the console's Submitter protocol).

The richer pause/decide/resume during-scan flow is issue #552 — out of
scope; these tests pin the refusal seam it will replace.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("aioca")  # session is CA-only

from ophyd_async.core import soft_signal_rw  # noqa: E402

from geecs_bluesky.exceptions import (  # noqa: E402
    ActionPlanCycleError,
    ActionPlanNotFoundError,
    GeecsConfigurationError,
)
from geecs_bluesky.session import GeecsSession  # noqa: E402
from geecs_bluesky.scanner_bridge.bluesky_scanner import BlueskyScanner  # noqa: E402
from geecs_schemas.action_plan import ActionPlan  # noqa: E402

REFUSAL = "scan in progress — action not started"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _LibraryResolver:
    """Minimal in-memory ConfigResolver over a named-plan dict."""

    def __init__(self, plans: dict[str, ActionPlan]) -> None:
        self._plans = plans

    def resolve_action_plan(self, name: str) -> ActionPlan:
        try:
            return self._plans[name]
        except KeyError:
            raise GeecsConfigurationError(
                f"action plan {name!r} is not in the action library. "
                f"Known plans: {sorted(self._plans)}"
            ) from None

    def action_plan_registry(self) -> dict[str, ActionPlan]:
        return dict(self._plans)


class _AutoSignalFactory:
    """Recording SettableFactory: soft signals created/connected on demand.

    Mirrors the production ``CaActionSignalFactory`` shape (lazy creation,
    per-key cache, connect at creation through the session's RE loop) so
    ``run_action``'s prefetch pass drives it exactly like the real thing.
    """

    def __init__(self, run_engine: Any) -> None:
        self._re = run_engine
        self._signals: dict[tuple[str, str], Any] = {}
        self._datatypes: dict[tuple[str, str], tuple[type, Any]] = {}
        self.connected: list[tuple[str, str]] = []
        self.disconnect_calls = 0

    def preset(self, device: str, variable: str, datatype: type, initial: Any) -> None:
        """Declare a non-float signal's dtype/initial ahead of creation."""
        self._datatypes[(device, variable)] = (datatype, initial)

    def _get(self, device: str, variable: str) -> Any:
        key = (device, variable)
        if key not in self._signals:
            datatype, initial = self._datatypes.get(key, (float, 0.0))
            signal = soft_signal_rw(datatype, initial, name=f"{device}-{variable}")
            asyncio.run_coroutine_threadsafe(signal.connect(), self._re._loop).result(
                timeout=10.0
            )
            self._signals[key] = signal
            self.connected.append(key)
        return self._signals[key]

    def get_settable(self, device: str, variable: str) -> Any:
        return self._get(device, variable)

    def get_readable(self, device: str, variable: str) -> Any:
        return self._get(device, variable)

    def value_of(self, device: str, variable: str) -> Any:
        signal = self._signals[(device, variable)]
        return asyncio.run_coroutine_threadsafe(
            signal.get_value(), self._re._loop
        ).result(timeout=10.0)

    async def disconnect(self) -> None:
        # Production drops its cache here; the fake keeps signals readable
        # so tests can assert final values after the run.
        self.disconnect_calls += 1


def _plan(steps: list[dict]) -> ActionPlan:
    return ActionPlan.model_validate({"steps": steps})


@pytest.fixture()
def session(monkeypatch) -> tuple[GeecsSession, _AutoSignalFactory]:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    factory = _AutoSignalFactory(s.RE)
    monkeypatch.setattr(s, "action_signal_factory", lambda: factory)
    return s, factory


def _forbidden_factory(monkeypatch, s: GeecsSession) -> None:
    """Pin that a code path never builds (or connects) an action factory."""

    def _explode() -> None:
        raise AssertionError("action_signal_factory must not be called here")

    monkeypatch.setattr(s, "action_signal_factory", _explode)


# ---------------------------------------------------------------------------
# run_action — execution through a mock factory
# ---------------------------------------------------------------------------


def test_run_action_executes_compiled_steps_in_order(session) -> None:
    s, factory = session
    factory.preset("U_PLC", "DO.Ch1", str, "")
    factory.preset("U_PLC", "DI.Ch2", str, "ready")
    plans = {
        "bracket": _plan(
            [
                {"do": "set", "device": "U_S1H", "variable": "Current", "value": 5.0},
                {"do": "wait", "seconds": 0.01},
                {"do": "set", "device": "U_PLC", "variable": "DO.Ch1", "value": "on"},
                {
                    "do": "check",
                    "device": "U_PLC",
                    "variable": "DI.Ch2",
                    "expected": "ready",
                },
            ]
        )
    }

    puts: list[tuple[str, Any]] = []
    s.RE.msg_hook = lambda msg: (
        puts.append((msg.obj.name, msg.args[0])) if msg.command == "set" else None
    )
    s.run_action("bracket", _LibraryResolver(plans))

    assert puts == [("U_S1H-Current", 5.0), ("U_PLC-DO.Ch1", "on")]
    assert factory.value_of("U_S1H", "Current") == 5.0
    assert factory.value_of("U_PLC", "DO.Ch1") == "on"
    # Every set/check target was prefetched (connected) before the run.
    assert factory.connected == [
        ("U_S1H", "Current"),
        ("U_PLC", "DO.Ch1"),
        ("U_PLC", "DI.Ch2"),
    ]
    # The factory rides the per-run cleanup path.
    assert factory.disconnect_calls == 1


def test_run_action_executes_nested_plans(session) -> None:
    """A ``run`` step inlines the nested plan's steps at its position."""
    s, factory = session
    plans = {
        "child": _plan(
            [{"do": "set", "device": "U_Dev", "variable": "VarB", "value": 2.0}]
        ),
        "parent": _plan(
            [
                {"do": "set", "device": "U_Dev", "variable": "VarA", "value": 1.0},
                {"do": "run", "plan": "child"},
                {"do": "set", "device": "U_Dev", "variable": "VarC", "value": 3.0},
            ]
        ),
    }

    puts: list[str] = []
    s.RE.msg_hook = lambda msg: (
        puts.append(msg.obj.name) if msg.command == "set" else None
    )
    s.run_action("parent", _LibraryResolver(plans))

    assert puts == ["U_Dev-VarA", "U_Dev-VarB", "U_Dev-VarC"]


# ---------------------------------------------------------------------------
# run_action — fail-fast (no factory, no connects, no execution)
# ---------------------------------------------------------------------------


def test_run_action_unknown_name_fails_fast(monkeypatch) -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    _forbidden_factory(monkeypatch, s)

    with pytest.raises(GeecsConfigurationError, match="ghost"):
        s.run_action("ghost", _LibraryResolver({}))


def test_run_action_unknown_nested_name_fails_before_any_connect(
    monkeypatch,
) -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    _forbidden_factory(monkeypatch, s)
    plans = {"parent": _plan([{"do": "run", "plan": "ghost"}])}

    with pytest.raises(ActionPlanNotFoundError) as excinfo:
        s.run_action("parent", _LibraryResolver(plans))
    assert excinfo.value.plan_name == "ghost"


def test_run_action_cycle_fails_before_any_connect(monkeypatch) -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    _forbidden_factory(monkeypatch, s)
    plans = {
        "a": _plan([{"do": "run", "plan": "b"}]),
        "b": _plan([{"do": "run", "plan": "a"}]),
    }

    with pytest.raises(ActionPlanCycleError):
        s.run_action("a", _LibraryResolver(plans))


def test_run_action_wraps_unreachable_target_operator_readably(
    session, monkeypatch
) -> None:
    """A connect failure names the action and the (device, variable)."""
    s, factory = session

    def _fail(device: str, variable: str) -> Any:
        raise TimeoutError("no PV")

    monkeypatch.setattr(factory, "get_settable", _fail)
    plans = {
        "bad": _plan(
            [{"do": "set", "device": "U_Gone", "variable": "Var", "value": 1.0}]
        )
    }

    with pytest.raises(GeecsConfigurationError, match=r"'bad'.*U_Gone:Var"):
        s.run_action("bad", _LibraryResolver(plans))
    # The factory still rode the cleanup path.
    assert factory.disconnect_calls == 1


# ---------------------------------------------------------------------------
# Refusal while a scan is in progress — exact message pinned
# ---------------------------------------------------------------------------


def test_session_refuses_when_run_engine_not_idle() -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    s.RE = SimpleNamespace(state="running")  # a scan owns the engine

    class _Untouchable:
        def resolve_action_plan(self, name: str) -> ActionPlan:
            raise AssertionError("refusal must come before resolution")

    with pytest.raises(RuntimeError) as excinfo:
        s.run_action("anything", _Untouchable())
    assert str(excinfo.value) == REFUSAL


def _bare_scanner(*, scanning: bool) -> BlueskyScanner:
    scanner = BlueskyScanner.__new__(BlueskyScanner)
    scanner._scan_thread = SimpleNamespace(is_alive=lambda: True) if scanning else None
    scanner._scan_finished = False
    return scanner


def test_bridge_run_action_refuses_while_scanning_exact_message() -> None:
    scanner = _bare_scanner(scanning=True)
    with pytest.raises(RuntimeError) as excinfo:
        scanner.run_action("anything")
    assert str(excinfo.value) == REFUSAL


def test_bridge_describe_action_works_while_scanning() -> None:
    # The dry-run is pure (zero CA) and is exactly what an operator wants
    # to consult mid-scan — only run_action carries the refusal.
    scanner = _bare_scanner(scanning=True)
    expected = [{"kind": "wait", "wait_s": 1.0}]
    scanner._session = SimpleNamespace(describe_action=lambda name, resolver: expected)
    scanner._action_resolver = lambda: object()
    steps = scanner.describe_action("anything")
    assert steps == expected


# ---------------------------------------------------------------------------
# Bridge delegation (the GUI contract)
# ---------------------------------------------------------------------------


def test_bridge_delegates_to_session_with_its_resolver() -> None:
    scanner = _bare_scanner(scanning=False)
    calls: list[tuple[str, str, Any]] = []
    sentinel_resolver = object()
    scanner._request_resolver = sentinel_resolver
    scanner._session = SimpleNamespace(
        run_action=lambda name, resolver: calls.append(("run", name, resolver)),
        describe_action=lambda name, resolver: (
            calls.append(("describe", name, resolver)) or [{"kind": "wait"}]
        ),
    )

    scanner.run_action("bracket")
    assert scanner.describe_action("bracket") == [{"kind": "wait"}]

    assert calls == [
        ("run", "bracket", sentinel_resolver),
        ("describe", "bracket", sentinel_resolver),
    ]


def test_bridge_builds_configs_repo_resolver_when_none_stored(monkeypatch) -> None:
    from geecs_bluesky.scanner_bridge import bluesky_scanner as module

    scanner = _bare_scanner(scanning=False)
    scanner._request_resolver = None
    scanner._experiment_dir = "Undulator"
    built: list[str] = []

    class _FakeResolver:
        def __init__(self, experiment: str) -> None:
            built.append(experiment)

    monkeypatch.setattr(module, "ConfigsRepoResolver", _FakeResolver)
    scanner._session = SimpleNamespace(run_action=lambda name, resolver: None)

    scanner.run_action("bracket")
    assert built == ["Undulator"]


# ---------------------------------------------------------------------------
# describe_action — pure dry-run, every step kind, zero connects
# ---------------------------------------------------------------------------


def test_describe_action_step_list_shape_and_flattening(monkeypatch) -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    _forbidden_factory(monkeypatch, s)  # pure: no factory, no connects
    plans = {
        "inner": _plan(
            [{"do": "set", "device": "U_Deep", "variable": "VarD", "value": 4}]
        ),
        "child": _plan(
            [
                {"do": "set", "device": "U_Dev", "variable": "VarB", "value": "on"},
                {"do": "wait", "seconds": 3.0},
                {"do": "run", "plan": "inner"},
            ]
        ),
        "top": _plan(
            [
                {"do": "set", "device": "U_S1H", "variable": "Current", "value": 5.0},
                {"do": "wait", "seconds": 1.5},
                {
                    "do": "check",
                    "device": "U_PLC",
                    "variable": "DI.Ch2",
                    "expected": "ready",
                },
                {"do": "run", "plan": "child"},
            ]
        ),
    }

    steps = s.describe_action("top", _LibraryResolver(plans))

    assert steps == [
        {
            "kind": "set",
            "device": "U_S1H",
            "variable": "Current",
            "value": 5.0,
            "wait_s": None,
            "from_plan": None,
        },
        {
            "kind": "wait",
            "device": None,
            "variable": None,
            "value": None,
            "wait_s": 1.5,
            "from_plan": None,
        },
        {
            "kind": "check",
            "device": "U_PLC",
            "variable": "DI.Ch2",
            "value": "ready",
            "wait_s": None,
            "from_plan": None,
        },
        {
            "kind": "set",
            "device": "U_Dev",
            "variable": "VarB",
            "value": "on",
            "wait_s": None,
            "from_plan": "child",
        },
        {
            "kind": "wait",
            "device": None,
            "variable": None,
            "value": None,
            "wait_s": 3.0,
            "from_plan": "child",
        },
        {
            "kind": "set",
            "device": "U_Deep",
            "variable": "VarD",
            "value": 4,
            "wait_s": None,
            "from_plan": "inner",  # innermost enclosing plan wins
        },
    ]


def test_describe_action_unknown_names_fail_fast(monkeypatch) -> None:
    s = GeecsSession("TestExp", tiled=False, mock=True)
    _forbidden_factory(monkeypatch, s)

    with pytest.raises(GeecsConfigurationError, match="ghost"):
        s.describe_action("ghost", _LibraryResolver({}))

    plans = {"top": _plan([{"do": "run", "plan": "ghost"}])}
    with pytest.raises(ActionPlanNotFoundError):
        s.describe_action("top", _LibraryResolver(plans))
