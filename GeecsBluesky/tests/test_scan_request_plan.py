"""Tests for geecs_scan_request_plan — the ScanRequest plan preamble (QS round 1).

Hermetic (ophyd-async mock backends, no gateway, no DB, no network).  Pins
the issue #633 acceptance surface:

- the preamble validates **fail-fast** on a bad request (nothing claimed,
  nothing constructed);
- devices are constructed **inside** the plan (RunEngine already running);
- the scan number is claimed **inside** the plan;
- for representative noscan and step-scan requests, ``RE(geecs_scan_
  request_plan(request))`` produces the same documents/ScanInfo as today's
  ``run_scan_request`` (start-doc metadata, streams/columns, row counts,
  exit status — uids/timestamps/folders normalized).

The mock trigger problem is new here: devices exist only mid-plan, so the
free-run pacer cannot be started up front.  An ``RE.msg_hook`` watches for
``stage`` messages and starts a pacer per staged device that has an
``acq_timestamp`` — works identically for both entry points, since staging
is always inside the run.
"""

from __future__ import annotations

import configparser
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky.utils import RunEngineInterrupted  # noqa: F401  (doc import guard)
from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.config_resolver import ConfigsRepoResolver  # noqa: E402
from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.plans.scan_request_plan import (  # noqa: E402
    geecs_scan_request_plan,
    set_plan_session,
)
from geecs_bluesky.scan_request_runner import run_scan_request  # noqa: E402
from geecs_schemas import ScanRequest  # noqa: E402
from tests.ca_mock_helpers import start_pacer  # noqa: E402
from tests.test_scan_request_runner import (  # noqa: E402
    LEGACY_ACTIONS,
    LEGACY_SAVE_ELEMENT,
    LEGACY_SCAN_DEVICES,
    LEGACY_SHOT_CONTROL,
)

# ---------------------------------------------------------------------------
# Fixtures: configs repo (reusing the runner suite's YAML corpus), sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def configs_root(tmp_path):
    exp = tmp_path / "LegacyExp"
    (exp / "save_devices").mkdir(parents=True)
    (exp / "save_devices" / "UC_Test.yaml").write_text(LEGACY_SAVE_ELEMENT)
    (exp / "shot_control_configurations").mkdir()
    (exp / "shot_control_configurations" / "HTU-Normal.yaml").write_text(
        LEGACY_SHOT_CONTROL
    )
    (exp / "scan_devices").mkdir()
    (exp / "scan_devices" / "scan_devices.yaml").write_text(LEGACY_SCAN_DEVICES)
    (exp / "action_library").mkdir()
    (exp / "action_library" / "actions.yaml").write_text(LEGACY_ACTIONS)
    return tmp_path


@pytest.fixture
def resolver(configs_root):
    return ConfigsRepoResolver("LegacyExp", experiments_root=configs_root)


@pytest.fixture(autouse=True)
def no_db(monkeypatch):
    """Neutralize the DB-backed providers (hermetic: no MySQL, no gateway).

    A real ``GeecsSession`` exposes ``experiment``, so the runner's policy /
    served-set factories would otherwise try the lab DB.  ``None`` policy =
    explicit-scalars-only + no telemetry + unserved check skipped, on BOTH
    entry points — parity is preserved.
    """
    monkeypatch.setattr(
        "geecs_bluesky.scan_request_runner.make_scalar_policy", lambda session: None
    )
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.make_scalar_policy",
        lambda session: None,
    )
    monkeypatch.setattr(
        "geecs_bluesky.scan_request_runner.make_served_set_provider",
        lambda session: None,
    )


@pytest.fixture(autouse=True)
def clear_worker_session():
    yield
    set_plan_session(None)


def _mock_session():
    from geecs_bluesky.session import GeecsSession

    return GeecsSession("LegacyExp", tiled=False, mock=True)


def _noscan_request(**overrides) -> ScanRequest:
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="free_run",
        save_sets=["UC_Test"],
        description="stats",
    )
    base.update(overrides)
    return ScanRequest.model_validate(base)


# ---------------------------------------------------------------------------
# Mock pacing + document collection
# ---------------------------------------------------------------------------


def _install_stage_pacer(session, pacers: list) -> None:
    """Start a mock-trigger pacer for every staged acq_timestamp device.

    Devices are constructed inside the runner/plan, so the pacer cannot be
    wired up front; ``stage`` is the first message that carries the device
    object after it is connected.
    """
    paced: set = set()

    def hook(msg) -> None:
        obj = msg.obj
        if (
            msg.command == "stage"
            and obj not in paced
            and hasattr(obj, "acq_timestamp")
        ):
            paced.add(obj)
            set_mock_value(obj.acq_timestamp, 1000.0)
            pacers.append(
                start_pacer(
                    session.RE,
                    [(obj, 1000.0)],
                    initial_delay=1.0,
                    interval=0.15,
                )
            )

    session.RE.msg_hook = hook


class _DocCollector:
    def __init__(self) -> None:
        self.start: dict | None = None
        self.stop: dict | None = None
        self.descriptors: list[dict] = []
        self.events: list[dict] = []

    def __call__(self, name: str, doc: dict) -> None:
        if name == "start":
            self.start = dict(doc)
        elif name == "stop":
            self.stop = dict(doc)
        elif name == "descriptor":
            self.descriptors.append(dict(doc))
        elif name == "event":
            self.events.append(dict(doc))


def _run_scan(entry_point, request, resolver, folder, monkeypatch):
    """Run *request* through one entry point; return the collected documents.

    ``entry_point`` is ``"runner"`` (today's ``run_scan_request``) or
    ``"plan"`` (``RE(geecs_scan_request_plan(request.model_dump()))`` with
    the worker-default session — the queue's exact call shape).  The claim
    is stubbed to scan 7 in *folder* at each entry point's claim site.
    """
    folder.mkdir(parents=True, exist_ok=True)
    session = _mock_session()
    docs = _DocCollector()
    token = session.RE.subscribe(docs)
    pacers: list = []
    _install_stage_pacer(session, pacers)

    def claim(experiment):
        assert experiment == "LegacyExp"
        return 7, str(folder)

    try:
        if entry_point == "runner":
            monkeypatch.setattr("geecs_bluesky.session.claim_scan_number", claim)
            run_scan_request(session, request, resolver)
        else:
            monkeypatch.setattr(
                "geecs_bluesky.plans.scan_request_plan.claim_scan_number", claim
            )
            set_plan_session(session)
            session.RE(geecs_scan_request_plan(request.model_dump(), resolver=resolver))
    finally:
        for pacer in pacers:
            pacer.cancel()
        session.RE.unsubscribe(token)
        session.RE.msg_hook = None
    return docs


# ---------------------------------------------------------------------------
# Document normalization / comparison
# ---------------------------------------------------------------------------

_VOLATILE_START_KEYS = {"uid", "time"}


def _normalize(value, folder):
    if isinstance(value, str):
        return value.replace(str(folder), "<scan_folder>")
    if isinstance(value, dict):
        return {k: _normalize(v, folder) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v, folder) for v in value]
    return value


def _start_essence(doc: dict, folder) -> dict:
    return {
        k: _normalize(v, folder)
        for k, v in doc.items()
        if k not in _VOLATILE_START_KEYS
    }


def _descriptor_essence(docs: _DocCollector) -> list[tuple]:
    return sorted(
        (d["name"], sorted(d["data_keys"]), sorted(d.get("object_keys", {})))
        for d in docs.descriptors
    )


def _event_essence(docs: _DocCollector) -> list[tuple[int, list[str]]]:
    return [(e["seq_num"], sorted(e["data"])) for e in docs.events]


def _assert_same_run(docs_a: _DocCollector, docs_b: _DocCollector, folder_a, folder_b):
    assert docs_a.start is not None and docs_b.start is not None
    assert _start_essence(docs_a.start, folder_a) == _start_essence(
        docs_b.start, folder_b
    )
    assert _descriptor_essence(docs_a) == _descriptor_essence(docs_b)
    assert _event_essence(docs_a) == _event_essence(docs_b)
    assert docs_a.stop is not None and docs_b.stop is not None
    assert docs_a.stop["exit_status"] == docs_b.stop["exit_status"] == "success"
    assert docs_a.stop["num_events"] == docs_b.stop["num_events"]


# ---------------------------------------------------------------------------
# Fail-fast + in-plan pins (issue #633 acceptance)
# ---------------------------------------------------------------------------


def test_bad_request_fails_in_the_preamble_and_claims_nothing(
    resolver, monkeypatch
) -> None:
    """Authoritative preamble validation: unknown name → error, no claim."""
    session = _mock_session()
    claims: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    request = _noscan_request(save_sets=["Nope"]).model_dump()
    with pytest.raises(GeecsConfigurationError, match="Nope"):
        session.RE(geecs_scan_request_plan(request, session=session, resolver=resolver))
    assert claims == [], "a failed validation must burn no scan number"


def test_creating_the_generator_does_no_work(resolver, monkeypatch) -> None:
    """The preamble runs when the RE iterates the plan, not at call time."""
    session = _mock_session()
    claims: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    resolved: list = []
    original = type(resolver).resolve_save_set

    def counting(self, name):
        resolved.append(name)
        return original(self, name)

    monkeypatch.setattr(type(resolver), "resolve_save_set", counting)
    plan = geecs_scan_request_plan(
        _noscan_request(save_sets=["Nope"]).model_dump(),
        session=session,
        resolver=resolver,
    )
    assert resolved == [] and claims == []
    with pytest.raises(GeecsConfigurationError):
        session.RE(plan)
    assert resolved == ["Nope"]


def test_strict_without_trigger_profile_refuses_before_claim(
    resolver, monkeypatch
) -> None:
    session = _mock_session()
    claims: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    request = _noscan_request(acquisition="strict").model_dump()
    with pytest.raises(GeecsConfigurationError, match="strict_shot_control"):
        session.RE(geecs_scan_request_plan(request, session=session, resolver=resolver))
    assert claims == []


def test_optimize_mode_without_a_loader_is_refused_loudly_after_validation(
    resolver, monkeypatch
) -> None:
    """No loader registered (W2): validated, then refused — never silently."""
    from geecs_bluesky.plans import scan_request_plan as srp

    session = _mock_session()
    claims: list = []
    # The optimize path claims via claim_scan (NOT claim_scan_number, which
    # is the noscan/step path's claim function) — patching the wrong name
    # here would make this tripwire pass vacuously even if the no-loader
    # refusal moved to after the claim (row 4, PR #644 review).
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    monkeypatch.setattr(srp, "_optimization_loader", None)
    request = ScanRequest.model_validate(
        dict(
            mode="optimize",
            shots_per_step=5,
            acquisition="free_run",
            save_sets=["UC_Test"],
            optimization={
                "variables": {"jet_z": [0.0, 1.0]},
                "objectives": {"counts": "MAXIMIZE"},
                "evaluator": {"module": "m", "class": "C"},
                "generator": {"name": "bayes_default"},
                "max_iterations": 3,
            },
        )
    )
    with pytest.raises(GeecsConfigurationError, match="set_optimization_loader"):
        session.RE(
            geecs_scan_request_plan(
                request.model_dump(), session=session, resolver=resolver
            )
        )
    assert claims == []
    # An unknown optimize VOCS name still fails *validation* first — the
    # authoritative preamble runs before the refusal.
    bad = request.model_copy(
        update={
            "optimization": request.optimization.model_copy(
                update={"variables": {"nope": [0.0, 1.0]}}
            )
        }
    )
    with pytest.raises(GeecsConfigurationError, match="nope"):
        session.RE(
            geecs_scan_request_plan(
                bad.model_dump(), session=session, resolver=resolver
            )
        )


def test_optimize_mode_reaches_a_registered_loader_and_runs_the_bins(
    resolver, monkeypatch
) -> None:
    """With a loader registered, the request reaches it and its objective/
    suggester drive the run — the W2 seam close (issue #640)."""
    import geecs_bluesky.plans.scan_request_plan as srp
    from geecs_bluesky.optimize import BinData

    session = _mock_session()
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan",
        lambda experiment: (None, None),
    )

    bind_calls: list[dict] = []
    finish_calls: list[bool] = []

    class ScriptedSuggester:
        def __init__(self) -> None:
            self._points = [{"jet_z": 0.1}, {"jet_z": 0.4}]
            self.observed: list[tuple] = []

        def suggest(self):
            return self._points.pop(0) if self._points else None

        def observe(self, inputs, objective_value, bin_data):
            self.observed.append((inputs, objective_value, bin_data))

    suggester = ScriptedSuggester()

    def objective(bin_data: BinData) -> float:
        # A real (non-stub) value derived from the bin's actual collected
        # rows — proves the bridge's objective ran against real event data
        # threaded through by the plan, not a canned return.
        return float(len(bin_data.rows))

    pacers: list = []

    class FakeBridge:
        device_requirements = None

        def bind(self, devices, scan_tag, scan_folder):
            bind_calls.append(
                {
                    "devices": devices,
                    "scan_tag": scan_tag,
                    "scan_folder": scan_folder,
                }
            )
            # geecs_adaptive_scan's t0 sync runs before any staging message —
            # devices are already connected by this point (bind() is called
            # after in-plan connects, before the adaptive-scan plan starts),
            # so this is the last hook available to seed acq_timestamp before
            # geecs_t0_sync reads it.
            for device in devices:
                if hasattr(device, "acq_timestamp"):
                    set_mock_value(device.acq_timestamp, 1000.0)
                    pacers.append(
                        start_pacer(
                            session.RE,
                            [(device, 1000.0)],
                            initial_delay=1.0,
                            interval=0.15,
                        )
                    )
            return objective, suggester

        def finish(self):
            finish_calls.append(True)

    loader_calls: list = []

    def fake_loader(spec):
        loader_calls.append(spec)
        return FakeBridge()

    monkeypatch.setattr(srp, "_optimization_loader", fake_loader)
    request = ScanRequest.model_validate(
        dict(
            mode="optimize",
            shots_per_step=2,
            acquisition="free_run",
            save_sets=["UC_Test"],
            optimization={
                "variables": {"jet_z": [0.0, 1.0]},
                "objectives": {"counts": "MAXIMIZE"},
                "evaluator": {"module": "m", "class": "C"},
                "generator": {"name": "bayes_default"},
                "max_iterations": 5,
            },
        )
    )
    docs = _DocCollector()
    token = session.RE.subscribe(docs)
    try:
        session.RE(
            geecs_scan_request_plan(
                request.model_dump(), session=session, resolver=resolver
            )
        )
    finally:
        for pacer in pacers:
            pacer.cancel()
        session.RE.unsubscribe(token)

    assert len(loader_calls) == 1
    assert loader_calls[0].model_dump() == request.optimization.model_dump()
    assert len(bind_calls) == 1
    assert bind_calls[0]["scan_tag"] is None and bind_calls[0]["scan_folder"] is None
    # The suggester's two scripted points both ran, and observed a real
    # objective value computed from the bin's actual collected rows — proof
    # the binder's objective/suggester threaded into the run, not a stub.
    assert [inputs for inputs, _, _ in suggester.observed] == [
        {"jet_z": 0.1},
        {"jet_z": 0.4},
    ]
    assert [value for _, value, _ in suggester.observed] == [
        pytest.approx(request.shots_per_step),
        pytest.approx(request.shots_per_step),
    ]
    assert finish_calls == [True]
    assert docs.start is not None and docs.start["plan_name"] == "geecs_adaptive_scan"
    assert docs.stop is not None and docs.stop["exit_status"] == "success"


def test_optimize_empty_effective_device_set_after_preflight_refused_pre_claim(
    resolver, monkeypatch
) -> None:
    """PR #644 review row 3: an unserved-preflight that drops every device
    must refuse *before* the claim — not fall through to
    ``reference = detectors[0]`` after the claim and blow up with an
    unrelated ``IndexError`` (burning a scan number for nothing).
    """
    import geecs_bluesky.plans.scan_request_plan as srp
    import geecs_bluesky.scan_request_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "make_served_set_provider",
        lambda session: SimpleNamespace(served_by_device=lambda: {}),
    )
    monkeypatch.setattr(
        srp,
        "_optimization_loader",
        lambda spec: SimpleNamespace(device_requirements=None),
    )
    claims: list = []
    monkeypatch.setattr(
        srp,
        "claim_scan",
        lambda experiment: claims.append(experiment) or (None, None),
    )
    session = _mock_session()
    request = ScanRequest.model_validate(
        dict(
            mode="optimize",
            shots_per_step=2,
            acquisition="free_run",
            save_sets=["UC_Test"],
            optimization={
                "variables": {"jet_z": [0.0, 1.0]},
                "objectives": {"counts": "MAXIMIZE"},
                "evaluator": {"module": "m", "class": "C"},
                "generator": {"name": "bayes_default"},
                "max_iterations": 3,
            },
        )
    )
    with pytest.raises(
        GeecsConfigurationError, match="unserved-variables pre-flight dropped"
    ):
        session.RE(
            geecs_scan_request_plan(
                request.model_dump(), session=session, resolver=resolver
            )
        )
    assert claims == [], "the preflight refusal must burn no scan number"


# ---------------------------------------------------------------------------
# Optimize mode: on_finish ladder, history persistence, metadata parity
# (PR #644 review rows 1, 2, 5, 7)
# ---------------------------------------------------------------------------


def _optimize_request(
    *, move_to_best_on_finish: bool = False, max_iterations: int = 5
) -> ScanRequest:
    return ScanRequest.model_validate(
        dict(
            mode="optimize",
            shots_per_step=2,
            acquisition="free_run",
            save_sets=["UC_Test"],
            optimization={
                "variables": {"jet_z": [0.0, 1.0]},
                "objectives": {"counts": "MAXIMIZE"},
                "evaluator": {"module": "m", "class": "C"},
                "generator": {"name": "bayes_default"},
                "max_iterations": max_iterations,
                "move_to_best_on_finish": move_to_best_on_finish,
            },
        )
    )


class _ScriptedSuggester:
    def __init__(self, points: list[dict]) -> None:
        self._points = list(points)
        self.observed: list[tuple] = []

    def suggest(self):
        return self._points.pop(0) if self._points else None

    def observe(self, inputs, objective_value, bin_data):
        self.observed.append((inputs, objective_value, bin_data))


class _RaisingSuggester(_ScriptedSuggester):
    """Scripted points, then a crash — simulates a suggester/operator abort
    mid-optimization (on_finish restore-on-failure tests)."""

    def __init__(self, points: list[dict], fail_after: int) -> None:
        super().__init__(points)
        self._fail_after = fail_after
        self._calls = 0

    def suggest(self):
        self._calls += 1
        if self._calls > self._fail_after:
            raise RuntimeError("suggester exploded")
        return super().suggest()


def _track_mv_and_rd(monkeypatch):
    """Wrap ``bluesky.plan_stubs.mv``/``rd`` to record calls without
    changing behavior — ``bps.mv``/``bps.rd`` are shared module functions
    used by both the plan body and ``geecs_adaptive_scan``, so this must
    wrap-and-delegate rather than replace.
    """
    real_mv = bps.mv
    real_rd = bps.rd
    mv_calls: list[tuple] = []
    rd_calls: list[tuple] = []

    def mv_wrapper(*args, **kwargs):
        mv_calls.append(args)
        return (yield from real_mv(*args, **kwargs))

    def rd_wrapper(obj, *args, **kwargs):
        value = yield from real_rd(obj, *args, **kwargs)
        rd_calls.append((obj, value))
        return value

    monkeypatch.setattr(bps, "mv", mv_wrapper)
    monkeypatch.setattr(bps, "rd", rd_wrapper)
    return mv_calls, rd_calls


def _run_optimize_with_bridge(
    session,
    resolver,
    request,
    monkeypatch,
    *,
    suggester,
    objective,
    scan_folder=None,
    finish_calls: list | None = None,
    bind_calls: list | None = None,
):
    """Drive ``geecs_scan_request_plan`` through optimize mode with a
    scripted bridge; returns ``(docs, finish_calls, bind_calls)``.

    *finish_calls*/*bind_calls* may be passed in by the caller so they stay
    inspectable even when the run raises (an abort/failure test cannot rely
    on this function's return value).
    """
    import geecs_bluesky.plans.scan_request_plan as srp

    if finish_calls is None:
        finish_calls = []
    if bind_calls is None:
        bind_calls = []
    monkeypatch.setattr(
        srp,
        "claim_scan",
        lambda experiment: (
            None,
            str(scan_folder) if scan_folder is not None else None,
        ),
    )
    pacers: list = []

    class FakeBridge:
        device_requirements = None

        def bind(self, devices, scan_tag, scan_folder):
            bind_calls.append(
                {"devices": devices, "scan_tag": scan_tag, "scan_folder": scan_folder}
            )
            for device in devices:
                if hasattr(device, "acq_timestamp"):
                    set_mock_value(device.acq_timestamp, 1000.0)
                    pacers.append(
                        start_pacer(
                            session.RE,
                            [(device, 1000.0)],
                            initial_delay=1.0,
                            interval=0.15,
                        )
                    )
            return objective, suggester

        def finish(self):
            finish_calls.append(True)

    monkeypatch.setattr(srp, "_optimization_loader", lambda spec: FakeBridge())

    docs = _DocCollector()
    token = session.RE.subscribe(docs)
    try:
        session.RE(
            geecs_scan_request_plan(
                request.model_dump(), session=session, resolver=resolver
            )
        )
    finally:
        for pacer in pacers:
            pacer.cancel()
        session.RE.unsubscribe(token)
        session.RE.msg_hook = None
    return docs, finish_calls, bind_calls


def test_optimize_on_finish_best_moves_to_the_best_observed_inputs(
    resolver, monkeypatch
) -> None:
    """PR #644 review row 1: move_to_best_on_finish selects the
    best-observed inputs across the whole run, not just the last suggested
    point."""
    session = _mock_session()
    mv_calls, _rd_calls = _track_mv_and_rd(monkeypatch)
    suggester = _ScriptedSuggester([{"jet_z": 0.1}, {"jet_z": 0.4}, {"jet_z": 0.7}])
    values = iter([1.0, 5.0, 2.0])  # best is the *middle* point, not the last

    def objective(bin_data):
        return next(values)

    request = _optimize_request(move_to_best_on_finish=True)
    docs, finish_calls, _bind_calls = _run_optimize_with_bridge(
        session,
        resolver,
        request,
        monkeypatch,
        suggester=suggester,
        objective=objective,
    )
    assert docs.stop is not None and docs.stop["exit_status"] == "success"
    assert len(mv_calls) == 4  # 3 suggested moves + the final move-to-best
    assert mv_calls[-1][1] == pytest.approx(0.4)
    assert finish_calls == [True]


def test_optimize_on_finish_best_falls_back_to_initial_when_no_finite_objective(
    resolver, monkeypatch, caplog
) -> None:
    """PR #644 review row 1: if every objective evaluation came back
    non-finite, on_finish=best falls back to the initial values (with a
    WARNING) instead of silently moving to nan."""
    import logging

    session = _mock_session()
    mv_calls, rd_calls = _track_mv_and_rd(monkeypatch)
    suggester = _ScriptedSuggester([{"jet_z": 0.1}, {"jet_z": 0.4}])

    def objective(bin_data):
        raise RuntimeError("evaluator is down")

    request = _optimize_request(move_to_best_on_finish=True)
    with caplog.at_level(logging.WARNING):
        docs, finish_calls, _bind_calls = _run_optimize_with_bridge(
            session,
            resolver,
            request,
            monkeypatch,
            suggester=suggester,
            objective=objective,
        )
    assert docs.stop is not None and docs.stop["exit_status"] == "success"
    initial_value = rd_calls[0][1]
    assert mv_calls[-1][1] == pytest.approx(initial_value)
    assert any("no finite objectives" in r.message for r in caplog.records)
    assert finish_calls == [True]


def test_optimize_on_finish_best_restores_initial_on_failure_and_skips_finish(
    resolver, monkeypatch, tmp_path
) -> None:
    """PR #644 review rows 1 + 5 (negative): a hard failure/abort restores
    the *initial* values (never best-so-far) before re-raising, writes no
    optimization.json, and never calls the bridge's finish()."""
    session = _mock_session()
    mv_calls, rd_calls = _track_mv_and_rd(monkeypatch)
    suggester = _RaisingSuggester([{"jet_z": 0.1}], fail_after=1)

    def objective(bin_data):
        return 5.0

    scan_folder = tmp_path / "Scan007"
    scan_folder.mkdir()
    finish_calls: list = []
    request = _optimize_request(move_to_best_on_finish=True)
    with pytest.raises(RuntimeError, match="suggester exploded"):
        _run_optimize_with_bridge(
            session,
            resolver,
            request,
            monkeypatch,
            suggester=suggester,
            objective=objective,
            scan_folder=scan_folder,
            finish_calls=finish_calls,
        )
    initial_value = rd_calls[0][1]
    assert mv_calls[-1][1] == pytest.approx(initial_value)
    assert not (scan_folder / "optimization.json").exists()
    assert finish_calls == []


def test_optimize_on_finish_hold_never_moves(resolver, monkeypatch) -> None:
    """PR #644 review row 5: on_finish left unset (move_to_best_on_finish
    False) means the plan never issues an extra move, in any outcome — the
    variables stay wherever the suggester last set them."""
    session = _mock_session()
    mv_calls, _rd_calls = _track_mv_and_rd(monkeypatch)
    suggester = _ScriptedSuggester([{"jet_z": 0.1}, {"jet_z": 0.4}])

    def objective(bin_data):
        return 5.0

    request = _optimize_request(move_to_best_on_finish=False)
    docs, finish_calls, _bind_calls = _run_optimize_with_bridge(
        session,
        resolver,
        request,
        monkeypatch,
        suggester=suggester,
        objective=objective,
    )
    assert docs.stop is not None and docs.stop["exit_status"] == "success"
    assert len(mv_calls) == 2  # exactly the 2 suggested moves, no extra move
    assert mv_calls[-1][1] == pytest.approx(0.4)
    assert finish_calls == [True]


def test_optimize_persists_history_to_optimization_json(
    resolver, monkeypatch, tmp_path
) -> None:
    """PR #644 review row 2: optimization.json persistence is gated on a
    real scan folder + non-empty history — exercised here with an actual
    tmp folder (the ``claim_scan`` stub used elsewhere in this file returns
    ``(None, None)``, which never reaches this code path at all)."""
    session = _mock_session()
    scan_folder = tmp_path / "Scan007"
    scan_folder.mkdir()
    suggester = _ScriptedSuggester([{"jet_z": 0.1}, {"jet_z": 0.4}])
    values = iter([1.0, 5.0])

    def objective(bin_data):
        return next(values)

    request = _optimize_request(move_to_best_on_finish=False)
    docs, _finish_calls, _bind_calls = _run_optimize_with_bridge(
        session,
        resolver,
        request,
        monkeypatch,
        suggester=suggester,
        objective=objective,
        scan_folder=scan_folder,
    )
    assert docs.stop is not None and docs.stop["exit_status"] == "success"
    history_path = scan_folder / "optimization.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert [h["inputs"] for h in history] == [{"jet_z": 0.1}, {"jet_z": 0.4}]
    assert [h["objective"] for h in history] == [pytest.approx(1.0), pytest.approx(5.0)]


def test_optimize_mode_records_db_scan_runtime_metadata(
    configs_root, resolver, monkeypatch
) -> None:
    """PR #644 review row 7: metadata parity between the plan's optimize
    body and the binder path's ``_run_optimize_request`` — ``db_scan_runtime``
    must land in the optimize start doc too, not just the noscan/step one.
    No existing test on either entry point asserted this positively before.
    """
    session = _mock_session()
    suggester = _ScriptedSuggester([{"jet_z": 0.1}])

    def objective(bin_data):
        return 1.0

    request = _optimize_request(move_to_best_on_finish=False)
    docs, _finish_calls, _bind_calls = _run_optimize_with_bridge(
        session,
        resolver,
        request,
        monkeypatch,
        suggester=suggester,
        objective=objective,
    )
    assert docs.start is not None
    assert docs.start["db_scan_runtime"] == {
        "db_scalars": "applied",
        "background_telemetry": "not_run_in_optimize",
    }


def test_optimize_mode_threads_applied_defaults_into_run_wrapper_metadata(
    configs_root, resolver, monkeypatch
) -> None:
    """PR #644 review row 7: ``applied_defaults`` provenance must reach
    ``geecs_run_wrapper``'s ``extra_md`` on the optimize path the same way
    it reaches ``build_step_scan_spec``'s ``md`` on the noscan/step path.

    This spies on the ``extra_md`` handed to ``geecs_run_wrapper`` rather
    than asserting on the real start doc, and sanitizes the dotted
    ``"actions.setup"`` key before letting the (otherwise unmodified) call
    proceed. An actions-based default is the only hermetic way to populate
    ``applied_defaults`` in optimize mode — a ``trigger_profile`` default
    would construct a real, network-backed ``ShotController`` that a mock
    session cannot satisfy — but real ``event_model.compose_run`` schema
    validation (only exercised here, never by the ``_FakeSession``
    binder-path tests) rejects any top-level start-doc key containing
    ``.``/``/``, so running the dotted key through unmodified would fail
    for a reason unrelated to what this test checks. That rejection is a
    real, pre-existing gap shared by both plan paths (`apply_experiment_
    defaults` in ``scan_request_runner.py`` is what produces the dotted
    key, for both step/noscan and optimize) — flagged to the coordinator
    separately rather than fixed here, since sanitizing it one-sidedly on
    the optimize path would itself break the parity this test checks.
    """
    import geecs_bluesky.plans.scan_request_plan as srp

    (configs_root / "LegacyExp" / "experiment_defaults.yaml").write_text(
        "actions:\n  setup: [close_shutters]\n"
    )
    session = _mock_session()
    suggester = _ScriptedSuggester([{"jet_z": 0.1}])

    def objective(bin_data):
        return 1.0

    request = _optimize_request(move_to_best_on_finish=False)

    captured_extra_md: dict = {}
    real_geecs_run_wrapper = srp.geecs_run_wrapper
    real_geecs_adaptive_scan = srp.geecs_adaptive_scan

    def _sanitize(md: dict) -> dict:
        # ``applied_defaults`` keys are dotted (e.g. "actions.setup") by
        # convention (see apply_experiment_defaults in scan_request_runner.py)
        # — real event-model schema validation rejects dots in ANY top-level
        # start-doc key, and validates ``applied_defaults``'s own keys under
        # that same rule. Only the nested keys need sanitizing here.
        sanitized = dict(md)
        if "applied_defaults" in sanitized:
            sanitized["applied_defaults"] = {
                k.replace(".", "_"): v for k, v in sanitized["applied_defaults"].items()
            }
        return sanitized

    def spy_geecs_run_wrapper(plan, **kwargs):
        extra_md = kwargs.get("extra_md") or {}
        captured_extra_md.update(extra_md)
        kwargs["extra_md"] = _sanitize(extra_md)
        return real_geecs_run_wrapper(plan, **kwargs)

    def spy_geecs_adaptive_scan(**kwargs):
        md = kwargs.get("md") or {}
        captured_extra_md.update(md)
        kwargs["md"] = _sanitize(md)
        return real_geecs_adaptive_scan(**kwargs)

    monkeypatch.setattr(srp, "geecs_run_wrapper", spy_geecs_run_wrapper)
    monkeypatch.setattr(srp, "geecs_adaptive_scan", spy_geecs_adaptive_scan)

    docs, _finish_calls, _bind_calls = _run_optimize_with_bridge(
        session,
        resolver,
        request,
        monkeypatch,
        suggester=suggester,
        objective=objective,
    )
    assert docs.start is not None
    assert captured_extra_md["applied_defaults"] == {
        "actions.setup": ["close_shutters"]
    }


def test_no_session_configured_is_a_clear_error(resolver) -> None:
    from bluesky import RunEngine

    plan = geecs_scan_request_plan(_noscan_request().model_dump(), resolver=resolver)
    with pytest.raises(GeecsConfigurationError, match="set_plan_session"):
        RunEngine(context_managers=[])(plan)


def test_devices_and_claim_happen_inside_the_running_plan(
    resolver, monkeypatch, tmp_path
) -> None:
    """The relocation pins: construction and the claim are plan-time events."""
    from geecs_bluesky.session import GeecsSession

    session = _mock_session()
    folder = tmp_path / "Scan007"
    folder.mkdir()

    factory_states: list[str] = []
    real_detector = GeecsSession.detector

    def recording_detector(self, *args, **kwargs):
        factory_states.append(session.RE.state)
        return real_detector(self, *args, **kwargs)

    monkeypatch.setattr(GeecsSession, "detector", recording_detector)

    claim_states: list[str] = []

    def claim(experiment):
        claim_states.append(session.RE.state)
        return 7, str(folder)

    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number", claim
    )
    pacers: list = []
    _install_stage_pacer(session, pacers)
    try:
        session.RE(
            geecs_scan_request_plan(
                _noscan_request().model_dump(), session=session, resolver=resolver
            )
        )
    finally:
        for pacer in pacers:
            pacer.cancel()
        session.RE.msg_hook = None
    assert factory_states == ["running"], "detector must be constructed in-plan"
    assert claim_states == ["running"], "scan number must be claimed in-plan"
    # The claim boundary held: ScanInfo + scan.log landed in the folder.
    assert (folder / "ScanInfoScan007.ini").exists()
    assert (folder / "scan.log").exists()


def test_trigger_profile_builds_the_controller_worker_side(
    resolver, monkeypatch, tmp_path
) -> None:
    """The ShotController is constructed inside the plan from the JSON names.

    A stub controller class stands in (mock CaPutSetter puts would need a
    gateway); the pin is construction-from-writes with the session's
    experiment/rep-rate, plus the mock-session skip of the reachability
    check — the writes themselves are the runner-suite-pinned adapter
    output.
    """
    session = _mock_session()
    built: list = []

    class _StubController:
        def __init__(self, writes) -> None:
            self.writes = writes
            self.connected = False

        @classmethod
        def from_writes(cls, writes, *, experiment=None, rep_rate_hz=1.0):
            controller = cls(writes)
            built.append((controller, experiment, rep_rate_hz))
            return controller

        async def connect_setters(self, timeout: float = 2.0) -> None:
            self.connected = True

        def require_strict_single_shot(self) -> None:  # pragma: no cover
            pass

        def arm(self):
            yield from ()

        def disarm(self):
            yield from ()

        def quiesce(self):
            yield from ()

    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.ShotController", _StubController
    )
    folder = tmp_path / "Scan007"
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: (7, str(folder)) if folder.mkdir() is None else None,
    )
    pacers: list = []
    _install_stage_pacer(session, pacers)
    try:
        session.RE(
            geecs_scan_request_plan(
                _noscan_request(trigger_profile="HTU-Normal").model_dump(),
                session=session,
                resolver=resolver,
            )
        )
    finally:
        for pacer in pacers:
            pacer.cancel()
        session.RE.msg_hook = None
    assert len(built) == 1
    controller, experiment, rep_rate = built[0]
    assert experiment == "LegacyExp" and rep_rate == session.rep_rate_hz
    assert controller.writes.defines_state("SCAN")
    assert not controller.connected, "mock sessions skip the CA reachability check"


# ---------------------------------------------------------------------------
# Document parity with run_scan_request (the acceptance contract)
# ---------------------------------------------------------------------------


def test_noscan_documents_match_run_scan_request(
    resolver, monkeypatch, tmp_path
) -> None:
    request = _noscan_request()
    docs_runner = _run_scan(
        "runner", request, resolver, tmp_path / "runner" / "Scan007", monkeypatch
    )
    docs_plan = _run_scan(
        "plan", request, resolver, tmp_path / "plan" / "Scan007", monkeypatch
    )
    _assert_same_run(
        docs_runner,
        docs_plan,
        tmp_path / "runner" / "Scan007",
        tmp_path / "plan" / "Scan007",
    )
    assert docs_plan.start["scan_number"] == 7
    assert docs_plan.start["scan_request_mode"] == "noscan"
    # Both entry points wrote the same legacy ScanInfo ini.
    ini_a = (tmp_path / "runner" / "Scan007" / "ScanInfoScan007.ini").read_text()
    ini_b = (tmp_path / "plan" / "Scan007" / "ScanInfoScan007.ini").read_text()
    assert ini_a == ini_b
    parser = configparser.ConfigParser()
    parser.read_string(ini_b)
    assert parser["Scan Info"]["scanmode"] == '"noscan"'
    # The plan claimed the number, so the plan owns the per-scan scan.log.
    assert (tmp_path / "plan" / "Scan007" / "scan.log").exists()


def test_step_scan_documents_match_run_scan_request(
    resolver, monkeypatch, tmp_path
) -> None:
    """Representative step scan: one setpoint axis + a request setup action."""
    request = _noscan_request(
        mode="step",
        axes=[{"variable": "jet_x", "positions": {"values": [1.0, 2.0]}}],
        actions={"setup": ["close_shutters"]},
    )
    docs_runner = _run_scan(
        "runner", request, resolver, tmp_path / "runner" / "Scan007", monkeypatch
    )
    docs_plan = _run_scan(
        "plan", request, resolver, tmp_path / "plan" / "Scan007", monkeypatch
    )
    _assert_same_run(
        docs_runner,
        docs_plan,
        tmp_path / "runner" / "Scan007",
        tmp_path / "plan" / "Scan007",
    )
    start = docs_plan.start
    assert start["scan_request_mode"] == "step"
    assert start["scan_variable"] == "jet_x"
    assert start["action_plans"] == {"setup": ["close_shutters"]}
    assert start["num_points"] == 2
    assert docs_plan.stop["num_events"]["primary"] == 4  # 2 positions × 2 shots
    ini_a = (tmp_path / "runner" / "Scan007" / "ScanInfoScan007.ini").read_text()
    ini_b = (tmp_path / "plan" / "Scan007" / "ScanInfoScan007.ini").read_text()
    assert ini_a == ini_b
    parser = configparser.ConfigParser()
    parser.read_string(ini_b)
    assert parser["Scan Info"]["scan parameter"] == '"U_ESP_JetXYZ:Position.Axis 1"'


def test_telemetry_documents_match_run_scan_request(
    resolver, monkeypatch, tmp_path
) -> None:
    """Telemetry parity: the in-plan connect fork cannot drift silently.

    The autouse ``no_db`` fixture forces the scalar policy to ``None`` (no
    telemetry) on both entry points, so the other parity tests never touch
    the Tier-2 path.  This test swaps in a hermetic fake
    ``ScalarPolicyProvider`` so the telemetry fork
    (``_connect_telemetry_plan`` vs the runner's
    ``build_telemetry_readables`` over ``session.telemetry_batch``) is
    actually exercised on BOTH entry points: same soft event columns, same
    ``background_telemetry`` metadata, same save-set wholesale exclusion.
    """

    class _FakeScalarPolicy:
        """Protocol-complete ScalarPolicyProvider over a fixed table."""

        _subscribed = {
            # Not in the UC_Test save set -> becomes Tier-2 telemetry.
            "U_BgMonitor": ["Pressure", "Temp"],
            # In the save set -> excluded wholesale (Tier-1 owns it).
            "U_Cam": ["MaxCounts"],
        }

        def get_variables(self, device: str) -> list[str]:
            return list(self._subscribed.get(device, []))

        def all_variables(self, device: str) -> list[str]:
            return list(self._subscribed.get(device, []))

        def subscribed_by_device(self) -> dict[str, list[str]]:
            return {d: list(v) for d, v in self._subscribed.items()}

    for target in (
        "geecs_bluesky.scan_request_runner.make_scalar_policy",
        "geecs_bluesky.plans.scan_request_plan.make_scalar_policy",
    ):
        monkeypatch.setattr(target, lambda session: _FakeScalarPolicy())

    request = _noscan_request()
    docs_runner = _run_scan(
        "runner", request, resolver, tmp_path / "runner" / "Scan007", monkeypatch
    )
    docs_plan = _run_scan(
        "plan", request, resolver, tmp_path / "plan" / "Scan007", monkeypatch
    )
    _assert_same_run(
        docs_runner,
        docs_plan,
        tmp_path / "runner" / "Scan007",
        tmp_path / "plan" / "Scan007",
    )
    # The Tier-2 selection made it into the run metadata (parity of the
    # value itself is already pinned by _assert_same_run's start essence).
    assert docs_plan.start["background_telemetry"] == {
        "U_BgMonitor": ["Pressure", "Temp"]
    }
    # ...and into the event columns (the group keeps member names).
    telemetry_keys = {
        k
        for d in docs_plan.descriptors
        for k in d["data_keys"]
        if k.startswith("telemetry_")
    }
    assert telemetry_keys, "telemetry columns must exist in the event stream"
    assert all("u_bgmonitor" in k for k in telemetry_keys)
    assert not any("u_cam" in k for k in telemetry_keys), (
        "a save-set device must never be duplicated as telemetry"
    )


def test_plan_opts_into_pause_on_failed_move_and_runner_does_not() -> None:
    """Decision-4 activation pin (#645): only the queueserver plan passes
    failed_move_policy='pause'; the bridge/console path keeps the builder's
    'raise' default (PauseSupervisor coexistence)."""
    import inspect

    from geecs_bluesky.plans import scan_request_plan as srp
    from geecs_bluesky import scan_request_runner as srr

    plan_src = inspect.getsource(srp)
    assert 'failed_move_policy="pause"' in plan_src
    runner_src = inspect.getsource(srr)
    assert "failed_move_policy" not in runner_src
    # The bridge's ACTUAL builder call lives in session.scan, not the runner
    # (reviewer sabotage on the 8d11fcd9 confirm proved the runner-only pin
    # was hollow) — session.py must not opt in either.
    import geecs_bluesky.session as _session_mod

    assert "failed_move_policy" not in inspect.getsource(_session_mod)


def test_optimize_path_registers_pause_quiescer() -> None:
    """The optimize branch wraps run_plan with the ShotControlPauseQuiescer
    (reviewer-flagged gap: geecs_adaptive_scan bypasses build_step_scan_plan,
    so without this a pause mid-optimization leaves the trigger firing)."""
    import inspect

    from geecs_bluesky.plans import scan_request_plan as srp

    src = inspect.getsource(srp)
    assert "_with_pause_quiescer(" in src
    assert "ShotControlPauseQuiescer(controller)" in src
