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

import pytest

pytest.importorskip("aioca")

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


def test_optimize_mode_is_refused_loudly_after_validation(
    resolver, monkeypatch
) -> None:
    """Optimize relocates in a later round (decision 5): validated, refused."""
    session = _mock_session()
    claims: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: claims.append(experiment) or (None, None),
    )
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
    with pytest.raises(NotImplementedError, match="optimize"):
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
