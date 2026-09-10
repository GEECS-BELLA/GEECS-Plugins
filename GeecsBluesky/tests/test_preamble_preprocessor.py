"""geecs_preamble — the GEECS scan preamble as a RunEngine preprocessor (#807 phase 2).

A stock ``bluesky.plans`` verb whose run metadata carries a ScanRequest gets
the funnel's preamble: validate, resolve, connect, claim, ScanInfo, saving,
metadata — then the finalize chain on the way out.  Hermetic: the mock
RunEngine, the test config corpus, no DB and no gateway.

The devices come from a :class:`~geecs_bluesky.namespace.GeecsNamespace`, so
the objects the preamble configures are the ones the plan was handed —
``GeecsNamespace.select``.
"""

from __future__ import annotations

import asyncio
import configparser

import bluesky.plans as bp
import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.config_resolver import ConfigsRepoResolver  # noqa: E402
from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace  # noqa: E402
from geecs_bluesky.preprocessors import (  # noqa: E402
    connect_on_demand,
    geecs_preamble,
    install_connect_on_demand,
    install_geecs_preamble,
)
from geecs_bluesky.session import GeecsSession  # noqa: E402
from geecs_schemas import ScanRequest  # noqa: E402
from tests.ca_mock_helpers import DocCollector  # noqa: E402
from tests.test_scan_request_runner import (  # noqa: E402
    LEGACY_EXP_SCAN_VARIABLES,
    LEGACY_SAVE_ELEMENT,
)

#: Strict shot control needs a non-empty ARMED state (the plan fires every
#: shot), which the shared free-run-era corpus profile does not define.
STRICT_SHOT_CONTROL = """\
device: U_DG645_ShotControl
variables:
  Trigger.Source:
    "OFF": "Single shot external rising edges"
    ARMED: "Single shot external rising edges"
    SCAN: "External rising edges"
    STANDBY: "External rising edges"
    SINGLESHOT: ""
  Trigger.ExecuteSingleShot:
    "OFF": ""
    ARMED: ""
    SCAN: ""
    STANDBY: ""
    SINGLESHOT: "on"
  Amplitude.Ch AB:
    "OFF": "0.5"
    ARMED: "4.0"
    SCAN: "4.0"
    STANDBY: "0.5"
"""


def row(name, *, settable=False, choices="numeric", tolerance=None):
    return {
        "name": name,
        "settable": settable,
        "variabletype": None,
        "choices": choices,
        "tolerance": tolerance,
        "units": "",
        "min": None,
        "max": None,
    }


#: The corpus save set names U_Cam (sync, native saving), U_Cam2 (sync) and
#: U_Slow (async) — the namespace must offer exactly those, with the save
#: controls U_Cam needs as its own served settables.
ROSTER = DeviceRoster(
    experiment="LegacyExp",
    variables={
        "U_Cam": [
            row("trigger", settable=True, choices="on,off"),
            row("MaxCounts"),
            row("save", settable=True, choices="on,off"),
            row("localsavingpath", settable=True, choices="path"),
        ],
        "U_Cam2": [row("trigger", settable=True, choices="on,off"), row("Val")],
        "U_Slow": [row("Pressure")],
        # The corpus scan variable `jet_z` targets this device; a step scan
        # needs its settable to exist as a namespace child.
        "U_ESP_JetXYZ": [
            row("Position.Axis 3", settable=True),
            row("Position.Axis 1", settable=True),
        ],
    },
    types={"U_Cam": "Point Grey Camera", "U_Cam2": "Point Grey Camera"},
    subscribed={
        "U_Cam": ["MaxCounts"],
        "U_Cam2": ["Val"],
        "U_Slow": ["Pressure"],
        "U_ESP_JetXYZ": ["Position.Axis 3"],
    },
)


@pytest.fixture
def configs_root(tmp_path):
    exp = tmp_path / "LegacyExp"
    (exp / "save_devices").mkdir(parents=True)
    (exp / "save_devices" / "UC_Test.yaml").write_text(LEGACY_SAVE_ELEMENT)
    (exp / "shot_control_configurations").mkdir()
    (exp / "shot_control_configurations" / "HTU-Normal.yaml").write_text(
        STRICT_SHOT_CONTROL
    )
    (exp / "scan_devices").mkdir()
    (exp / "scan_devices" / "scan_variables.yaml").write_text(LEGACY_EXP_SCAN_VARIABLES)
    return tmp_path


@pytest.fixture
def resolver(configs_root):
    return ConfigsRepoResolver("LegacyExp", experiments_root=configs_root)


@pytest.fixture(autouse=True)
def no_db(monkeypatch):
    """Neutralize the DB-backed providers (hermetic: no MySQL, no gateway)."""
    monkeypatch.setattr(
        "geecs_bluesky.scan_request_runner.make_scalar_policy", lambda session: None
    )
    monkeypatch.setattr(
        "geecs_bluesky.plans.preamble.make_scalar_policy", lambda session: None
    )
    monkeypatch.setattr(
        "geecs_bluesky.scan_request_runner.make_served_set_provider",
        lambda session: None,
    )


@pytest.fixture
def namespace():
    return GeecsNamespace(ROSTER)


@pytest.fixture
def session():
    return GeecsSession("LegacyExp", tiled=False, mock=True)


def _save_set(namespace) -> list:
    """The UC_Test save set's devices — what a stock plan must be handed.

    The preamble configures native saving for exactly these, so a plan that
    reads fewer would leave saved files with no event row to join them to
    (refused; see ``_check_plan_covers_save_set``).
    """
    return [namespace["U_Cam"], namespace["U_Cam2"], namespace["U_Slow"]]


def _request(**overrides) -> dict:
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="strict",
        save_sets=["UC_Test"],
        trigger_profile="HTU-Normal",
        description="stats",
    )
    base.update(overrides)
    return ScanRequest.model_validate(base).model_dump(mode="json")


class _RecordingSetter:
    """A shot-control setter that records writes instead of doing a caput.

    ``ShotController.from_writes`` already takes a ``setter_factory`` seam,
    so nothing in the engine is stubbed: the real controller replays the
    real profile, and the test sees the exact ordered state writes.
    """

    def __init__(self, device: str, variable: str, log: list, on_write) -> None:
        self._device, self._variable, self._log, self._on_write = (
            device,
            variable,
            log,
            on_write,
        )

    def set(self, value):
        from ophyd_async.core import AsyncStatus

        self._log.append((self._device, self._variable, value))
        self._on_write(self._variable, value)

        async def _done() -> None:
            return None

        return AsyncStatus(_done())


def _machine(RE, namespace, monkeypatch):
    """Stand in for the machine: a shot happens only when the plan fires.

    Returns the ordered ``(device, variable, value)`` write log, which is
    what pins the Gate-2 bracket: ARMED (free run halted) → saving on →
    SINGLESHOT per shot → saving off → STANDBY.
    """
    from geecs_bluesky.shot_controller import ShotController

    from geecs_bluesky.devices.ca.triggerable import CaTriggerable

    writes: list = []
    shots = {"n": 0}
    # Whatever the plan actually triggers is what the fired shot must reach
    # — recorded here rather than named, so the same stand-in serves the
    # namespace devices and the funnel's own per-scan ones.
    armed: list = []
    original_trigger = CaTriggerable.trigger

    def trigger(self, *args, **kwargs):
        if self not in armed:
            armed.append(self)
        self._trigger_timeout = 2.0
        return original_trigger(self, *args, **kwargs)

    monkeypatch.setattr(CaTriggerable, "trigger", trigger)

    def on_write(variable: str, value) -> None:
        # The profile fires by writing Trigger.ExecuteSingleShot='on'.
        if variable.endswith("ExecuteSingleShot") and value == "on":
            shots["n"] += 1
            for cam in armed:
                if cam._monitoring:
                    set_mock_value(cam.acq_timestamp, 1000.0 + shots["n"])

    original = ShotController.from_writes.__func__

    def from_writes(cls, w, **kwargs):
        kwargs["setter_factory"] = lambda device, variable: _RecordingSetter(
            device, variable, writes, on_write
        )
        return original(cls, w, **kwargs)

    monkeypatch.setattr(ShotController, "from_writes", classmethod(from_writes))

    # Every settable put goes into the SAME ordered log, so the bracket the
    # changelog leads with — arm (ARMED) before save-on, save-off before
    # disarm (STANDBY) — is pinned by one sequence rather than inferred from
    # two.  Wrapping the class covers the lazily connected children too.
    from geecs_bluesky.devices.ca.settable import CaSettable

    original_set = CaSettable.set

    def recording_set(self, value, *args, **kwargs):
        writes.append(
            (
                getattr(self, "_geecs_device_name", self.name),
                getattr(self, "_variable", self.name),
                value,
            )
        )
        return original_set(self, value, *args, **kwargs)

    monkeypatch.setattr(CaSettable, "set", recording_set)
    return writes


def _install(session, resolver, namespace, folder, monkeypatch):
    install_geecs_preamble(
        session.RE, session=session, resolver=resolver, namespace=namespace
    )
    install_connect_on_demand(session.RE, mock=True)

    def claim(experiment):
        assert experiment == "LegacyExp"
        return 7, str(folder)

    monkeypatch.setattr("geecs_bluesky.plans.run_wrapper.claim_scan_number", claim)


# ---------------------------------------------------------------------- shape
def test_install_keeps_connect_on_demand_outermost(session, namespace) -> None:
    install_geecs_preamble(session.RE, session=session, namespace=namespace)
    install_connect_on_demand(session.RE, mock=True)
    install_geecs_preamble(session.RE, session=session, namespace=namespace)  # again
    funcs = [getattr(p, "func", p) for p in session.RE.preprocessors]
    assert funcs == [geecs_preamble, connect_on_demand]


def test_a_plan_without_the_geecs_key_is_untouched(session, namespace) -> None:
    """No request, no preamble: the preprocessor must be a pure no-op."""
    install_geecs_preamble(session.RE, session=session, namespace=namespace)
    install_connect_on_demand(session.RE, mock=True)
    docs = DocCollector()
    # no request → no controller → nothing fires, so pace the trigger freely
    cam = namespace["U_Cam"]
    cam._trigger_timeout = 2.0

    async def free_run() -> None:
        ticks = 0
        while True:
            ticks += 1
            if cam._monitoring:
                set_mock_value(cam.acq_timestamp, 5000.0 + ticks)
            await asyncio.sleep(0.02)

    pacer = asyncio.run_coroutine_threadsafe(free_run(), session.RE._loop)
    try:
        session.RE(bp.count([cam], num=1), docs)
    finally:
        pacer.cancel()
    assert "scan_number" not in docs.start and "scan_folder" not in docs.start
    assert docs.docs["stop"][0]["exit_status"] == "success"


# -------------------------------------------------------------- the preamble
def test_stock_count_gets_the_full_preamble(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """bp.count + md={"geecs": request} == a prepared, claimed, recorded scan."""
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    writes = _machine(session.RE, namespace, monkeypatch)
    docs = DocCollector()
    session.RE(
        bp.count(
            [namespace["U_Cam"], namespace["U_Cam2"], namespace["U_Slow"]],
            num=2,
            md={"geecs": _request()},
        ),
        docs,
    )

    start = docs.start
    # the claim reached the start document
    assert start["scan_number"] == 7 and start["scan_id"] == 7
    assert start["scan_folder"] == str(folder)
    assert start["bluesky_backend"] is True and start["experiment"] == "LegacyExp"
    # GEECS request metadata every downstream reader needs
    assert start["scan_request_mode"] == "noscan"
    assert start["save_sets"] == ["UC_Test"]
    assert start["description"] == "stats"
    assert "geecs_scalar_headers" in start
    # native saving was configured for the save set's saving device
    assert "U_Cam" in start["nonscalar_save_paths"]
    # the stock plan keeps its own identity
    assert start["plan_name"] == "count"
    # ScanInfo was written post-claim
    ini = folder / "ScanInfoScan007.ini"
    assert ini.exists()
    parser = configparser.ConfigParser()
    parser.read_string(ini.read_text())
    assert parser["Scan Info"]["scanmode"] == '"noscan"'
    # and the run actually recorded shots — one fire each
    assert len(docs.primary_events()) == 2
    fires = [w for w in writes if w[1].endswith("ExecuteSingleShot") and w[2] == "on"]
    assert len(fires) == 2, writes
    assert docs.docs["stop"][0]["exit_status"] == "success"


def test_the_preamble_disarms_and_stops_saving_on_the_way_out(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """The finalize chain runs: save off, then the trigger disarmed."""
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    cam = namespace["U_Cam"]
    writes = _machine(session.RE, namespace, monkeypatch)
    session.RE(
        bp.count(_save_set(namespace), num=1, md={"geecs": _request(shots_per_step=1)}),
        DocCollector(),
    )
    # mock backends do not echo a put onto the readback, so read the setpoint
    saved = asyncio.run_coroutine_threadsafe(
        cam.save._setpoint.get_value(), session.RE._loop
    ).result(timeout=5)
    assert saved == "off"
    path_written = asyncio.run_coroutine_threadsafe(
        cam.localsavingpath._setpoint.get_value(), session.RE._loop
    ).result(timeout=5)
    assert path_written.endswith("U_Cam")  # save-on wrote the per-device dir

    # Gate-2 save windowing, the ordering this bracket exists for: the free
    # run is HALTED (ARMED) before saving is enabled, and saving is disabled
    # again before the trigger is released (STANDBY) — otherwise the camera
    # writes orphan frames with no event row.
    sources = [w[2] for w in writes if w[1] == "Trigger.Source"]
    assert sources[0] == "Single shot external rising edges"  # ARMED, halted
    assert sources[-1] == "External rising edges"  # STANDBY, released last

    # …and the save writes sit INSIDE that bracket.  Asserted on positions
    # in the one ordered log, so swapping arm_single_shot and
    # save_enable_plan fails this test rather than passing it quietly.
    order = [(w[1], w[2]) for w in writes]
    arm = order.index(("Trigger.Source", "Single shot external rising edges"))
    save_on = order.index(("save", "on"))
    save_off = len(order) - 1 - order[::-1].index(("save", "off"))
    release = (
        len(order) - 1 - order[::-1].index(("Trigger.Source", "External rising edges"))
    )
    assert arm < save_on, order  # halt the free run BEFORE saving starts
    assert save_off < release, order  # stop saving BEFORE edges come back


def test_free_run_is_refused_before_the_claim(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """Stock plans are strict-only; free-run is retired (#807)."""
    folder = tmp_path / "Scan007"
    _install(session, resolver, namespace, folder, monkeypatch)
    with pytest.raises(GeecsConfigurationError, match="strict shot control only"):
        session.RE(
            bp.count(
                [namespace["U_Cam"]],
                num=1,
                md={"geecs": _request(acquisition="free_run")},
            )
        )
    assert not (folder / "ScanInfoScan007.ini").exists()


def test_a_failure_before_the_claim_burns_no_scan_number(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """The pre-claim invariant survives the move to a preprocessor."""
    folder = tmp_path / "Scan007"
    _install(session, resolver, namespace, folder, monkeypatch)
    claimed: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.run_wrapper.claim_scan_number",
        lambda experiment: claimed.append(experiment) or (7, str(folder)),
    )
    with pytest.raises(Exception):
        session.RE(
            bp.count(
                [namespace["U_Cam"]], num=1, md={"geecs": _request(save_sets=["Nope"])}
            )
        )
    assert claimed == []
    assert not folder.exists()


# ------------------------------------------------------- the namespace source
def test_a_save_set_device_missing_from_the_namespace_is_loud(
    session, resolver, tmp_path, monkeypatch
) -> None:
    thin = GeecsNamespace(
        DeviceRoster(
            experiment="LegacyExp",
            variables={"U_Cam": ROSTER.variables["U_Cam"]},
            types={"U_Cam": "Point Grey Camera"},
            subscribed={"U_Cam": ["MaxCounts"]},
        )
    )
    folder = tmp_path / "Scan007"
    _install(session, resolver, thin, folder, monkeypatch)
    with pytest.raises(GeecsConfigurationError, match="not in the device namespace"):
        session.RE(
            bp.count([thin["U_Cam"]], num=1, md={"geecs": _request(shots_per_step=1)})
        )


def test_a_save_set_scalar_the_device_does_not_read_is_loud(
    session, resolver, tmp_path, monkeypatch
) -> None:
    """The namespace reads the DB subscribed list; a mismatch must not be silent."""
    variables = dict(ROSTER.variables)
    variables["U_Cam2"] = [
        row("trigger", settable=True, choices="on,off"),
        row("Other"),
    ]
    ns = GeecsNamespace(
        DeviceRoster(
            experiment="LegacyExp",
            variables=variables,
            types=ROSTER.types,
            subscribed={**ROSTER.subscribed, "U_Cam2": ["Other"]},
        )
    )
    folder = tmp_path / "Scan007"
    _install(session, resolver, ns, folder, monkeypatch)
    with pytest.raises(GeecsConfigurationError, match="which the namespace device"):
        session.RE(
            bp.count([ns["U_Cam"]], num=1, md={"geecs": _request(shots_per_step=1)})
        )


# ------------------------------------------------- what the funnel got free
def test_per_run_device_state_does_not_leak_into_the_next_plan(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """A namespace noun outlives the run: nothing it configured may survive.

    The per-scan device classes were thrown away at the end of a scan and
    got this for free.  If it leaked, an unrelated later plan would emit a
    ``nonscalar_save_path`` column — and asset documents — pointing into
    the previous scan's folder.
    """
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    cam = namespace["U_Cam"]
    session.RE(
        bp.count(_save_set(namespace), num=1, md={"geecs": _request(shots_per_step=1)}),
        DocCollector(),
    )
    assert cam._save_nonscalar_data is False
    assert cam._nonscalar_save_path is None
    assert cam._asset_definitions == ()
    assert cam._asset_scan_number is None

    # and the proof at the document level: a plain plan, no GEECS key
    cam._trigger_timeout = 2.0

    async def free_run() -> None:
        ticks = 0
        while True:
            ticks += 1
            if cam._monitoring:
                set_mock_value(cam.acq_timestamp, 9000.0 + ticks)
            await asyncio.sleep(0.02)

    docs = DocCollector()
    pacer = asyncio.run_coroutine_threadsafe(free_run(), session.RE._loop)
    try:
        session.RE(bp.count([cam], num=1), docs)
    finally:
        pacer.cancel()
    columns = set(docs.primary_events()[0]["data"])
    assert not [c for c in columns if c.endswith("nonscalar_save_path")], columns


def test_a_plan_that_does_not_read_the_whole_save_set_is_refused(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """Saved but unread = orphan frames with no acq_timestamp to join on."""
    folder = tmp_path / "Scan007"
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    claimed: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.run_wrapper.claim_scan_number",
        lambda experiment: claimed.append(experiment) or (7, str(folder)),
    )
    with pytest.raises(GeecsConfigurationError, match="which this plan does not read"):
        session.RE(
            bp.count(
                [namespace["U_Cam"]], num=1, md={"geecs": _request(shots_per_step=1)}
            )
        )
    assert claimed == []  # refused before the claim


def test_a_plan_whose_shape_contradicts_the_request_is_refused(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """ScanInfo records the request's numbers, so they must match the plan's."""
    folder = tmp_path / "Scan007"
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    claimed: list = []
    monkeypatch.setattr(
        "geecs_bluesky.plans.run_wrapper.claim_scan_number",
        lambda experiment: claimed.append(experiment) or (7, str(folder)),
    )
    with pytest.raises(GeecsConfigurationError, match="recorded shots"):
        session.RE(
            bp.count(
                _save_set(namespace), num=2, md={"geecs": _request(shots_per_step=5)}
            )
        )
    assert claimed == []


def test_a_dropped_frame_is_re_fired_not_fatal(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """The stock door gets geecs_single_shot's bounded refire, not a bare wait.

    The first fire produces no frame (the ~1% camera drop); the scan must
    recover rather than abort, and must still record exactly one row.
    """
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    cams = [namespace["U_Cam"], namespace["U_Cam2"]]
    for cam in cams:
        cam._trigger_timeout = 0.6
    writes = _machine(session.RE, namespace, monkeypatch)
    # Swallow the first fire: the trigger box pulses, no camera sees a frame.
    dropped = {"done": False}
    original = namespace["U_Cam"].__class__.trigger

    from geecs_bluesky.shot_controller import ShotController

    real_fire = ShotController.fire_shot

    def fire_shot(self):
        if not dropped["done"]:
            dropped["done"] = True

            def _swallow():
                return iter(())

            return _swallow()
        return real_fire(self)

    monkeypatch.setattr(ShotController, "fire_shot", fire_shot)
    _ = original
    docs = DocCollector()
    session.RE(
        bp.count(_save_set(namespace), num=1, md={"geecs": _request(shots_per_step=1)}),
        docs,
    )
    # Without the refire the FailedStatus propagates into bp.count, which
    # has no handler, and the run aborts.  It recovered, and recorded one
    # row for one physical shot — strict semantics survive the retry.
    assert docs.docs["stop"][0]["exit_status"] == "success"
    assert len(docs.primary_events()) == 1
    fires = [w for w in writes if w[1].endswith("ExecuteSingleShot") and w[2] == "on"]
    assert 1 <= len(fires) <= 3, writes  # bounded: max_refires=2


def test_telemetry_columns_reach_the_event(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """Background telemetry is primary-stream columns, one value per row.

    The funnel gets that by appending the group to the plan's read set; a
    stock plan reads only its own detectors, so the preprocessor injects
    the reads into the same event.  Advertising the columns in the start
    document while emitting none would be the worse bug.  The corpus has
    no DB-backed scalar policy, so the group itself is stubbed — what is
    under test is the injection, not the selection.
    """
    from ophyd_async.core import StandardReadable, soft_signal_r_and_setter

    class _Telemetry(StandardReadable):
        def __init__(self, name: str) -> None:
            with self.add_children_as_readables():
                self.pressure, _ = soft_signal_r_and_setter(float, 1.25)
            super().__init__(name=name)

    group = _Telemetry("tier2")

    def fake_telemetry(session_, save_set, scalar_policy):
        yield from ()
        return [group], {"U_Tier2": ["Pressure"]}

    monkeypatch.setattr(
        "geecs_bluesky.plans.preamble._connect_telemetry_plan", fake_telemetry
    )
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    docs = DocCollector()
    session.RE(
        bp.count(_save_set(namespace), num=1, md={"geecs": _request(shots_per_step=1)}),
        docs,
    )
    assert docs.docs["stop"][0]["exit_status"] == "success"
    columns = set(docs.primary_events()[0]["data"])
    assert "tier2-pressure" in columns, sorted(columns)


def test_scalar_headers_carry_the_scan_axis(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """The swept motor's legacy header must reach the s-file exporter."""
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    axis = namespace.variable("U_ESP_JetXYZ", "Position.Axis 3")
    points = [0.0, 1.0]
    docs = DocCollector()
    session.RE(
        bp.list_scan(
            _save_set(namespace),
            axis,
            points,
            md={
                "geecs": _request(
                    mode="step",
                    shots_per_step=1,
                    axes=[{"variable": "jet_z", "positions": {"values": points}}],
                )
            },
        ),
        docs,
    )
    headers = docs.start["geecs_scalar_headers"]
    assert any("jetxyz" in key.lower() for key in headers), headers


def test_a_position_the_request_never_declared_is_refused(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """Equal point counts must not hide a different set of points."""
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    axis = namespace.variable("U_ESP_JetXYZ", "Position.Axis 3")
    with pytest.raises(GeecsConfigurationError, match="not among the positions"):
        session.RE(
            bp.list_scan(
                _save_set(namespace),
                axis,
                [0.0, 7.0],  # the request declares 0 and 1
                md={
                    "geecs": _request(
                        mode="step",
                        shots_per_step=1,
                        axes=[
                            {"variable": "jet_z", "positions": {"values": [0.0, 1.0]}}
                        ],
                    )
                },
            )
        )


def test_a_stock_scan_writes_a_scan_log(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """/triage reads scans/ScanNNN/scan.log; the stock door must write one."""
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    _machine(session.RE, namespace, monkeypatch)
    session.RE(
        bp.count(_save_set(namespace), num=1, md={"geecs": _request(shots_per_step=1)}),
        DocCollector(),
    )
    logs = list(folder.glob("*.log"))
    assert logs, sorted(p.name for p in folder.iterdir())


def test_install_preserves_the_connect_on_demand_configuration(
    session, namespace
) -> None:
    """A mock worker must not be sent at real Channel Access by a re-install."""
    install_connect_on_demand(session.RE, mock=True, timeout=45.0)
    install_geecs_preamble(session.RE, session=session, namespace=namespace)
    connect = session.RE.preprocessors[-1]
    assert getattr(connect, "func", None) is connect_on_demand
    assert connect.keywords["mock"] is True
    assert connect.keywords["timeout"] == 45.0


def test_the_stock_door_and_the_funnel_record_the_same_scan(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """The acceptance contract: one request, two doors, one scan record.

    Every GEECS-owned start-document key is compared exactly.  The keys a
    stock plan owns (``plan_name``, ``detectors``, ``hints``, the plan's own
    argument echo) legitimately differ — that difference is the point of
    the migration — so they are excluded by name rather than by guesswork,
    and the exclusion list is the thing to look at when a key goes missing.
    """
    from geecs_bluesky.plan_session import set_plan_session
    from geecs_bluesky.plans.scan_request_plan import geecs_scan_request_plan

    request = _request(shots_per_step=1)
    plan_owned = {
        "uid",
        "time",
        "scan_id",
        "versions",
        "plan_type",
        "plan_name",
        "plan_args",
        "plan_pattern",
        "plan_pattern_args",
        "plan_pattern_module",
        "detectors",
        "motors",
        "hints",
        "num_points",
        "num_intervals",
    }

    def _geecs_keys(start: dict, folder) -> dict:
        return {
            k: str(v).replace(str(folder), "<scan_folder>")
            for k, v in start.items()
            if k not in plan_owned
        }

    # ---- door 1: the funnel -------------------------------------------
    funnel_folder = tmp_path / "funnel" / "Scan007"
    funnel_folder.mkdir(parents=True, exist_ok=True)
    _machine(session.RE, namespace, monkeypatch)
    monkeypatch.setattr(
        "geecs_bluesky.plans.scan_request_plan.claim_scan_number",
        lambda experiment: (7, str(funnel_folder)),
    )
    set_plan_session(session)
    funnel_docs = DocCollector()
    try:
        session.RE(geecs_scan_request_plan(request, resolver=resolver), funnel_docs)
    finally:
        set_plan_session(None)

    # ---- door 2: a stock plan -----------------------------------------
    stock_folder = tmp_path / "stock" / "Scan007"
    stock_folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, stock_folder, monkeypatch)
    stock_docs = DocCollector()
    session.RE(bp.count(_save_set(namespace), num=1, md={"geecs": request}), stock_docs)

    assert _geecs_keys(stock_docs.start, stock_folder) == _geecs_keys(
        funnel_docs.start, funnel_folder
    )
    # …the same rows, with the same device columns.
    assert len(stock_docs.primary_events()) == len(funnel_docs.primary_events())
    stock_columns = set(stock_docs.primary_events()[0]["data"])
    funnel_columns = set(funnel_docs.primary_events()[0]["data"])
    # The one known, scheduled difference: the funnel appends a synthetic
    # ScanContext device for the per-row step bookkeeping.  Stock plans get
    # that from the per-step function (GEECS-Plugins#807 phase 3), which is
    # also when ScanContext retires.  Asserted as an EXACT set so any other
    # column that appears on one door and not the other fails here.
    scan_context = {"bin_number", "scan_event_index", "shot_index_in_bin"}
    assert funnel_columns - stock_columns == scan_context
    assert stock_columns - funnel_columns == set()
    ini_a = (funnel_folder / "ScanInfoScan007.ini").read_text()
    ini_b = (stock_folder / "ScanInfoScan007.ini").read_text()
    assert ini_a == ini_b


# ------------------------------------------------- axis topologies refused
def test_axis_topologies_the_namespace_cannot_build_are_refused(namespace) -> None:
    """Silently degrading a confirming or motor axis would take every row
    at the wrong position, with nothing in the data saying so."""
    from geecs_bluesky.plans.preamble import _namespace_movable
    from geecs_bluesky.scan_request_runner import (
        PlainMovableTarget,
        PseudoMovableTarget,
    )

    plain = PlainMovableTarget(
        device="U_ESP_JetXYZ", variable="Position.Axis 3", kind="setpoint", confirm=None
    )
    assert _namespace_movable(namespace, plain) is namespace.variable(
        "U_ESP_JetXYZ", "Position.Axis 3"
    )

    with pytest.raises(GeecsConfigurationError, match="pseudo/composite"):
        _namespace_movable(
            namespace,
            PseudoMovableTarget(variable_name="combo", mode="absolute", components=()),
        )
    with pytest.raises(GeecsConfigurationError, match="confirms on"):
        _namespace_movable(
            namespace,
            PlainMovableTarget(
                device="U_ESP_JetXYZ",
                variable="Position.Axis 3",
                kind="setpoint",
                confirm="U_Cam:MaxCounts",
            ),
        )
    # kind: motor with no DB tolerance → a plain setpoint, which would
    # complete before the readback converged.
    with pytest.raises(GeecsConfigurationError, match="kind: motor"):
        _namespace_movable(
            namespace,
            PlainMovableTarget(
                device="U_ESP_JetXYZ",
                variable="Position.Axis 3",
                kind="motor",
                confirm=None,
            ),
        )
