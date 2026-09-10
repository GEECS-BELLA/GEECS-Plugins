"""geecs_preamble — the GEECS scan preamble as a RunEngine preprocessor (#807 phase 2).

A stock ``bluesky.plans`` verb whose run metadata carries a ScanRequest gets
the funnel's preamble: validate, resolve, connect, claim, ScanInfo, saving,
metadata — then the finalize chain on the way out.  Hermetic: the mock
RunEngine, the test config corpus, no DB and no gateway.

The devices come from a :class:`~geecs_bluesky.namespace.GeecsNamespace`, so
the objects the preamble configures are the ones the plan was handed —
`preamble.namespace_detectors`.
"""

from __future__ import annotations

import asyncio
import configparser
from collections import defaultdict

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
from tests.test_scan_request_runner import (  # noqa: E402
    LEGACY_EXP_SCAN_VARIABLES,
    LEGACY_SAVE_ELEMENT,
    LEGACY_SHOT_CONTROL,
)


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
    },
    types={"U_Cam": "Point Grey Camera", "U_Cam2": "Point Grey Camera"},
    subscribed={"U_Cam": ["MaxCounts"], "U_Cam2": ["Val"], "U_Slow": ["Pressure"]},
)


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


class Docs:
    def __init__(self) -> None:
        self.docs: dict[str, list[dict]] = defaultdict(list)

    def __call__(self, name: str, doc: dict) -> None:
        self.docs[name].append(doc)

    @property
    def start(self) -> dict:
        return self.docs["start"][0]

    def primary_events(self) -> list[dict]:
        uids = {d["uid"] for d in self.docs["descriptor"] if d["name"] == "primary"}
        return [e for e in self.docs["event"] if e["descriptor"] in uids]


def _request(**overrides) -> dict:
    base = dict(
        mode="noscan",
        shots_per_step=2,
        acquisition="free_run",
        save_sets=["UC_Test"],
        description="stats",
    )
    base.update(overrides)
    return ScanRequest.model_validate(base).model_dump(mode="json")


def _pace(RE, namespace):
    """The fake trigger: advance every triggered device's acq_timestamp."""
    cams = [namespace["U_Cam"], namespace["U_Cam2"]]
    for cam in cams:
        cam._trigger_timeout = 2.0

    async def pace() -> None:
        # Pace whichever cameras this plan actually connected — a plan need
        # not stage them all.
        ticks = 0
        while True:
            ticks += 1
            for cam in cams:
                if cam._monitoring:
                    set_mock_value(cam.acq_timestamp, 1000.0 + ticks)
            await asyncio.sleep(0.02)

    return asyncio.run_coroutine_threadsafe(pace(), RE._loop)


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
    docs = Docs()
    pacer = _pace(session.RE, namespace)
    try:
        session.RE(bp.count([namespace["U_Cam"]], num=1), docs)
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
    docs = Docs()
    pacer = _pace(session.RE, namespace)
    try:
        session.RE(
            bp.count(
                [namespace["U_Cam"], namespace["U_Cam2"], namespace["U_Slow"]],
                num=2,
                md={"geecs": _request()},
            ),
            docs,
        )
    finally:
        pacer.cancel()

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
    # and the run actually recorded shots
    assert len(docs.primary_events()) == 2
    assert docs.docs["stop"][0]["exit_status"] == "success"


def test_the_preamble_disarms_and_stops_saving_on_the_way_out(
    session, resolver, namespace, tmp_path, monkeypatch
) -> None:
    """The finalize chain runs: save off, then the trigger disarmed."""
    folder = tmp_path / "Scan007"
    folder.mkdir(parents=True, exist_ok=True)
    _install(session, resolver, namespace, folder, monkeypatch)
    cam = namespace["U_Cam"]
    pacer = _pace(session.RE, namespace)
    try:
        session.RE(bp.count([cam], num=1, md={"geecs": _request()}), Docs())
    finally:
        pacer.cancel()
    # mock backends do not echo a put onto the readback, so read the setpoint
    saved = asyncio.run_coroutine_threadsafe(
        cam.save._setpoint.get_value(), session.RE._loop
    ).result(timeout=5)
    assert saved == "off"
    path_written = asyncio.run_coroutine_threadsafe(
        cam.localsavingpath._setpoint.get_value(), session.RE._loop
    ).result(timeout=5)
    assert path_written.endswith("U_Cam")  # save-on wrote the per-device dir


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
        session.RE(bp.count([thin["U_Cam"]], num=1, md={"geecs": _request()}))


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
        session.RE(bp.count([ns["U_Cam"]], num=1, md={"geecs": _request()}))
