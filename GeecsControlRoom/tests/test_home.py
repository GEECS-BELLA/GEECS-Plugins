"""The home: the two lists sort the module's words, the launcher reads home.yaml, the strip survives a silent scanner, the directory lists the registry by kind.

The kit's rules on ``home.html`` are ``test_app.py``'s template guard, which walks every template.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_control_room import device_pages
from geecs_control_room.app import create_app
from geecs_control_room.home import NO_PAGE, directory, launcher, sort_devices, words
from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings
from tests.test_device_pages import _NATIVE, _PANEL, _room

_WORDS = ["live", "stale", "not answering", "unreachable", "off", "error", "unknown"]


def _detail(
    verdict: str, *, heartbeat: str | None = "live", sp: str | None = None
) -> dict:
    roles = [{"role": "readback", "axis": None, "verdict": verdict}]
    roles += [{"role": "setpoint", "axis": "m1", "verdict": sp}] if sp else []
    return {"heartbeat": heartbeat, "roles": roles}


def test_every_word_lands_in_its_list() -> None:
    roster = [{"name": w, "family": "f", "state": w} for w in _WORDS]
    roster.append({"name": "alarmed_sp", "family": "f", "state": "live"})
    details = {w: _detail(w) for w in _WORDS} | {"alarmed_sp": _detail("live", sp="error")}  # fmt: skip
    lists = sort_devices(roster, details, [], {})
    assert [d["name"] for d in lists["not_live"]] == [w for w in _WORDS if w != "live"]
    assert [d["name"] for d in lists["in_alarm"]] == ["error", "alarmed_sp"]
    assert lists["in_alarm"][1]["cause"] == "error: setpoint[m1]"


def test_a_row_names_its_cause_ioc_and_last_command() -> None:
    roster = [{"name": "ps1", "family": "fastps", "state": "unreachable"}]
    journal = [
        {"device": "ps1", "outcome": "accepted", "role": "current"},
        {"device": "ps1", "outcome": "refused", "role": "current"},
    ]
    detail = _detail("unreachable", heartbeat="unreachable")
    (row,) = sort_devices(roster, {"ps1": detail}, journal, {"ps1": "ps1-ioc"})["not_live"]  # fmt: skip
    assert row["cause"] == "heartbeat unreachable; unreachable: readback"
    assert (row["ioc"], row["last"]["outcome"], row["kit"]) == ("ps1-ioc", "refused", "degraded")  # fmt: skip
    (silent,) = sort_devices(roster, {}, [], {})["not_live"]
    assert "no detail" in silent["cause"]
    (between,) = sort_devices([roster[0] | {"state": "stale"}], {"ps1": _detail("live")}, [], {})["not_live"]  # fmt: skip
    assert between["cause"] == "the module's state is stale; it names no role"


def _module(roster: list[dict], fail: bool = False) -> ModuleClient:
    def answer(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("refused")
        path = request.url.path
        if path.endswith("/devices"):
            return httpx.Response(200, json={"experiment": "Bench", "devices": roster})
        if path.endswith("/journal"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=_detail("live"))

    return ModuleClient(
        "http://canopy.test", "k", transport=httpx.MockTransport(answer)
    )


def test_the_lists_read_none_when_all_are_live_and_never_when_the_module_is_silent(settings: Settings) -> None:  # fmt: skip
    roster = [{"name": "m", "family": "motor_record", "state": "live"}]
    with TestClient(create_app(settings, _module(roster))) as c:
        body = c.get("/api/home/devices").json()
    assert body["ok"] and body["not_live"] == body["in_alarm"] == []
    assert body["devices"] == [{"name": "m", "family": "motor_record"}]
    with TestClient(create_app(settings, _module(roster, fail=True))) as c:
        body = c.get("/api/home/devices").json()
    assert (
        not body["ok"] and "not_live" not in body and "did not answer" in body["detail"]
    )


def test_an_unread_detail_never_lets_the_alarm_list_read_none(settings: Settings) -> None:  # fmt: skip
    """A live roster with a timed-out detail: ok, no alarms read, and the device named as unread (not "none")."""
    roster = [{"name": "m", "family": "motor_record", "state": "live"}]

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/devices"):
            return httpx.Response(200, json={"experiment": "Bench", "devices": roster})
        if request.url.path.endswith("/journal"):
            return httpx.Response(200, json=[])
        raise httpx.ReadTimeout("the module took too long")

    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
    with TestClient(create_app(settings, module)) as c:
        body = c.get("/api/home/devices").json()
    assert body["ok"] and body["in_alarm"] == [] and body["unread"] == [{"name": "m"}]
    assert sort_devices(roster, {"m": _detail("live")}, [], {})["unread"] == []


def test_the_strip_says_the_scanner_is_unreachable(settings: Settings, module: ModuleClient) -> None:  # fmt: skip
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    scanner = httpx.AsyncClient(base_url="http://scanner.test", transport=httpx.MockTransport(refuse))  # fmt: skip
    with TestClient(create_app(settings, module, scanner)) as c:
        assert c.get("/api/home/strip").json()["scanner"] == "unreachable"
        assert c.get("/").status_code == 200


@pytest.mark.parametrize(("re_state", "run", "kit"), [("running", "running", "running"), ("paused", "paused", "paused"), ("idle", "idle", "ok")])  # fmt: skip
def test_the_strip_takes_the_run_word_from_the_manager(settings: Settings, module: ModuleClient, re_state: str, run: str, kit: str) -> None:  # fmt: skip
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"connected": True, "re_state": re_state})
        progress = {"scan_number": 12, "shots_done": 7, "planned_total": 50, "state": "running"}  # fmt: skip
        return httpx.Response(200, json=progress)  # the documents lag a pause

    scanner = httpx.AsyncClient(base_url="http://scanner.test", transport=httpx.MockTransport(answer))  # fmt: skip
    with TestClient(create_app(settings, module, scanner)) as c:
        s = c.get("/api/home/strip").json()
    assert (s["run"], s["kit"], s["scan_number"], s["shots_done"]) == (run, kit, 12, 7)


_HOME = """schema_version: 1
areas:
  - title: Bench
    screens: [sim_bench, absent]
device_pages:
  - {title: Cameras, families: [ad_camera_chain]}
  - {title: Pumps, families: [no_such_family]}
links: [scanner, portal]
"""


def test_the_launcher_disables_what_is_missing_with_the_reason(settings: Settings) -> None:  # fmt: skip
    folder = settings.experiment_dir
    (folder / "screens").mkdir(parents=True)
    (folder / "screens" / "sim_bench.yaml").write_text("title: x\n")
    (folder / "home.yaml").write_text(_HOME)
    s = Settings(**{**settings.__dict__, "links": {"portal": "http://portal.test/"}})
    got = launcher(s)
    (_, (bench, absent)), = got.areas  # fmt: skip
    assert (bench.href, absent.href, absent.reason) == ("/screens/sim_bench", None, "no screens/absent.yaml")  # fmt: skip
    assert [p.reason for p in got.device_pages] == ["", "no family no_such_family installed"]  # fmt: skip
    assert [(e.href, bool(e.reason)) for e in got.links] == [(None, True), ("http://portal.test/", False)]  # fmt: skip
    (folder / "home.yaml").write_text("links: [http://elsewhere]\n")
    assert launcher(s).state == "error"
    (folder / "home.yaml").unlink()
    assert launcher(s).state == "empty"


def test_settings_read_the_links_the_scanner_pair_included() -> None:
    env = {
        "GEECS_EXPERIMENT": "Bench",
        "GEECS_CONFIGS_ROOT": "/configs",
        "GEECS_CONTROL_ROOM_CANOPY_URL": "http://canopy.test",
        "GEECS_CONTROL_ROOM_READ_KEY": "k",
        "GEECS_CONTROL_ROOM_SCANNER_URL": "http://127.0.0.1:8300/",
        "GEECS_CONTROL_ROOM_SCANNER_BROWSER_URL": "/scanner",
        "GEECS_CONTROL_ROOM_LOGBOOK_URL": "/logbook",
    }
    s = Settings.from_env(env)
    assert s.scanner_url == "http://127.0.0.1:8300"
    assert dict(s.links) == {"scanner": "/scanner", "logbook": "/logbook"}


def test_a_live_db_device_is_not_read_in_detail_and_never_unread(settings: Settings) -> None:  # fmt: skip
    roster = [{"name": n, "source": "gateway", "state": s} for n, s in (("U_A", "live"), ("U_B", "unreachable"))]  # fmt: skip
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/devices"):
            return httpx.Response(200, json={"experiment": "Bench", "devices": roster})
        if path.endswith("/journal"):
            return httpx.Response(200, json=[])
        asked.append(path.rsplit("/", 1)[-1])
        return httpx.Response(200, json=_detail("unreachable"))

    module = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(answer))  # fmt: skip
    with TestClient(create_app(settings, module)) as c:
        body = c.get("/api/home/devices").json()
    assert asked == ["U_B"] and body["unread"] == []
    assert [row["name"] for row in body["not_live"]] == ["U_B"]


async def test_the_baseline_sums_the_golden_diffs_and_is_none_without_one(settings: Settings) -> None:  # fmt: skip
    from geecs_control_room.home import baseline

    presets = [{"id": "g1", "name": "a", "golden": True, "scope": {"devices": ["X"]}}, {"id": "n", "name": "b", "golden": False, "scope": {}}, {"id": "g2", "name": "c", "golden": True, "scope": {"screen": ["none"]}}]  # fmt: skip
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(
            request.url.path
            + ("?" + request.url.query.decode() if request.url.query else "")
        )
        if request.url.path.endswith("/presets"):
            return httpx.Response(200, json={"presets": presets})
        return httpx.Response(
            200,
            json={
                "counts": {"differs": 2 if "g1" in request.url.path else 1},
                "rows": [
                    {"word": "differs"},
                    {
                        "word": "stale" if "g2" in request.url.path else "same"
                    },  # a word the module has not got yet: still not compared
                ],
            },
        )

    module = ModuleClient(
        "http://canopy.test", "k", transport=httpx.MockTransport(answer)
    )
    body = await baseline(settings, module)
    assert (body["golden"], body["differs"], body["incomplete"], [p["name"] for p in body["presets"]]) == (2, 3, 1, ["a", "c"])  # fmt: skip
    assert sorted(asked)[1:] == [
        "/geecs/v1/presets/g1/diff",
        "/geecs/v1/presets/g2/diff",
    ]  # a golden's own scope: the module's; no screen file here
    presets[:] = [presets[1]]
    assert await baseline(settings, module) == {"ok": True, "golden": 0, "differs": 0, "incomplete": 0, "presets": []}  # fmt: skip
    silent = ModuleClient("http://canopy.test", "k", transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))))  # fmt: skip
    assert (await baseline(settings, silent))["ok"] is False


# ---- the device directory: the registry by page kind, the module's words beside each row


@pytest.fixture
def registry(settings: Settings) -> Path:
    """The device pages' registry documents (two cameras, a motor, a supply) in the experiment's folder."""
    folder = settings.experiment_dir
    folder.mkdir(parents=True)
    (folder / "native_devices.yaml").write_text(_NATIVE)
    (folder / "panel_devices.yaml").write_text(_PANEL)
    return folder


def _row(html: str, name: str) -> str:
    return re.search(rf'<tr data-name="{name}".*?</tr>', html, re.S).group(0)


def test_the_directory_lists_the_registry_by_kind_and_hides_an_empty_kind(settings: Settings, registry: Path) -> None:  # fmt: skip
    found = directory(settings)  # no roster read: the registry's devices alone
    assert found.state == "ok" and [(k, [r["name"] for r in rows]) for k, rows in found.groups] == [
        ("camera", ["Cam1", "Cam2"]), ("motor", ["Mot1"]), ("supply", ["ps1"])]  # fmt: skip
    with _room(settings) as c:
        html = c.get("/").text
    assert re.findall(r'data-group="([^"]+)"', html) == [
        "camera",
        "motor",
        "supply",
        "settables",
    ]  # no trace device: no group; the roster's DB device under settables, last
    assert (
        '<a class="eyebrow" href="/device/camera"' in html
    )  # the heading is the kind's picker
    cam = _row(html, "Cam2")
    assert (
        '<a href="/device/camera/Cam2">Cam2</a>' in cam
        and "<td>ad_camera_chain</td>" in cam
    )
    assert 'data-word="state"' in cam and 'data-word="heartbeat"' in cam
    db = _row(html, "U_DB")  # a DB device is the roster's, read once at page load
    assert (
        '<a href="/device/settables/U_DB">U_DB</a>' in db
        and "<td>PhantomCamera</td>" in db
    )
    assert html.index('id="directory"') > html.index(
        "Launcher"
    )  # the curated launcher stays above


def test_a_family_no_page_draws_is_listed_last_without_a_link(settings: Settings, registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    unpaged = {
        k: w for k, w in device_pages._WORD.items() if w != "supply"
    }  # a kind with no page yet
    monkeypatch.setattr(device_pages, "_WORD", unpaged)
    groups = directory(settings).groups
    assert [k for k, _ in groups] == ["camera", "motor", NO_PAGE] and [r["name"] for r in groups[-1][1]] == ["ps1"]  # fmt: skip
    with _room(settings) as c:
        html = c.get("/").text
    ps1 = _row(html, "ps1")
    assert "<a " not in ps1 and '<td class="id">ps1</td>' in ps1
    assert (
        f'<span class="eyebrow" title="no device page draws these families yet">{NO_PAGE} · 1</span>'
        in html
    )


def test_the_directory_words_are_the_modules_for_every_listed_device(settings: Settings, registry: Path) -> None:  # fmt: skip
    with _room(settings) as c:
        body = c.get("/api/home/devices").json()
    assert set(body["words"]) == {
        "Cam1",
        "Cam2",
        "Mot1",
        "ps1",
        "U_DB",
    }  # the DB device too: it is listed
    assert (
        body["words"]["U_DB"]["state"] == "live"
        and body["words"]["U_DB"]["heartbeat"] is None
    )  # no IOC
    assert body["words"]["Cam2"] == {"state": "error", "kit": "failed", "heartbeat": "live", "heartbeatKit": "ok"}  # fmt: skip
    assert body["words"]["Mot1"]["kit"] == "degraded"
    # not served: its heartbeat is unknown, never "no IOC" (the registry may name one)
    assert body["words"]["ps1"] == {"state": "not on the roster", "kit": "unknown", "heartbeat": "unknown", "heartbeatKit": "unknown"}  # fmt: skip
    roster = [{"name": "a", "state": "live"}, {"name": "b", "state": "live"}]
    got = words(["a", "b", "c"], roster, {"a": None, "b": {"heartbeat": None, "roles": []}})  # fmt: skip
    assert (
        got["a"]["heartbeat"] == "unknown" and got["b"]["heartbeat"] is None
    )  # fmt: skip  (unread vs no IOC)
    assert got["c"]["heartbeat"] == "unknown"  # off the roster


def test_the_directory_rows_stay_when_the_module_is_silent(settings: Settings, registry: Path) -> None:  # fmt: skip
    with _room(settings, down=True) as c:
        html, body = c.get("/").text, c.get("/api/home/devices").json()
    assert (
        '<a href="/device/camera/Cam1">Cam1</a>' in html
        and not body["ok"]
        and "words" not in body
    )


_CONFLICT = ("schema_version: 1\ndevices:\n  Mot1: {kind: epics_motor, axes: [m1]}\n", "schema_version: 1\ndevices:\n  Mot1: {family: easydriver}\n")  # fmt: skip


@pytest.mark.parametrize(
    ("native", "panel", "state", "title"),
    [
        (None, None, "empty", "No device in the registry"),
        ("devices: [", None, "error", "The registry cannot be read"),
        (*_CONFLICT, "empty", "No device the directory can list"),  # both files read; their one device conflicts
    ],
)  # fmt: skip
def test_the_directory_says_why_it_has_no_rows(settings: Settings, native: str | None, panel: str | None, state: str, title: str) -> None:  # fmt: skip
    where = settings.experiment_dir
    where.mkdir(parents=True)
    if native is not None:
        (where / "native_devices.yaml").write_text(native)
    if panel is not None:
        (where / "panel_devices.yaml").write_text(panel)
    with _room(settings, down=True) as c:  # no DB device either: the module is silent
        html = c.get("/").text.split('id="directory"')[1]
    assert (
        f'<div class="state" data-state="{state}">\n      <div class="t">{title}</div>'
        in html
    )
    assert "GEECS DB devices not listed: the module did not answer" in html
    assert (
        'id="directory-filter"' not in html and 'id="directory-chip"' not in html
    )  # fmt: skip  (no "0/0 live" beside the pane)
