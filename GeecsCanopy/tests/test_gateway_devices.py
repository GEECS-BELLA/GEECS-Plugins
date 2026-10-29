"""The DB-served devices through the module: each test pins one rule of the gateway device model."""

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from geecs_core.db.geecs_db import GeecsDb

from geecs_canopy.gateway import GatewaySource
from geecs_canopy.journal import Journal
from geecs_canopy.module import Module, ModuleSettings
from geecs_canopy.routes import build_app
from tests.test_module import KEY, FakeHare, live


def row(name, settable=False, vartype="numeric", choices=None, lo=None, hi=None):
    return {"name": name, "settable": settable, "variabletype": vartype, "choices": choices, "min": lo, "max": hi, "units": "", "alias": ""}  # fmt: skip


VARIABLES = {
    "U_PLC": [
        row("DO.Ch1", True, "choice", "on,off"),
        row("DI.Ch1", False, "choice", "on,off"),
    ],  # fmt: skip
    "U_Stage": [
        row("Position.Axis 1", True, lo=-10.0, hi=10.0),
        row("Velocity", True),
        row("Comment", True, "string"),
    ],  # fmt: skip
    "U_Off": [
        row("Current", True, lo=0.0, hi=1.0)
    ],  # disabled: not among the endpoints
}
TYPES = {"U_PLC": "PLC Digital Output", "U_Stage": "Newport ESP"}
POLICY = """\
grants:
  - {operators: [ann], families: [PLC Digital Output, Newport ESP]}
hazards:
  - {devices: [U_PLC], roles: [DO.Ch1]}
"""
PREFIX = "test:u_plc:"


@pytest.fixture
def db(monkeypatch):
    state = SimpleNamespace(down=False)

    def answer(value):
        def call(cls, experiment, **_):
            if state.down:
                raise ConnectionError("no route to the DB")
            return value

        return classmethod(call)

    monkeypatch.setattr(GeecsDb, "get_experiment_devices", answer({"U_PLC": ("h", 1), "U_Stage": ("h", 2)}))  # fmt: skip
    monkeypatch.setattr(GeecsDb, "get_experiment_device_variables", answer(VARIABLES))
    monkeypatch.setattr(
        GeecsDb, "get_subscribed_variables", answer({"U_PLC": ["DI.Ch1"]})
    )  # get=yes: served read-only
    monkeypatch.setattr(GeecsDb, "get_experiment_device_types", answer(TYPES))
    return state


@pytest.fixture
async def env(tmp_path, db):
    folder = tmp_path / "scanner_configs" / "experiments" / "Test"
    folder.mkdir(parents=True)
    (folder / "operator_policy.yaml").write_text(POLICY)
    clock, hare = [100.0], FakeHare()
    settings = ModuleSettings("Test", tmp_path, tmp_path, settle_timeout_s=0.05)
    source = GatewaySource("Test", "mysql://ro:pw@db.invalid:3306/geecs")
    journal = Journal(tmp_path / "j.sqlite3")
    module = Module(settings, hare, journal, clock=lambda: clock[0], gateway=source.resolve)  # fmt: skip
    await module.start()
    transport = httpx.ASGITransport(app=build_app(module, read_access=lambda: KEY))
    client = httpx.AsyncClient(transport=transport, base_url="http://t")
    return SimpleNamespace(module=module, hare=hare, client=client, clock=clock, db=db)


async def put(env, device, **body):
    return await env.client.post(f"/v1/devices/{device}/put", json={"operator": "ann", **body})  # fmt: skip


async def roles(env, device):
    body = (await env.client.get(f"/v1/devices/{device}")).json()
    return body, {r["role"]: r for r in body["roles"]}


async def test_the_roster_carries_the_db_devices_by_type_and_says_when_it_read(env):
    body = (await env.client.get("/v1/devices")).json()
    found = {(d["name"], d["family"], d["source"]) for d in body["devices"]}
    assert found == {("U_PLC", "PLC Digital Output", "gateway"), ("U_Stage", "Newport ESP", "gateway")}  # fmt: skip
    assert body["sources"]["gateway"]["error"] is None
    assert body["sources"]["gateway"]["readAt"]


async def test_a_role_is_the_variable_name_typed_by_the_served_set(env):
    _, plc = await roles(env, "U_PLC")
    assert (plc["DO.Ch1"]["type"], plc["DO.Ch1"]["writable"]) == ("enum", True)
    assert (
        plc["DO.Ch1"]["writePv"] == plc["DO.Ch1"]["pv"] + ":SP" == PREFIX + "do_ch1:SP"
    )
    assert (plc["DI.Ch1"]["writable"], plc["DI.Ch1"]["writePv"]) == (False, None)
    body, stage = await roles(env, "U_Stage")
    assert stage["Position.Axis 1"]["type"] == "setpoint"
    assert body["source"] == "gateway" and body["axes"] == []
    assert (await env.client.get("/v1/devices/U_Off")).status_code == 404  # disabled


async def test_only_the_connected_pv_is_watched_and_it_is_the_state(env):
    connected = "test:u_plc:connected"
    assert connected in env.module._watched
    assert not any(":do_ch1" in pv or ":position" in pv for pv in env.module._watched)
    env.hare.on_frame({connected: live(0)})  # Disconnected
    body, plc = await roles(env, "U_PLC")
    assert body["state"] == "unreachable" and plc["DO.Ch1"]["verdict"] is None
    env.hare.on_frame({connected: live(1)})
    assert (await roles(env, "U_PLC"))[0]["state"] == "live"


async def test_a_numeric_with_limits_writes_its_setpoint_and_waits_for_the_set(env):
    response = await put(env, "U_Stage", role="Position.Axis 1", value=2.5)
    assert response.json()["outcome"] == "accepted", response.json()
    assert env.hare.puts == [("test:u_stage:position_axis_1:SP", 2.5, True, 0.05)]
    response = await put(env, "U_Stage", role="Position.Axis 1", value=11)
    assert response.json()["code"] == "out_of_bounds"


async def test_a_numeric_without_limits_is_refused_naming_its_db_row(env):
    response = await put(env, "U_Stage", role="Velocity", value=1.0)
    assert response.json()["code"] == "bounds_unknown"
    assert "U_Stage's Velocity row in the GEECS DB" in response.json()["detail"]
    assert env.hare.puts == []


async def test_text_takes_text_and_a_read_only_variable_is_not_allowed(env):
    assert (await put(env, "U_Stage", role="Comment", value="hi")).json()["outcome"] == "accepted"  # fmt: skip
    response = await put(env, "U_PLC", role="DI.Ch1", value=1, confirm=True)
    assert response.json()["code"] == "not_allowed"


async def test_a_policy_hazard_demands_the_confirm_and_the_route_says_so(env):
    assert (await roles(env, "U_PLC"))[1]["DO.Ch1"]["confirm"] == "hazard"
    env.hare.meta[PREFIX + "do_ch1:SP"] = {"enumChoices": ["on", "off"]}
    response = await put(env, "U_PLC", role="DO.Ch1", value=1)
    assert (response.status_code, response.json()["code"]) == (428, "confirm_required")
    response = await put(env, "U_PLC", role="DO.Ch1", value=1, confirm=True)
    assert response.json()["outcome"] == "accepted"


async def test_a_grant_by_db_device_type_permits_and_another_type_does_not(env):
    devices = env.module._gateway_devices
    devices["U_Stage"] = replace(devices["U_Stage"], family="Thorlabs APT")
    response = await put(env, "U_Stage", role="Position.Axis 1", value=1.0)
    assert (response.status_code, response.json()["code"]) == (403, "not_permitted")


async def test_the_db_down_keeps_the_last_set_and_the_roster_says_so(env):
    env.db.down, env.clock[0] = True, env.clock[0] + 61
    await env.module.read_gateway()
    body = (await env.client.get("/v1/devices")).json()
    assert {d["name"] for d in body["devices"]} == {"U_PLC", "U_Stage"}
    assert "no route to the DB" in body["sources"]["gateway"]["error"]
    assert body["sources"]["gateway"]["readAt"]  # the last good read's
