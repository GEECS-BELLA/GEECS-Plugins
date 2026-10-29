"""The device module: each test pins one rule (Appendix A of Planning/epics_estate.md, the brief's gates)."""

import asyncio
import math
from types import SimpleNamespace

import httpx
import pytest
from geecs_schemas import JournalEntry

from geecs_canopy import canopy_surface
from geecs_canopy.gates import PutRequest, put_role
from geecs_canopy.hare import HareRefused
from geecs_canopy.journal import Journal
from geecs_canopy.module import Module, ModuleSettings
from geecs_canopy.routes import build_app

REGISTRY = """\
schema_version: 1
devices:
  Cam1: {kind: aravis_camera, trigger_source: line0}
  Mot1: {kind: epics_motor, axes: [m1]}
"""
PANELS = """\
schema_version: 1
devices:
  ps1: {family: easydriver, prefix: "bench:ps1:", allow: [current, reset]}
"""
POLICY = """\
grants:
  - {operators: [ann], families: [ad_camera_chain, motor_record, easydriver]}
"""
KEY = SimpleNamespace(
    name="display", read=True, write=False
)  # an operator's key: no Canopy write


def live(value, **wire):
    return {
        "value": value,
        "connected": True,
        "severity": "NO_ALARM",
        "alarmStatus": 0,
    } | wire


class FakeHare:
    def __init__(self):
        self.values, self.meta, self.puts, self.clamp = {}, {}, [], {}
        self.refuse, self.slow = None, False

    async def get(self, pv):
        return self.values.get(pv, {"pv": pv, "value": None, "connected": False})

    async def describe(self, pv):
        return self.meta.get(pv, {"pv": pv})

    async def put(self, pv, value, *, wait, timeout):
        if self.refuse:
            await asyncio.sleep(timeout if self.slow else 0)
            raise self.refuse
        self.puts.append((pv, value, wait, timeout))
        self.values[pv] = live(min(value, self.clamp.get(pv, value)))

    async def watch(self, pvs, on_frame):
        self.on_frame = on_frame
        return pvs


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "scanner_configs" / "experiments" / "Test"
    folder.mkdir(parents=True)
    (folder / "native_devices.yaml").write_text(REGISTRY)
    (folder / "panel_devices.yaml").write_text(PANELS)
    (folder / "operator_policy.yaml").write_text(POLICY)
    clock = [100.0]
    settings = ModuleSettings("Test", tmp_path, tmp_path, 3.0, settle_timeout_s=0.05)
    hare = FakeHare()
    module = Module(
        settings, hare, Journal(tmp_path / "j.sqlite3"), clock=lambda: clock[0]
    )
    app = build_app(module, read_access=lambda: KEY)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    )
    return SimpleNamespace(
        module=module, hare=hare, client=client, clock=clock, folder=folder
    )


async def pv(env, device, role, axis=None, write=False):
    d = (await env.module.devices())[device]
    await asyncio.sleep(0)  # the subscription runs as a task
    return d.pv(next(r for r in d.roles if r.name == role), axis, write=write)


async def roles(env, device):
    body = (await env.client.get(f"/v1/devices/{device}")).json()
    return body, {(r["role"], r["axis"]): r for r in body["roles"]}


async def put(env, device, **body):
    return await env.client.post(
        f"/v1/devices/{device}/put", json={"operator": "ann", **body}
    )


async def test_devices_come_from_the_documents_and_the_families(env):
    body = (await env.client.get("/v1/devices")).json()
    found = {(d["name"], d["family"], tuple(d["axes"])) for d in body["devices"]}
    assert found == {
        ("Cam1", "ad_camera_chain", ()),
        ("Mot1", "motor_record", ("m1",)),
        ("ps1", "easydriver", ()),
    }


async def test_a_changing_value_with_a_still_counter_is_stale(env):  # rule 1
    mean, counter = await pv(env, "Cam1", "mean"), await pv(env, "Cam1", "stat_counter")
    flag = await pv(env, "Cam1", "compute_statistics")
    env.hare.on_frame({mean: live(1.0), counter: live(5), flag: live(1)})
    env.clock[0] += 4
    env.hare.on_frame(
        {mean: live(2.0), counter: live(5)}
    )  # the value moves, the producer does not
    by_role = (await roles(env, "Cam1"))[1]
    assert (
        by_role[("mean", None)]["verdict"]
        == by_role[("stat_counter", None)]["verdict"]
        == "stale"
    )
    env.hare.on_frame({counter: live(6)})
    assert (await roles(env, "Cam1"))[1][("mean", None)]["verdict"] == "live"


async def test_no_evidence_yet_is_unknown_not_live(env):
    counter = await pv(env, "Cam1", "stat_counter")
    flag = await pv(env, "Cam1", "compute_statistics")
    env.hare.on_frame({await pv(env, "Cam1", "mean"): live(1.0), counter: live(5)})
    env.hare.on_frame({flag: live(1)})
    assert (await roles(env, "Cam1"))[1][("mean", None)]["verdict"] == "unknown"


async def test_a_value_whose_compute_flag_is_off_reads_off(env):  # rule 1
    flag = await pv(env, "Cam1", "compute_centroid")
    env.hare.on_frame({await pv(env, "Cam1", "cx"): live(3.0), flag: live(0)})
    assert (await roles(env, "Cam1"))[1][("cx", None)]["verdict"] == "off"


async def test_a_computed_value_whose_flag_is_unavailable_is_unknown(env):  # rule 1
    cx, flag = await pv(env, "Cam1", "cx"), await pv(env, "Cam1", "compute_centroid")
    counter = await pv(env, "Cam1", "stat_counter")
    env.hare.on_frame({cx: live(3.0), counter: live(1), flag: live(0)})
    assert (await roles(env, "Cam1"))[1][("cx", None)]["verdict"] == "off"
    env.hare.on_frame({counter: live(2), flag: {"value": None, "connected": False}})
    assert (await roles(env, "Cam1"))[1][("cx", None)]["verdict"] == "unknown"


async def test_disconnected_is_unreachable_and_a_fault_outranks_it(env):  # rule 9
    current, fault = await pv(env, "ps1", "current"), await pv(env, "ps1", "fault")
    env.hare.on_frame({current: {"value": None, "connected": False}})
    body, by_role = await roles(env, "ps1")
    assert by_role[("current", None)]["verdict"] == "unreachable"
    assert body["state"] == "unknown"  # a dropped field is not a stale or failed device
    env.hare.on_frame({fault: live(1)})
    assert (await roles(env, "ps1"))[0]["state"] == "error"


async def test_an_invalid_comm_alarm_reads_not_answering(env):
    reading = live(
        0.0, severity="INVALID", alarmStatus=9
    )  # COMM: the IOC answers, the supply does not
    current = await pv(env, "ps1", "current")
    env.hare.on_frame({current: reading})
    assert (await roles(env, "ps1"))[1][("current", None)]["verdict"] == "not answering"


async def test_a_never_processed_record_is_unknown_not_error(env):
    current, sp = await pv(env, "ps1", "current"), await pv(env, "ps1", "current_sp")
    env.hare.on_frame(
        {current: live(0.0), sp: live(0.0, severity="INVALID", alarmStatus=17)}
    )
    body, by_role = await roles(env, "ps1")
    assert by_role[("current_sp", None)]["verdict"] == "unknown"  # UDF: no value yet
    assert by_role[("current", None)]["verdict"] == "live"


async def test_nan_travels_as_null(env):  # rule 10
    voltage = await pv(env, "ps1", "voltage")
    env.hare.on_frame({voltage: live(math.nan)})
    assert (await roles(env, "ps1"))[1][("voltage", None)]["value"] is None


async def test_without_a_policy_every_put_is_denied(env):
    (env.folder / "operator_policy.yaml").unlink()
    response = await put(env, "ps1", role="reset", value=1)
    assert (response.status_code, response.json()["code"]) == (403, "not_permitted")
    assert env.hare.puts == []


async def test_a_role_outside_the_allow_list_is_denied_before_hare(env):  # rule 8
    response = await put(env, "ps1", role="output_sp", value=1, confirm=True)
    assert (response.status_code, response.json()["code"]) == (403, "not_allowed")
    assert env.hare.puts == []


async def test_a_hazard_role_needs_the_confirm(env):  # rule 18
    denied = await put(env, "Mot1", role="homf", axis="m1", value=1)
    assert (denied.status_code, denied.json()["code"]) == (428, "confirm_required")
    assert (
        await put(env, "Mot1", role="homf", axis="m1", value=1, confirm=True)
    ).status_code == 200


async def test_a_held_or_unreadable_run_lock_denies(env):  # rule 6
    lock = await pv(env, "Cam1", "lock")
    unread = await put(env, "Cam1", role="exposure", value=0.1)
    assert (unread.status_code, unread.json()["code"]) == (423, "run_lock_unreadable")
    env.hare.meta[lock] = {"enumChoices": ["Free", "Locked"]}
    env.hare.values[lock] = live(1)
    held = await put(env, "Cam1", role="exposure", value=0.1)
    assert (held.status_code, held.json()["code"]) == (423, "run_lock_held")
    env.hare.values[lock] = live(0)
    assert (await put(env, "Cam1", role="exposure", value=0.1)).status_code == 200


async def test_the_bounds_gate_refuses_before_hare_and_fails_closed(env):  # rule 11
    sp = await pv(env, "ps1", "current_sp")
    env.hare.meta[sp] = {"controlLow": 0.0, "controlHigh": 10.0}
    beyond = await put(env, "ps1", role="current_sp", value=11.0)
    assert (beyond.status_code, beyond.json()["code"]) == (422, "out_of_bounds")
    env.hare.meta[sp] = {"controlLow": 0.0, "controlHigh": 0.0}  # unset DRVL/DRVH
    unset = await put(env, "ps1", role="current_sp", value=1.0)
    assert (unset.status_code, unset.json()["code"]) == (422, "bounds_unknown")
    assert env.hare.puts == []


async def test_an_enum_is_checked_against_the_record_choices(env):
    env.hare.meta[await pv(env, "Cam1", "transform", write=True)] = {
        "enumChoices": ["None", "Rot90", "Flip"]
    }
    assert (await put(env, "Cam1", role="transform", value="Rot45")).status_code == 422
    assert (
        await put(env, "Cam1", role="transform", value="Flip")
    ).status_code == 422  # not the family's
    nan = b'{"role": "transform", "value": NaN, "operator": "ann"}'  # JSON.parse refuses it; Python does not
    headers = {"content-type": "application/json"}
    response = await env.client.post(
        "/v1/devices/Cam1/put", content=nan, headers=headers
    )
    assert response.status_code == 422
    assert (await put(env, "Cam1", role="transform", value=1)).status_code == 200


async def test_a_hare_refusal_keeps_hares_status_and_code(env):
    env.hare.refuse = HareRefused(502, "pv_write_failed", "Write access denied")
    response = await put(env, "ps1", role="reset", value=1)
    assert response.status_code == 502
    assert response.json() | {"journalId": None} == {
        "outcome": "refused", "device": "ps1", "role": "reset", "axis": None, "journalId": None,
        "code": "pv_write_failed", "detail": "Write access denied",
    }  # fmt: skip


async def test_a_clamped_write_reports_the_value_the_record_took(
    env,
):  # rule 11: the clamp is silent
    sp = await pv(env, "ps1", "current_sp")
    env.hare.meta[sp] = {"controlLow": 0.0, "controlHigh": 10.0}
    env.hare.clamp[sp] = 4.0
    body = (await put(env, "ps1", role="current_sp", value=5.0)).json()
    assert (body["outcome"], body["requested"], body["value"]) == ("accepted", 5.0, 4.0)
    assert body["detail"] == "clamped: the record took 4.0"


async def test_a_role_that_settles_gets_the_settle_timeout(
    env,
):  # C3: not Hare's 5 s connect timeout
    await put(env, "Mot1", role="twf", axis="m1", value=1)
    await put(env, "ps1", role="reset", value=1)
    assert [(wait, timeout) for *_, wait, timeout in env.hare.puts] == [
        (True, 0.05),
        (True, 5.0),
    ]


async def test_a_move_outlasting_its_timeout_is_accepted_not_refused(env):
    env.hare.refuse, env.hare.slow = HareRefused(502, "pv_write_failed", "t/o"), True
    unconnected = (await put(env, "Mot1", role="twf", axis="m1", value=1)).json()
    assert unconnected["outcome"] == "refused"  # the timeout also covers connecting
    env.hare.values[await pv(env, "Mot1", "twf", "m1", write=True)] = live(1)
    body = (await put(env, "Mot1", role="twf", axis="m1", value=1)).json()
    assert (body["outcome"], body["detail"][:10]) == ("accepted", "dispatched")


async def test_a_policy_that_is_not_yaml_denies_and_is_journaled(env):
    (env.folder / "operator_policy.yaml").write_text("grants:\n\t- bad")
    response = await put(env, "ps1", role="reset", value=1)
    assert (response.status_code, response.json()["code"]) == (403, "not_permitted")
    assert env.module.journal.last(1)[0].outcome == "denied"


async def test_the_put_path_is_a_function_and_every_attempt_is_journaled(
    env,
):  # a restore's row
    for value in (1, 2):
        await put_role(
            env.module, PutRequest("ps1", "reset", value, "ann", key_name="ops")
        )
    rows = (await env.client.get("/v1/journal?entries=1")).json()
    entry = JournalEntry.model_validate(rows[0])
    assert (len(rows), entry.id, entry.value, entry.outcome) == (1, 3, 2, "refused")


async def test_no_put_reaches_hare_without_its_attempt_row(env):
    env.module.journal._db.execute("PRAGMA query_only=ON")
    response = await put(env, "ps1", role="reset", value=1)
    assert (response.status_code, response.json()["code"]) == (
        503,
        "journal_unavailable",
    )
    assert env.hare.puts == []


async def test_an_accepted_put_leaves_its_attempt_and_its_outcome(env):
    await put(env, "ps1", role="reset", value=1)
    attempt, outcome = env.module.journal.last(2)
    assert (attempt.outcome, outcome.outcome) == ("attempted", "accepted")
    assert (outcome.attempt, attempt.attempt) == (attempt.id, None)


async def test_an_operator_put_needs_no_canopy_write(env):  # Hare/Squirrel 403: live
    await put(env, "ps1", role="reset", value=1)
    row = env.module.journal.last(1)[0]
    assert (row.outcome, row.key_name) == ("accepted", "display")


def test_a_missing_canopy_name_refuses_to_start_naming_the_rollback():
    with pytest.raises(
        canopy_surface.SurfaceDrift, match=r"json:nope.*uvicorn canopy\.main:app"
    ):
        canopy_surface.self_test(
            ["json:dumps", "json:nope"], {}, installed="0.0.2.dev211+g0a5ee23b6"
        )


def test_a_pin_bump_without_reverifying_the_surface_refuses_to_start():
    with pytest.raises(canopy_surface.SurfaceDrift, match="verified at"):
        canopy_surface.self_test(
            [], {}, image="ghcr.io/slaclab/canopy@sha256:0000", installed="g0a5ee23"
        )
