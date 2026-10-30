"""The preset routes over Squirrel's storage: each test one rule (lane P's brief)."""

import asyncio

import httpx
import pytest

from geecs_canopy.presets import describe, parse, same
from geecs_canopy.testing import FakeSquirrel, live, make_env

MOT, MOT_RB = "test:mot1:m1.VAL", "test:mot1:m1.RBV"
SP, RB = "bench:ps1:current:SP", "bench:ps1:current"


@pytest.fixture
async def env(tmp_path):
    squirrel = FakeSquirrel()
    env = make_env(tmp_path, squirrel=squirrel)
    env.squirrel = squirrel
    env.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app), base_url="http://t")  # fmt: skip
    env.hare.meta[SP] = {"controlLow": 0.0, "controlHigh": 5.0}
    env.hare.meta[MOT] = {"controlLow": -10.0, "controlHigh": 10.0}
    for pv, value in ((MOT, 2.0), (MOT_RB, 2.0), (SP, 1.0), (RB, 1.0)):
        env.hare.values[pv] = live(value)
    # the registry as squirrel-sync leaves it, with a tolerance on the motor and a retired row
    for sp, rb, device, role, tol in ((MOT, MOT_RB, "Mot1", "m1.val", 0.01), (SP, RB, "ps1", "current_sp", 0.0), ("gone:SP", None, "Gone", "x", 0.0)):  # fmt: skip
        squirrel.handle(httpx.Request("POST", "http://s/squirrel/v1/pvs", json={"setpointAddress": sp, "readbackAddress": rb, "device": device, "description": role, "absTolerance": tol}))  # fmt: skip
    squirrel.values = {MOT: 2.0, SP: 1.0}  # gone:SP is unreachable at save time
    return env


async def save(env, name="bench-a", **scope):
    return await env.client.post("/v1/presets", json={"name": name, "operator": "ann", "scope": scope or {"devices": ["Mot1", "ps1"]}})  # fmt: skip


async def ended(env, run_id):
    for _ in range(500):
        body = (await env.client.get(f"/v1/actions/runs/{run_id}")).json()
        if body["state"] in ("done", "failed", "stopped"):
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(body)


def test_the_description_carries_operator_scope_golden_and_a_persons_lines():
    text = describe("ann", {"devices": ["Mot1", "ps1"], "screen": []}, golden=True, extra=["after the realignment"])  # fmt: skip
    assert (
        text == "saved by ann\nscope: devices Mot1, ps1\ngolden\nafter the realignment"
    )
    assert parse(text) == (
        "ann",
        {"devices": ["Mot1", "ps1"]},
        True,
        ["after the realignment"],
    )
    assert parse("lane C1") == (None, {}, False, ["lane C1"])
    assert (
        same(1.0, 1.004, 0.01, 0)
        and not same(1.0, 1.02, 0.01, 0)
        and same("on", "on", 0, 0)
    )


async def test_a_save_records_scope_and_operator_and_one_journal_row(env):
    r = await save(env)
    assert r.status_code == 201, r.json()
    assert (r.json()["name"], r.json()["operator"], r.json()["scope"], r.json()["golden"]) == ("bench-a", "ann", {"devices": ["Mot1", "ps1"]}, False)  # fmt: skip
    listed = (await env.client.get("/v1/presets")).json()["presets"]
    assert [p["name"] for p in listed] == ["bench-a"] and listed[0]["pvCount"] == 3
    rows = (await env.client.get("/v1/journal")).json()
    assert [(r["device"], r["role"], r["value"], r["operator"], r["outcome"]) for r in rows] == [("squirrel", "snapshot", "bench-a", "ann", "attempted"), ("squirrel", "snapshot", "bench-a", "ann", "accepted")]  # fmt: skip
    assert rows[1]["attempt"] == rows[0]["id"]


async def test_the_diff_reads_tolerance_unsaved_and_retired_rows(env):
    sid = (await save(env, devices=["Mot1", "ps1", "Gone"])).json()["id"]
    env.hare.values[MOT] = live(2.005)  # inside ±0.01
    env.hare.values[SP] = live(1.5)
    body = (await env.client.get(f"/v1/presets/{sid}/diff")).json()
    by = {(r["device"], r["role"]): r for r in body["rows"]}
    assert (by[("Mot1", "val")]["axis"], by[("Mot1", "val")]["word"], by[("Mot1", "val")]["tolerance"]) == ("m1", "same", "±0.01")  # fmt: skip
    assert (by[("ps1", "current_sp")]["word"], by[("ps1", "current_sp")]["saved"], by[("ps1", "current_sp")]["live"]) == ("differs", 1.0, 1.5)  # fmt: skip
    assert by[("Gone", "x")]["word"] == "retired" and body["counts"] == {"differs": 1, "unsaved": 0, "retired": 1, "unread": 0, "unsupported": 0}  # fmt: skip
    env.squirrel.values = {MOT: 2.0}  # the supply unreachable at save time
    sid2 = (await save(env, name="b", devices=["ps1"])).json()["id"]
    (row,) = (await env.client.get(f"/v1/presets/{sid2}/diff")).json()["rows"]
    assert row["word"] == "unsaved" and row["saved"] is None
    subset = (await env.client.get(f"/v1/presets/{sid}/diff", params={"devices": ["Mot1"]})).json()  # fmt: skip
    assert [r["device"] for r in subset["rows"]] == ["Mot1"]


async def test_a_restore_is_a_run_of_the_differing_rows_motion_last(env):
    sid = (await save(env)).json()["id"]
    env.hare.values[MOT], env.hare.values[SP] = live(3.0), live(1.0)
    nothing = await env.client.post(f"/v1/presets/{sid}/restore", json={"operator": "ann", "devices": ["ps1"]})  # fmt: skip
    assert (nothing.status_code, nothing.json()["code"]) == (409, "nothing_to_restore") and "1 same" in nothing.json()["detail"]  # fmt: skip
    unknown = await env.client.post(f"/v1/presets/{sid}/restore", json={"operator": "ann", "devices": ["Gone"]})  # fmt: skip
    assert (unknown.status_code, unknown.json()["code"]) == (422, "not_found")
    env.hare.values[SP] = live(1.5)
    started = await env.client.post(
        f"/v1/presets/{sid}/restore", json={"operator": "ann"}
    )
    assert started.status_code == 202, started.json()
    body = await ended(env, started.json()["runId"])
    assert body["plan"] == f"restore:{sid}" and body["state"] == "done", body
    assert [(s["device"], s["role"], s.get("axis")) for s in body["steps"]] == [("ps1", "current_sp", None), ("Mot1", "val", "m1")]  # fmt: skip
    assert [p[:2] for p in env.hare.puts] == [
        (SP, 1.0),
        (MOT, 2.0),
    ]  # the supply before the move; only what differed
    assert env.squirrel.restores == []  # Squirrel's own restore route is never called
    rows = (await env.client.get("/v1/journal")).json()
    assert {r["run"] for r in rows if r["device"] != "squirrel"} == {body["id"]}


async def test_a_hazard_row_needs_the_confirm_and_a_subset_restores_alone(env):
    (env.folder / "operator_policy.yaml").write_text("grants:\n  - {operators: [ann], families: [motor_record, easydriver]}\nhazards:\n  - {devices: [Mot1], roles: [val]}\n")  # fmt: skip
    sid = (await save(env)).json()["id"]
    env.hare.values[MOT], env.hare.values[SP] = live(3.0), live(1.5)
    refused = await env.client.post(
        f"/v1/presets/{sid}/restore", json={"operator": "ann"}
    )
    assert (refused.status_code, refused.json()["code"]) == (428, "confirm_required") and env.hare.puts == []  # fmt: skip
    only = await env.client.post(f"/v1/presets/{sid}/restore", json={"operator": "ann", "devices": ["ps1"]})  # fmt: skip
    assert (
        only.status_code == 202
        and (await ended(env, only.json()["runId"]))["state"] == "done"
    )
    assert [p[:2] for p in env.hare.puts] == [(SP, 1.0)]


async def test_one_golden_per_scope_and_a_persons_lines_survive(env):
    a = (await save(env, name="a")).json()["id"]
    b = (await save(env, name="b")).json()["id"]
    other = (await save(env, name="c", families=["motor_record"])).json()["id"]
    env.squirrel.snapshots[a]["description"] += "\nkept by hand"

    def gold(sid, op="ann"):
        return env.client.post(f"/v1/presets/{sid}/golden", json={"operator": op})

    assert (await gold(a)).json()["golden"] is True
    await gold(other)
    assert (await gold(b)).json()["golden"] is True
    marked = {
        p["id"]
        for p in (await env.client.get("/v1/presets")).json()["presets"]
        if p["golden"]
    }
    assert marked == {b, other}  # a lost the mark to b; the other scope keeps its own
    assert (
        env.squirrel.snapshots[a]["description"]
        == "saved by ann\nscope: devices Mot1, ps1\nkept by hand"
    )
    rows = (await env.client.get("/v1/journal")).json()
    assert [(r["role"], r["value"]) for r in rows if r["role"] == "golden"] == [("golden", "a"), ("golden", "c"), ("golden", "b")]  # fmt: skip
    assert "a unmarked" in rows[-1]["detail"]
    assert (await gold(b, "x\ngolden")).status_code == 422  # the grammar's characters


async def test_a_screen_scope_reads_the_screen_file_and_no_scope_restores_a_subset_only(
    env,
):
    folder = env.folder / "screens"
    folder.mkdir()
    (folder / "bench.yaml").write_text("schema_version: 1\ntitle: Bench\ngroups:\n  - title: A\n    at: [1, 1]\n    widgets:\n      - {kind: setpoint, bind: {device: Mot1, role: val, axis: m1}}\n      - {kind: readout, bind: {pv: bare:pv}}\n")  # fmt: skip
    presets = env.app.state.presets
    assert await presets.scope_devices({"families": ["motor_record"]}) == {"Mot1"}
    assert await presets.scope_devices({"screen": ["bench"]}) == {"Mot1"}
    assert await presets.scope_devices({}) is None
    with pytest.raises(Exception, match="cannot be read"):
        await presets.scope_devices({"screen": ["missing"]})
    sid = (await save(env, name="bench", screen=["bench"])).json()["id"]
    env.hare.values[MOT], env.hare.values[SP] = live(3.0), live(1.5)
    body = (await env.client.get(f"/v1/presets/{sid}/diff")).json()
    assert [r["device"] for r in body["rows"]] == [
        "Mot1"
    ]  # the screen's devices, resolved here
    env.squirrel.snapshots[sid]["description"] = (
        "lane C1"  # a snapshot saved outside the module: no scope
    )
    bare = await env.client.post(f"/v1/presets/{sid}/restore", json={"operator": "ann"})
    assert (bare.status_code, bare.json()["code"]) == (422, "scope_required") and env.hare.puts == []  # fmt: skip
    assert (await env.client.get(f"/v1/presets/{sid}/diff")).json()["counts"][
        "differs"
    ] == 2  # a diff of no scope reads everything


async def test_without_squirrel_the_routes_say_so(tmp_path):
    env = make_env(tmp_path)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=env.app), base_url="http://t"
    )
    r = await client.get("/v1/presets")
    assert (r.status_code, r.json()["code"]) == (503, "squirrel_unconfigured")
