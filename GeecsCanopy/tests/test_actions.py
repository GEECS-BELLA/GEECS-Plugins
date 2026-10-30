"""The action executor: each test pins one rule of a run (lane X's brief)."""

import asyncio

import httpx
import pytest
import yaml

from geecs_canopy.testing import live, make_env

SP, RB = "bench:ps1:current:SP", "bench:ps1:current"
SET = {"do": "set", "device": "ps1", "variable": "current_sp", "value": 1.0}


@pytest.fixture
def env(tmp_path):
    env = make_env(tmp_path)
    env.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app), base_url="http://t")  # fmt: skip
    env.hare.meta[SP] = {"controlLow": 0.0, "controlHigh": 5.0}
    env.hare.meta["test:mot1:m1.VAL"] = {"controlLow": -10.0, "controlHigh": 10.0}
    return env


def library(env, **plans):
    folder = env.folder / "action_library"
    folder.mkdir(exist_ok=True)
    doc = {"plans": {n: {"steps": s, "description": f"{n} plan"} for n, s in plans.items()}}  # fmt: skip
    (folder / "actions.yaml").write_text(yaml.safe_dump(doc))


async def run(env, name, **body):
    return await env.client.post(f"/v1/actions/{name}/run", json={"operator": "ann", **body})  # fmt: skip


async def ended(env, run_id):
    for _ in range(500):
        body = (await env.client.get(f"/v1/actions/runs/{run_id}")).json()
        if body["state"] in ("done", "failed", "stopped"):
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} never ended: {body}")


async def test_a_run_writes_through_the_gate_and_its_rows_carry_its_id(env):
    env.hare.values[RB] = live(1.0)
    env.hare.values["test:mot1:m1.RBV"] = live(2.0)
    move = {"do": "set", "device": "Mot1", "variable": "m1", "value": 2.0}
    check = {"do": "check", "device": "ps1", "variable": "current", "expected": 1}
    at = {"do": "check", "device": "Mot1", "variable": "m1", "expected": 2}
    library(env, park=[SET, move, {"do": "wait", "seconds": 0.01}, check, at])
    started = await run(env, "park")
    assert started.status_code == 202, started.json()
    body = await ended(env, started.json()["runId"])
    assert body["state"] == "done", body
    assert [(s["do"], s.get("role"), s.get("axis")) for s in body["steps"]][:2] == [("set", "current_sp", None), ("set", "val", "m1")]  # fmt: skip
    assert [p[:2] for p in env.hare.puts] == [(SP, 1.0), ("test:mot1:m1.VAL", 2.0)]
    rows = (await env.client.get("/v1/journal")).json()
    assert len(rows) == 4 and {r["run"] for r in rows} == {body["id"]}


async def test_the_precheck_refuses_before_any_journal_row(env):
    library(env, bad=[SET, {**SET, "device": "ghost"}], high=[{**SET, "value": 9.0}])
    refused = await run(env, "bad")
    assert (refused.status_code, refused.json()["code"], refused.json()["steps"]) == (404, "not_found", [2])  # fmt: skip
    assert (await run(env, "high")).json()["code"] == "out_of_bounds"
    assert (await run(env, "bad", operator="bob")).json()["code"] == "not_permitted"
    assert env.hare.puts == [] and (await env.client.get("/v1/journal")).json() == []


async def test_a_failed_check_stops_the_run_with_the_value_read(env):
    env.hare.values[RB] = live(0.5)
    check = {"do": "check", "device": "ps1", "variable": "current", "expected": 1}
    library(env, verify=[check, SET])
    body = await ended(env, (await run(env, "verify")).json()["runId"])
    assert body["state"] == "failed"
    assert (body["steps"][0]["code"], body["steps"][1]["state"]) == ("check_failed", "skipped")  # fmt: skip
    assert "0.5" in body["steps"][0]["detail"] and env.hare.puts == []


async def test_a_stop_lands_between_steps_and_one_plan_runs_once(env):
    library(env, slow=[{"do": "wait", "seconds": 5}, SET])
    run_id = (await run(env, "slow")).json()["runId"]
    again = await run(env, "slow")
    assert (again.status_code, again.json()["code"]) == (409, "already_running")
    await env.client.post(f"/v1/actions/runs/{run_id}/stop")
    body = await ended(env, run_id)
    assert body["state"] == "stopped" and [s["state"] for s in body["steps"]] == ["stopped", "skipped"]  # fmt: skip
    assert env.hare.puts == []


async def test_a_device_written_by_one_run_makes_another_wait(env):
    gate, order, put = asyncio.Event(), [], env.hare.put

    async def held(pv, value, *, wait, timeout):
        order.append(("start", value))
        if value == 1.0:
            await gate.wait()
        await put(pv, value, wait=wait, timeout=timeout)
        order.append(("end", value))

    env.hare.put = held
    library(env, one=[SET], two=[{**SET, "value": 2.0}])
    first = (await run(env, "one")).json()["runId"]
    await asyncio.sleep(0.05)
    second = (await run(env, "two")).json()["runId"]
    await asyncio.sleep(0.05)
    assert order == [("start", 1.0)]  # the second run's set waits for the device
    gate.set()
    await ended(env, first), await ended(env, second)
    assert order == [("start", 1.0), ("end", 1.0), ("start", 2.0), ("end", 2.0)]


async def test_a_hazard_role_anywhere_needs_the_confirm_once(env):
    policy = yaml.safe_load((env.folder / "operator_policy.yaml").read_text())
    policy["hazards"] = [{"devices": ["ps1"], "roles": ["current_sp"]}]
    (env.folder / "operator_policy.yaml").write_text(yaml.safe_dump(policy))
    library(env, twice=[SET, {**SET, "value": 2.0}])
    refused = await run(env, "twice")
    assert (refused.status_code, refused.json()["steps"]) == (428, [1, 2])
    listed = (await env.client.get("/v1/actions")).json()["plans"][0]
    assert [h["index"] for h in listed["hazardSteps"]] == [1, 2]
    body = await ended(env, (await run(env, "twice", confirm=True)).json()["runId"])
    assert body["state"] == "done" and len(env.hare.puts) == 2


async def test_a_library_that_does_not_validate_is_503(env):
    (env.folder / "action_library").mkdir()
    (env.folder / "action_library" / "actions.yaml").write_text(
        "plans: {x: {steps: []}}"
    )
    answer = await env.client.get("/v1/actions")
    assert (answer.status_code, answer.json()["code"]) == (503, "library_unavailable")
