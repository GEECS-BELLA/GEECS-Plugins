"""``geecs-canopy squirrel-sync``: the derivation and the reconcile, each test one rule (lane P's brief)."""

from __future__ import annotations

import pytest
from geecs_schemas import Role

from geecs_canopy.cli import squirrel_sync
from geecs_canopy.native import NativeSource, Resolved
from geecs_canopy.squirrel import derive
from geecs_canopy.testing import EXPERIMENT, FakeSquirrel, write_documents

MOT = "test:mot1:m1.VAL"
#: A DB-served device, as GatewaySource.resolve builds one (a role per variable, :SP written).
DB = Resolved("U_Stage", "Stage", "test:u_stage:", (), (Role.model_construct(name="Pos", suffix="Pos", type="setpoint", label="Pos", writable=True, write="Pos:SP"),), (), None, frozenset({"Pos"}), "gateway")  # fmt: skip


@pytest.fixture
def bench(tmp_path):
    write_documents(tmp_path)
    devices = NativeSource(EXPERIMENT, tmp_path / "scanner_configs" / "experiments").resolve()  # fmt: skip
    env = {"GEECS_CANOPY_EXPERIMENT": EXPERIMENT, "GEECS_CANOPY_CONFIGS_ROOT": str(tmp_path), "GEECS_CANOPY_STATE_DIR": str(tmp_path), "GEECS_CANOPY_SQUIRREL_URL": "http://canopy.test", "GEECS_CANOPY_SQUIRREL_KEY": "service-write-key"}  # fmt: skip
    return devices, env, FakeSquirrel()


def test_the_derivation_names_setpoint_readback_device_role_and_tags(bench):
    rows = derive([*bench[0], DB])
    motor, supply, db = (
        rows[MOT],
        rows["bench:ps1:current:SP"],
        rows["test:u_stage:Pos:SP"],
    )
    assert (motor.readback, motor.device, motor.role) == ("test:mot1:m1.RBV", "Mot1", "m1.val")  # fmt: skip
    assert motor.tags == {
        ("family", "motor_record"),
        ("source", "native"),
        ("axis", "m1"),
    }
    assert (supply.readback, supply.role, supply.tags) == ("bench:ps1:current", "current_sp", {("family", "easydriver"), ("source", "native")})  # fmt: skip
    assert (db.readback, db.role, db.tags) == ("test:u_stage:Pos", "Pos", {("family", "Stage"), ("source", "gateway")})  # fmt: skip
    assert "bench:ps1:reset" not in rows and not any(
        e.role == "reset" for e in rows.values()
    )  # a command is a press, not a setting
    assert all(e.abs_tolerance == 0.0 for e in rows.values())


async def test_the_sync_adds_updates_removes_and_checks(bench, capsys):
    _, env, squirrel = bench
    hand = {"id": "a" * 32, "setpointAddress": "hand:made:SP", "readbackAddress": None, "device": "x", "description": None, "absTolerance": 0.01, "relTolerance": 0.0, "tags": []}  # fmt: skip
    stale = {"id": "b" * 32, "setpointAddress": MOT, "readbackAddress": "test:mot1:m1.RBV", "device": "simmotor1", "description": "axis m1", "absTolerance": 0.01, "relTolerance": 0.0, "tags": []}  # fmt: skip
    squirrel.pvs = {hand["id"]: hand, stale["id"]: stale}
    assert await squirrel_sync(check=True, env=env, transport=squirrel.transport()) == 1
    out = capsys.readouterr().out
    assert "- hand:made:SP" in out and f"~ {MOT}  device, description, tolerance, tags" in out and "would change" in out  # fmt: skip
    assert (
        squirrel.pvs == {hand["id"]: hand, stale["id"]: stale} and not squirrel.groups
    )  # --check wrote nothing
    assert (
        await squirrel_sync(check=False, env=env, transport=squirrel.transport()) == 0
    )
    by_sp = {p["setpointAddress"]: p for p in squirrel.pvs.values()}
    assert (
        "hand:made:SP" not in by_sp and by_sp[MOT]["id"] == stale["id"]
    )  # updated in place, not recreated
    assert (by_sp[MOT]["device"], by_sp[MOT]["description"], by_sp[MOT]["absTolerance"]) == ("Mot1", "m1.val", 0.0)  # fmt: skip
    assert {t["name"] for t in by_sp[MOT]["tags"]} == {"motor_record", "native", "m1"}
    assert "bench:ps1:current:SP" in by_sp and all(k == "service-write-key" for k in squirrel.keys)  # fmt: skip
    assert (
        await squirrel_sync(check=True, env=env, transport=squirrel.transport()) == 0
    )  # idempotent


async def test_a_changed_readback_replaces_the_row_and_no_key_is_usage(bench):
    _, env, squirrel = bench
    old = {"id": "c" * 32, "setpointAddress": MOT, "readbackAddress": "test:mot1:m1.OLD", "device": "Mot1", "description": "m1.val", "absTolerance": 0.0, "relTolerance": 0.0, "tags": []}  # fmt: skip
    squirrel.pvs = {old["id"]: old}
    assert (
        await squirrel_sync(check=False, env=env, transport=squirrel.transport()) == 0
    )
    (row,) = [p for p in squirrel.pvs.values() if p["setpointAddress"] == MOT]
    assert row["id"] != old["id"] and row["readbackAddress"] == "test:mot1:m1.RBV"
    assert await squirrel_sync(check=False, env={k: v for k, v in env.items() if "KEY" not in k}) == 2  # fmt: skip
