"""The adapter's answers, the DB through a connection double, Canopy's contract."""

import sys
import types
from pathlib import Path

import pytest
from canopy.raccoon.adapters.base import DirectoryUnavailable, InvalidQuery
from geecs_core.db import geecs_db

from geecs_canopy_directory.adapter import GeecsDirectoryAdapter
from tests.canopy.raccoon.adapters import contract  # Canopy's conformance suite

REGISTRY = """\
schema_version: 1
devices:
  UC_VisaEBeam6: {kind: aravis_camera, trigger_source: line0}
  SimMotor1: {kind: epics_motor, axes: [m1, m2]}
"""


BAD = [("", {}), ("A|B", {}), ("*", {"element_type": "x"}), ("*", {"sort_by": "z"})]


def _var(i: int, name: str, set_: str, vtype: str) -> tuple:  # a devicetype row
    return ("U_Jet", i, name, "", None, None, set_, vtype, None, None, None, "", "")


DB_ROWS = {  # canned rows per query, keyed by a fragment of its SQL
    "d.ipaddress": [("U_Jet", "host", "1")],
    "edv.variablename": [("U_Jet", "Temperature")],
    "dtv.id": [
        _var(1, "Position", "yes", "numeric"),
        _var(2, "Temperature", "no", "numeric"),
        _var(3, "Notes", "no", "string"),
    ],
    "v.devicetype_variable_id": [],
    "d.devicetype": [("U_Jet", "Jet stage")],
}


def _patch_db(monkeypatch, *, fail: bool = False) -> None:
    class _Connection:  # connection and cursor in one
        def cursor(self) -> "_Connection":
            return self

        def execute(self, query: str, params: tuple) -> None:
            self.rows = next(rows for key, rows in DB_ROWS.items() if key in query)

        def fetchall(self) -> list[tuple]:
            return self.rows

        def close(self) -> None:
            pass

    def connect(_connector):
        if fail:
            raise ConnectionError("no route to the DB")
        return _Connection()

    connector = types.ModuleType("mysql.connector")
    monkeypatch.setitem(
        sys.modules, "mysql", types.SimpleNamespace(connector=connector)
    )
    monkeypatch.setitem(sys.modules, "mysql.connector", connector)
    monkeypatch.setattr(geecs_db, "_connect_mysql", connect)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "Undulator").mkdir()
    (tmp_path / "Undulator" / "native_devices.yaml").write_text(REGISTRY)
    return tmp_path


@pytest.fixture
async def adapter(root: Path, monkeypatch):
    _patch_db(monkeypatch)
    a = GeecsDirectoryAdapter("Undulator", root)
    yield a
    await a.close()


async def test_a_native_motor_lists_its_pvs_per_axis(adapter) -> None:
    pvs = await adapter.list_pvs("device:SimMotor1")
    assert pvs[:2] == ["undulator:simmotor1:m1.DESC", "undulator:simmotor1:m1.EGU"]
    assert "undulator:simmotor1:m2.RBV" in pvs and len(pvs) == 2 * 29
    elements = await adapter.list_elements("simmotor1:m2:*")
    assert elements[3] == "SimMotor1:m2:rbv"
    rbv = await adapter.list_pvs("element:SimMotor1:m2:rbv")
    assert rbv == ["undulator:simmotor1:m2.RBV"]


async def test_the_camera_lists_its_chain_with_write_records(adapter) -> None:
    pvs = await adapter.list_pvs("device:UC_VisaEBeam6")
    assert pvs[0] == "undulator:uc_visaebeam6:cam1:DetectorState_RBV"
    cam = "undulator:uc_visaebeam6:cam1:AcquireTime"  # the role's write record
    assert await adapter.list_pvs("element:UC_VisaEBeam6:exposure") == [
        cam + "_RBV",
        cam,
    ]
    assert await adapter.list_devices("*", tag="ad_camera_chain") == ["UC_VisaEBeam6"]


async def test_a_db_device_lists_the_gateways_served_pvs(adapter) -> None:
    assert await adapter.list_pvs("device:U_Jet") == [
        "undulator:u_jet:position",
        "undulator:u_jet:position:SP",
        "undulator:u_jet:temperature",
        "undulator:u_jet:connected",
    ]
    assert await adapter.list_devices("%", element_type="gateway") == ["U_Jet"]
    # a literal PV name answers itself: Raccoon's `exists` is built on this
    sp = "undulator:u_jet:position:SP"
    assert await adapter.list_pvs(sp) == [sp]


@pytest.mark.parametrize("pattern, kw", BAD)
async def test_a_bad_query_is_invalid(adapter, pattern, kw) -> None:
    with pytest.raises(InvalidQuery):
        await adapter.list_devices(pattern, **kw)


async def test_a_failing_db_is_unavailable_on_its_branch(root, monkeypatch) -> None:
    _patch_db(monkeypatch, fail=True)
    a = GeecsDirectoryAdapter("Undulator", root)
    with pytest.raises(DirectoryUnavailable, match="GEECS DB"):
        await a.list_devices("*")
    native = await a.list_devices("*", element_type="native")
    assert native == ["SimMotor1", "UC_VisaEBeam6"]
    health = await a.health()
    assert not health.healthy and "GEECS DB" in health.last_error
    await a.close()


async def test_health_with_the_db_off(root) -> None:
    a = GeecsDirectoryAdapter("Undulator", root, db=False)
    assert (await a.health()).healthy
    with pytest.raises(DirectoryUnavailable, match="db=off"):
        await a.list_devices("*", element_type="gateway")
    await a.close()


async def test_canopys_contract_suite(adapter) -> None:
    await contract.assert_identity_attributes(adapter)
    await contract.assert_health(adapter)
    await contract.assert_list_pvs_invariants(adapter, pattern="device:*")
    await contract.assert_list_devices_invariants(adapter, pattern="*")
    await contract.assert_list_elements_invariants(adapter, pattern="*")
    await contract.assert_close_idempotent(adapter)
