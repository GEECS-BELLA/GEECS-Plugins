"""The adapter's answers: native documents, the DB through a connection double, Canopy's contract."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from geecs_core.db import geecs_db

from geecs_canopy.directory import GeecsDirectoryAdapter, canopy_base
from geecs_canopy.gateway import DB_URL_ENV, GatewaySource
from geecs_canopy.native import NativeSource

base = canopy_base()

REGISTRY = """\
schema_version: 1
devices:
  UC_VisaEBeam6: {kind: aravis_camera, trigger_source: line0}
  SimMotor1: {kind: epics_motor, axes: [m1, m2]}
"""
PANELS = """\
schema_version: 1
devices:
  eztest1: {family: easydriver, roles: {current: I_RBV}, omit: [interlock]}
  BenchSupply: {family: easydriver, prefix: "bench:ps1:"}
"""
URL = "mysql://reader:p%40ss@db.example:3307/geecs"


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
        def cursor(self):
            return self

        def execute(self, query: str, params: tuple) -> None:
            self.rows = next(rows for key, rows in DB_ROWS.items() if key in query)

        def fetchall(self):
            return self.rows

        def close(self) -> None:
            pass

    def connect(_connector):
        if fail:
            raise ConnectionError("no route to the DB")
        return _Connection()

    monkeypatch.setattr(geecs_db, "_connect_mysql", connect)
    monkeypatch.setattr(geecs_db, "_credentials", None)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "Undulator").mkdir()
    (tmp_path / "Undulator" / "native_devices.yaml").write_text(REGISTRY)
    (tmp_path / "Undulator" / "panel_devices.yaml").write_text(PANELS)
    return tmp_path


def _adapter(root: Path, *, db: bool = True, ttl: float = 30.0):
    gateway = GatewaySource("Undulator", URL) if db else None
    return GeecsDirectoryAdapter(
        NativeSource("Undulator", root), gateway, health_ttl_s=ttl
    )


@pytest.fixture
async def adapter(root: Path, monkeypatch):
    _patch_db(monkeypatch)
    a = _adapter(root)
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
    exposure = await adapter.list_pvs("element:UC_VisaEBeam6:exposure")
    assert exposure == [cam + "_RBV", cam]
    assert await adapter.list_devices("*", tag="ad_camera_chain") == ["UC_VisaEBeam6"]


async def test_panel_devices_join_by_name_with_their_own_records(adapter) -> None:
    tagged = await adapter.list_devices("*", tag="easydriver", sort_by="name")
    assert tagged == ["BenchSupply", "eztest1"]
    current = await adapter.list_pvs("element:eztest1:current")
    assert current == ["undulator:eztest1:I_RBV"]  # the device's own record
    assert await adapter.list_elements("eztest1:interlock") == []  # omitted here
    assert await adapter.list_pvs("element:BenchSupply:fault") == ["bench:ps1:fault"]


async def test_a_db_device_lists_the_gateways_served_pvs(adapter) -> None:
    jet = "undulator:u_jet:"
    assert await adapter.list_pvs("device:U_Jet") == [
        *(jet + v for v in ("position", "position:SP", "temperature", "connected"))
    ]
    assert await adapter.list_devices("%", element_type="gateway") == ["U_Jet"]
    sp = jet + "position:SP"  # a literal PV answers itself: Raccoon's `exists`
    assert await adapter.list_pvs(sp) == [sp]
    creds = geecs_db._find_credentials()  # from the URL, never a config.ini
    assert creds == dict(
        host="db.example", port=3307, database="geecs", user="reader", password="p@ss"
    )


BAD = [("", {}), ("A|B", {}), ("*", {"element_type": "x"}), ("*", {"sort_by": "z"})]


@pytest.mark.parametrize("pattern, kw", BAD)
async def test_a_bad_query_is_invalid(adapter, pattern, kw) -> None:
    with pytest.raises(base.InvalidQuery):
        await adapter.list_devices(pattern, **kw)


DISAGREE = [
    ("devices: {SimMotor1: {family: easydriver}}", "one family per device"),
    ('devices: {SimMotor1: {family: motor_record, prefix: "x:"}}', "derived"),
    ("devices: {eztest1: {family: nope}}", "does not validate"),
]


@pytest.mark.parametrize("panels, match", DISAGREE)
async def test_documents_that_disagree_are_misconfigured(root, panels, match) -> None:
    (root / "Undulator" / "panel_devices.yaml").write_text(panels)
    a = _adapter(root, db=False)
    with pytest.raises(base.AdapterMisconfigured, match=match):
        await a.list_devices("*")
    await a.close()


async def test_a_failing_db_is_unavailable_on_its_branch_only(root, monkeypatch):
    _patch_db(monkeypatch, fail=True)
    a = _adapter(root)
    with pytest.raises(base.DirectoryUnavailable, match="GEECS DB"):
        await a.list_devices("*")
    native = await a.list_devices("*", element_type="native", sort_by="name")
    assert native == ["BenchSupply", "SimMotor1", "UC_VisaEBeam6", "eztest1"]
    health = await a.health()
    assert not health.healthy and "GEECS DB" in health.last_error
    await a.close()


async def test_health_reads_the_sources_once_per_ttl(root) -> None:
    a = _adapter(root, db=False, ttl=60.0)
    first = await a.health()
    assert first.healthy
    (root / "Undulator" / "native_devices.yaml").write_text("devices: {X: {}}")
    assert await a.health() is first  # cached: no source read within the TTL
    with pytest.raises(base.DirectoryUnavailable, match="db=off"):
        await a.list_devices("*", element_type="gateway")
    await a.close()


def test_from_config_reads_lowercased_extras_and_refuses_gaps(root, monkeypatch):
    configs = root / "configs"
    (configs / "scanner_configs").mkdir(parents=True)
    (root / "Undulator").rename(configs / "scanner_configs" / "experiments")
    extras = {"experiment": "Undulator", "configs_root": str(configs), "db": "off"}
    a = GeecsDirectoryAdapter.from_config(SimpleNamespace(extras=extras))
    assert a._gateway is None and a._native.root.name == "experiments"
    with pytest.raises(base.AdapterMisconfigured, match="experiment"):
        GeecsDirectoryAdapter.from_config(SimpleNamespace(extras={"db": "off"}))
    monkeypatch.setenv(DB_URL_ENV, "postgres://nope")
    with pytest.raises(base.AdapterMisconfigured, match=DB_URL_ENV):
        GeecsDirectoryAdapter.from_config(
            SimpleNamespace(extras={**extras, "db": "on"})
        )


def test_outside_canopy_the_error_says_so(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "canopy.raccoon.adapters", None)
    with pytest.raises(ImportError, match="runs inside Canopy"):
        canopy_base()


SOURCE = os.environ.get("CANOPY_SOURCE")


@pytest.mark.skipif(
    not (SOURCE and hasattr(base, "DirectoryAdapter")),
    reason="needs Canopy installed and a clone at the pin in CANOPY_SOURCE (DEPLOYMENT.md)",
)
async def test_canopys_contract_suite(adapter) -> None:
    path = Path(SOURCE or "") / "tests/canopy/raccoon/adapters/contract.py"
    spec = importlib.util.spec_from_file_location("canopy_contract", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    await contract.assert_identity_attributes(adapter)
    await contract.assert_health(adapter)
    await contract.assert_list_pvs_invariants(adapter, pattern="device:*")
    await contract.assert_list_devices_invariants(adapter, pattern="*")
    await contract.assert_list_elements_invariants(adapter, pattern="*")
    await contract.assert_close_idempotent(adapter)
