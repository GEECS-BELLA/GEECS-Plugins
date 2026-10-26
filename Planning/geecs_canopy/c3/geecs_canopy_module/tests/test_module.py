"""What the brief pins, against a fake Hare: no Canopy needed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from geecs_canopy_module.family import load_devices
from geecs_canopy_module.journal import Journal
from geecs_canopy_module.routes import HareRefused, build_app

CAM = "undulator:uc_visaebeam6:"


class FakeHare:
    """Values by PV (a list is returned one element per read, last one sticking)."""

    def __init__(
        self, values: dict[str, object] | None = None, refuse: HareRefused | None = None
    ) -> None:
        self.values = values or {}
        self.refuse = refuse
        self.calls: list[tuple[str, str]] = []

    async def get(self, pv: str) -> dict[str, object]:
        self.calls.append(("get", pv))
        value = self.values.get(pv, 1.0)
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if value is None:
            return {"pv": pv, "value": None, "connected": False, "severity": "INVALID"}
        return {
            "pv": pv,
            "value": value,
            "connected": True,
            "severity": "NO_ALARM",
            "tsNs": 0,
        }

    async def put(
        self, pv: str, value: object, *, wait: bool, timeout: float | None
    ) -> None:
        self.calls.append(("put", pv))
        assert wait is True
        if self.refuse:
            raise self.refuse


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(str(tmp_path / "journal.sqlite3"))


def client(hare: FakeHare, journal: Journal) -> TestClient:
    key = SimpleNamespace(name="test-key")
    app = build_app(
        hare,
        journal,
        load_devices(),
        read_access=lambda: key,
        write_access=lambda: key,
        window_s=0,
    )
    return TestClient(app)


def roles(body: dict) -> dict[tuple[str, str | None], dict]:
    return {(r["role"], r["axis"]): r for r in body["roles"]}


def test_motor_route_builds_every_role_per_axis(journal):
    body = client(FakeHare(), journal).get("/v1/devices/U_SimMotor1").json()

    by_role = roles(body)
    assert body["family"] == "motor_record"
    assert {axis for _, axis in by_role} == {"m1", "m2"}
    assert len(by_role) == 2 * 29
    assert by_role[("rbv", "m1")]["pv"] == "undulator:simmotor1:m1.RBV"
    assert by_role[("val", "m2")]["value"] == 1.0
    assert by_role[("val", "m2")]["verdict"] == "connected"


@pytest.mark.parametrize(
    ("counter", "flag", "mean", "expected"),
    [
        ([10.0, 12.0], 1, 5.0, "fresh"),
        ([10.0, 10.0], 1, 5.0, "stale"),
        ([10.0, 12.0], 0, 5.0, "off"),
        ([10.0, 12.0], 1, None, "disconnected"),
        (None, 1, 5.0, "stale"),
    ],
    ids=[
        "counter-advanced",
        "counter-still",
        "flag-off",
        "pv-disconnected",
        "counter-disconnected",
    ],
)
def test_freshness_by_counter_and_flag(journal, counter, flag, mean, expected):
    hare = FakeHare(
        {
            f"{CAM}STAT:ArrayCounter_RBV": counter,
            f"{CAM}STAT:ComputeStatistics": flag,
            f"{CAM}STAT:MeanValue_RBV": mean,
        }
    )
    body = client(hare, journal).get("/v1/devices/UC_VisaEBeam6").json()

    assert roles(body)[("mean", None)]["verdict"] == expected


def test_a_changing_value_is_not_freshness(journal):
    hare = FakeHare(
        {
            f"{CAM}STAT:ArrayCounter_RBV": [10.0, 10.0],
            f"{CAM}STAT:MeanValue_RBV": [5.0, 6.0],
        }
    )
    body = client(hare, journal).get("/v1/devices/UC_VisaEBeam6").json()

    assert roles(body)[("mean", None)]["verdict"] == "stale"


def test_a_role_outside_the_allow_list_is_denied_before_hare(journal):
    hare = FakeHare()
    answer = client(hare, journal).post(
        "/v1/devices/U_FpsTest1/put",
        json={"role": "output_sp", "value": 1, "operator": "op"},
    )

    assert answer.status_code == 403
    assert answer.json()["outcome"] == "denied"
    assert answer.json()["code"] == "denied"
    assert hare.calls == []
    assert [r["outcome"] for r in journal.rows()] == ["denied"]


def test_an_accepted_put_is_journaled(journal):
    hare = FakeHare()
    answer = client(hare, journal).post(
        "/v1/devices/U_SimMotor1/put",
        json={"role": "twf", "axis": "m1", "value": 1, "operator": "op"},
    )

    assert answer.status_code == 200
    assert answer.json()["outcome"] == "accepted"
    assert hare.calls == [("put", "undulator:simmotor1:m1.TWF")]
    (row,) = journal.rows()
    assert row["id"] == answer.json()["journalId"]
    assert (
        row["operator"],
        row["key_name"],
        row["device"],
        row["role"],
        row["axis"],
    ) == (
        "op",
        "test-key",
        "U_SimMotor1",
        "twf",
        "m1",
    )
    assert (row["outcome"], row["canopy_code"]) == ("accepted", None)


def test_hare_pv_write_failed_passes_through_as_refused(journal):
    refusal = HareRefused(
        502,
        "pv_write_failed",
        "CA write to undulator:fpstest1:current_ramp:SP: Write access denied",
    )
    hare = FakeHare(refuse=refusal)
    answer = client(hare, journal).post(
        "/v1/devices/U_FpsTest1/put",
        json={"role": "current_sp", "value": 0.5, "operator": "op"},
    )

    assert answer.status_code == 502
    assert answer.json()["outcome"] == "refused"
    assert (answer.json()["code"], answer.json()["detail"]) == (
        "pv_write_failed",
        refusal.detail,
    )
    (row,) = journal.rows()
    assert (row["outcome"], row["canopy_code"], row["detail"]) == (
        "refused",
        "pv_write_failed",
        refusal.detail,
    )
