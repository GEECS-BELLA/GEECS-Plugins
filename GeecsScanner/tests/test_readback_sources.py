"""The production sources behind /api/settables and /api/readback, with their remotes faked."""

from __future__ import annotations

import sys
import time
import types

import pytest

from geecs_scanner.service.readback import CaReadback
from geecs_scanner.service.settables import DbSettables


class _FakeDb:
    """A GeecsDb double: fails N times, then answers."""

    def __init__(self, failures: int, rows: dict) -> None:
        self.failures, self.rows, self.calls = failures, rows, 0

    def get_experiment_device_variables(self, experiment: str) -> dict:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("2003: can't connect to MySQL server")
        return self.rows


_ROWS = {
    "U_S1H": [
        {
            "name": "Current",
            "settable": True,
            "variabletype": "numeric",
            "choices": None,
            "alias": "S1H",
            "units": "A",
            "min": -5,
            "max": 5,
        }
    ]
}


def test_db_settables_reports_a_failed_read_and_does_not_cache_it() -> None:
    db = _FakeDb(failures=1, rows=_ROWS)
    src = DbSettables("Exp", db=db)
    first = src.settables()
    assert first.items == [] and first.source == "db" and "MySQL" in first.detail
    second = src.settables()  # retried, now cached
    assert [s.name for s in second.items] == ["U_S1H:Current"] and second.detail == ""
    src.settables()
    assert db.calls == 2  # the success is served from the cache


class _CaValue(float):
    """What aioca returns for FORMAT_TIME: a float with .ok and .timestamp."""

    ok = True
    timestamp = 0.0


class _CaNothing:
    ok = False


def _fake_aioca(monkeypatch, answer) -> list:
    calls: list = []

    async def caget(pv, **kw):
        calls.append((pv, kw))
        return answer

    mod = types.ModuleType("aioca")
    mod.caget, mod.FORMAT_TIME = caget, object()
    monkeypatch.setitem(sys.modules, "aioca", mod)
    return calls


@pytest.mark.anyio
async def test_ca_readback_reads_the_gateway_readback_pv_with_its_stamp(
    monkeypatch,
) -> None:
    v = _CaValue(0.99989)
    v.timestamp = time.time() - 2.0
    calls = _fake_aioca(monkeypatch, v)
    out = await CaReadback("Undulator", timeout=0.7).read("U_S1H", "Current", units="A")
    assert calls[0][0] == "undulator:u_s1h:current"  # the readback, never :SP
    assert calls[0][1]["timeout"] == 0.7 and calls[0][1]["throw"] is False
    assert out.ok and out.value == pytest.approx(0.99989) and out.units == "A"
    assert out.pv == "undulator:u_s1h:current" and 1.5 < out.age_s < 10


@pytest.mark.anyio
async def test_ca_readback_reports_a_silent_gateway(monkeypatch) -> None:
    _fake_aioca(monkeypatch, _CaNothing())
    out = await CaReadback("Undulator").read("U_S1H", "Current")
    assert out.ok is False and out.value is None and "no answer" in out.detail


@pytest.mark.anyio
async def test_ca_readback_refuses_a_non_numeric_answer(monkeypatch) -> None:
    class _Str(str):
        ok = True
        timestamp = 1.0

    _fake_aioca(monkeypatch, _Str("OFF"))
    out = await CaReadback("Undulator").read("U_PLC", "shutter1")
    assert out.ok is False and "not a number" in out.detail


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
