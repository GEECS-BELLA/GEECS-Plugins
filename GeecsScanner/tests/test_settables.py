"""The movable panel's list: numeric settables only, aliased ones first (PR 5b)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.readback import parse_device_variable
from geecs_scanner.service.settables import build_settables


def _row(name, *, settable=True, vartype="numeric", choices=None, alias="", units=""):
    return {
        "name": name,
        "settable": settable,
        "variabletype": vartype,
        "choices": choices,
        "alias": alias,
        "units": units,
        "min": None,
        "max": None,
    }


def test_build_wraps_the_core_list_as_the_api_model() -> None:
    rows = {
        "U_Jet": [_row("Position.Axis 3", alias="Jet_Z (mm)", units="mm")],
        "U_S1H": [_row("Current")],
    }
    out = build_settables(rows)
    assert [(o.alias, o.name) for o in out] == [
        ("Jet_Z (mm)", "U_Jet:Position.Axis 3"),
        ("", "U_S1H:Current"),
    ]
    assert (
        out[0].model_dump()["units"] == "mm"
    )  # the filter and order themselves: GEECS-Core tests


def test_parse_device_variable_refuses_anything_but_device_colon_variable() -> None:
    assert parse_device_variable(" U_S1H : Current ") == ("U_S1H", "Current")
    for bad in ("U_S1H", ":Current", "U_S1H:", "S1H current"):
        with pytest.raises(ScannerError):
            parse_device_variable(bad)


def test_settables_route_lists_the_demo_devices_alias_first(client: TestClient) -> None:
    r = client.get("/api/settables")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    out = r.json()
    assert out["source"] == "demo" and out["detail"] == ""
    names = [s["name"] for s in out["items"]]
    aliases = [s["alias"] for s in out["items"]]
    assert names[:2] == ["U_Hexapod:xpos", "U_S1H:current"] and all(aliases[:2])
    assert all(a == "" for a in aliases[2:]) and names[2:] == sorted(
        names[2:], key=str.lower
    )


def test_readback_follows_a_move_and_carries_units(client: TestClient, manager) -> None:
    before = client.get(
        "/api/readback", params={"variable": "U_S1H:current", "units": "A"}
    ).json()
    assert before["ok"] is True and before["value"] == 0.0 and before["units"] == "A"
    assert before["age_s"] == 0.0 and before["variable"] == "U_S1H:current"
    r = client.post("/api/move", json={"variable": "U_S1H:current", "value": 1.5})
    assert r.status_code == 200, r.text
    manager.step()  # the fake worker finishes the mv item
    after = client.get("/api/readback", params={"variable": "U_S1H:current"}).json()
    assert after["value"] == 1.5


def test_readback_refuses_a_non_canonical_name(client: TestClient) -> None:
    r = client.get("/api/readback", params={"variable": "S1H current"})
    assert r.status_code == 400 and "Device:Variable" in r.json()["error"]["message"]
