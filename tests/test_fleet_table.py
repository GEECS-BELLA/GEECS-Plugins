"""Pin ``scripts/fleet_table.py``'s glyph rule: findings mark ``!``, facts do not."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fleet_table", REPO_ROOT / "scripts" / "fleet_table.py"
)
assert spec and spec.loader
fleet_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet_table)


def _merged(*lines: str) -> dict[str, dict[str, str]]:
    return fleet_table.parse([line + "\n" for line in lines])


def test_info_fields_show_but_do_not_mark_attention() -> None:
    m = _merged("role=CA gateway\tstate=ok\tversion=0.20.2\tinfo=99 devices connected")
    rec = m["CA gateway"]
    assert fleet_table.glyph(rec) == "✓"
    assert fleet_table.notes(rec) == []
    assert fleet_table.infos(rec) == ["99 devices connected"]


def test_note_fields_mark_attention() -> None:
    m = _merged(
        "role=PVA image gateways\tstate=ok\tversion=0.5.0 ×8\tinfo=8 of 9 deployed up\tinfo=2 not deployed\tnote=1 unreachable: 192.168.8.201"
    )
    rec = m["PVA image gateways"]
    assert fleet_table.glyph(rec) == "!"
    assert fleet_table.notes(rec) == ["1 unreachable: 192.168.8.201"]
    assert fleet_table.infos(rec) == ["8 of 9 deployed up", "2 not deployed"]


def test_baked_venv_and_behind_master_are_facts_ahead_is_a_finding() -> None:
    base = "role=GEECS-MCP\tsvc=geecs-mcp.service\tmanaged=systemd\tstate=active/running\tsha=13a2a42c\tbaked=~/geecs-mcp-venv\tpkg=geecs-mcp\tpyproject=0.8.6\tinstalled=0.8.6"
    m = _merged(
        base, "role=GEECS-MCP\tfor_sha=13a2a42c\tmaster_rel=9 behind origin/master"
    )
    rec = m["GEECS-MCP"]
    assert fleet_table.glyph(rec) == "✓"
    assert fleet_table.infos(rec) == ["baked venv", "9 behind master"]

    m = _merged(
        base,
        "role=GEECS-MCP\tfor_sha=13a2a42c\tmaster_rel=2 ahead, 0 behind origin/master",
    )
    assert fleet_table.glyph(m["GEECS-MCP"]) == "!"
    assert "2 ahead, 0 behind master" in fleet_table.notes(m["GEECS-MCP"])


def test_mcp_not_listening_is_down_not_absent() -> None:
    m = _merged("role=GEECS-MCP\tstate=down\tnote=not listening")
    assert fleet_table.glyph(m["GEECS-MCP"]) == "✗"


def test_real_findings_still_mark_attention() -> None:
    m = _merged(
        "role=Data Portal\tsvc=geecs-data-portal.service\tmanaged=systemd\tstate=active/running\tpkg=geecs-data-portal\tpyproject=0.20.2\tinstalled=0.20.1"
    )
    rec = m["Data Portal"]
    assert fleet_table.glyph(rec) == "!"
    assert fleet_table.notes(rec) == ["venv 0.20.1 ≠ pyproject 0.20.2"]
