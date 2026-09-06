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


def test_readiness_oneshot_is_its_own_clean_row() -> None:
    """The geecs-qserver-ready oneshot (active/exited) is not a second RE Manager process."""
    m = _merged(
        "role=Queueserver RE Manager\tsvc=geecs-qserver.service\tmanaged=systemd\tstate=active/running\tpkg=geecs-bluesky\tpyproject=0.76.0\tinstalled=0.76.0",
        "role=Queueserver readiness\tsvc=geecs-qserver-ready.service\tmanaged=systemd\tstate=active/exited\tpkg=geecs-bluesky\tpyproject=0.76.0",
    )
    assert fleet_table.glyph(m["Queueserver RE Manager"]) == "✓"
    assert fleet_table.notes(m["Queueserver RE Manager"]) == []
    assert fleet_table.glyph(m["Queueserver readiness"]) == "✓"
    assert fleet_table.version(m["Queueserver readiness"]) == "geecs-bluesky 0.76.0"


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


def test_supervised_redis_is_a_clean_row_unsupervised_is_a_finding() -> None:
    """The Redis row's whole job: answering is not the same as supervised.

    Host finding 2026-09-06 — the queueserver ran for two weeks against a
    hand-built Redis in no unit, because ``launch_re_manager.sh`` starts its
    own daemonized ``redis-server`` whenever nothing answers on 6379.
    """
    m = _merged(
        "role=Redis\tsvc=redis-server.service\tmanaged=systemd\tstate=active/running\tversion=6.0.16\tinfo=loopback 6379, enabled"
    )
    rec = m["Redis"]
    assert fleet_table.glyph(rec) == "✓"
    assert fleet_table.notes(rec) == []
    assert fleet_table.version(rec) == "6.0.16"

    m = _merged(
        "role=Redis\tsvc=redis :6379\tmanaged=geecs-qserver.service\tstate=running pid 453172\tversion=8.10.1\tnote=answering on 6379 but NOT supervised by redis-server.service (owner: geecs-qserver.service) — the launcher fallback"
    )
    rec = m["Redis"]
    assert fleet_table.glyph(rec) == "!"
    assert len(fleet_table.notes(rec)) == 1
    assert "NOT supervised" in fleet_table.notes(rec)[0]


def test_absent_redis_is_down() -> None:
    m = _merged(
        "role=Redis\tsvc=redis :6379\tmanaged=none\tstate=down\tnote=nothing on 6379 — the queueserver keeps no queue/history/permissions"
    )
    rec = m["Redis"]
    assert fleet_table.glyph(rec) == "✗"


def test_redis_sorts_after_the_services_that_depend_on_it() -> None:
    """Display order: the state store reads below its consumers, not above."""
    order = fleet_table.ROLE_ORDER
    assert order.index("Redis") > order.index("Queueserver RE Manager")
    assert order.index("Redis") > order.index("Capture daemon")
