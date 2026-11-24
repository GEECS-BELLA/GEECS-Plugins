"""Pin ``scripts/qserver_probe.py``'s verdict — a running manager is not a ready one (#793).

The probe's 0MQ calls need ``bluesky-queueserver``; the reduction from the
manager's replies to ready / notready / unknown / down is pure — it is
``geecs_bluesky.qs_client.readiness_verdict``'s, mapped to exit codes — and
pinned here from the root env without it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "qserver_probe.py"

spec = importlib.util.spec_from_file_location("qserver_probe", PROBE)
assert spec and spec.loader
qserver_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qserver_probe)
summarize = qserver_probe.summarize

FIVE_PLANS = {
    "success": True,
    "plans_allowed": {
        f"geecs_{n}_plan": {}
        for n in ("noscan", "optimize", "apply_setup", "scan", "scan_request")
    },
}


def _status(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "manager_state": "idle",
        "worker_environment_exists": True,
        "worker_environment_state": "idle",
        "re_state": "idle",
        "items_in_queue": 0,
        "running_item_uid": None,
    }
    base.update(over)
    return base


def test_open_environment_with_plans_is_ready() -> None:
    code, line = summarize(_status(), FIVE_PLANS, "primary")
    assert code == 0
    assert line.startswith("ready ")
    assert (
        "environment idle" in line
        and "5 plans allowed (primary)" in line
        and "queue 0" in line
    )


def test_closed_environment_is_not_ready_even_though_the_manager_answers() -> None:
    """The 2026-09-04 shape: unit active, port listening, plan list empty."""
    status = _status(
        worker_environment_exists=False,
        worker_environment_state="closed",
        re_state=None,
    )
    code, line = summarize(status, {"success": True, "plans_allowed": {}}, "primary")
    assert code == 5
    assert line.startswith("notready ")
    assert (
        "environment is closed" in line
        and "allowed plans" in line
        and "qserver environment open" in line
    )


def test_an_opening_environment_is_not_ready_and_says_not_to_restart() -> None:
    """The readiness unit is importing the profile: not ready yet, never 'closed'."""
    status = _status(
        manager_state="creating_environment",
        worker_environment_exists=False,
        worker_environment_state="initializing",
        re_state=None,
    )
    code, line = summarize(status, None, "primary")
    assert code == 5
    assert line.startswith("notready ")
    assert "being opened" in line and "do not restart" in line
    assert "is closed" not in line


def test_a_probe_python_without_geecs_bluesky_is_unknown_never_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No hand-copied rule to fall back on: without the library, readiness is unknown."""
    import sys

    monkeypatch.setitem(sys.modules, "geecs_bluesky.qs_client.client", None)
    code, line = summarize(_status(), FIVE_PLANS, "primary")
    assert code == 6
    assert line.startswith("unknown geecs_bluesky does not import")


def test_the_verdict_is_readiness_verdicts_own_sentence() -> None:
    """One definition of ready: the probe carries the library's detail verbatim."""
    from geecs_bluesky.qs_client.client import (
        queue_status_from_manager,
        readiness_verdict,
    )

    status = _status(worker_environment_exists=False, worker_environment_state="closed")
    _code, line = summarize(status, None, "primary")
    expected = readiness_verdict(queue_status_from_manager(status), None).detail
    assert expected in line


def test_open_environment_with_zero_plans_is_not_ready() -> None:
    code, line = summarize(_status(), {"success": True, "plans_allowed": {}}, "primary")
    assert code == 5
    assert "0 plans" in line and "primary" in line


def test_no_answer_is_down() -> None:
    code, line = summarize(None, None, "primary")
    assert code == 1
    assert line.startswith("down ")


def test_unanswered_plans_allowed_is_unknown_never_ready() -> None:
    """The second 0MQ request timed out: status alone cannot assert readiness."""
    code, line = summarize(_status(), None, "primary")
    assert code == 6
    assert line.startswith("unknown ")
    assert "plans_allowed unanswered" in line
    assert not line.startswith("ready")


def test_refused_plans_allowed_is_unknown_never_ready() -> None:
    reply = {"success": False, "msg": "user group 'primary' is not defined"}
    code, line = summarize(_status(), reply, "primary")
    assert code == 6
    assert line.startswith("unknown ")
    assert "plans_allowed unanswered" in line and "not defined" in line


def test_closed_environment_wins_over_an_unanswered_plan_list() -> None:
    status = _status(worker_environment_exists=False, worker_environment_state="closed")
    code, line = summarize(status, None, "primary")
    assert code == 5 and line.startswith("notready ")


def test_running_item_and_queue_depth_are_reported() -> None:
    status = _status(
        manager_state="executing_queue",
        re_state="running",
        items_in_queue=2,
        running_item_uid="abc",
    )
    code, line = summarize(status, FIVE_PLANS, "primary")
    assert code == 0
    assert "queue 2" in line and "item running" in line and "RE running" in line


def test_fleet_status_wires_the_probe() -> None:
    text = (REPO_ROOT / "scripts" / "fleet_status.sh").read_text()
    assert "qserver_probe.py" in text
    assert "notready" in text.lower() or "NOT READY" in text


@pytest.mark.parametrize(
    ("ini", "expected"),
    [
        ("[qserver]\nhost = worker.example\n", "tcp://worker.example:60615"),
        (
            "[qserver]\nhost = worker.example\ncontrol_addr = tcp://10.0.0.9:61000\n",
            "tcp://10.0.0.9:61000",
        ),
        ("[qserver]\ncontrol_addr = tcp://10.0.0.9:61000\n", "tcp://10.0.0.9:61000"),
        ("[tiled]\nuri = http://x:8000\n", None),
        ("[qserver]\n", None),
    ],
)
def test_control_addr_follows_the_qs_client_precedence(
    tmp_path: Path, ini: str, expected: str | None
) -> None:
    """``control_addr`` verbatim, else ``tcp://<host>:60615`` — never a hand-built address over an override."""
    path = tmp_path / "config.ini"
    path.write_text(ini)
    assert qserver_probe.control_addr_from_ini(path) == expected


def test_missing_config_yields_no_address(tmp_path: Path) -> None:
    assert qserver_probe.control_addr_from_ini(tmp_path / "absent.ini") is None


def test_explicit_addr_wins() -> None:
    assert qserver_probe.resolve_control_addr("tcp://h:1") == "tcp://h:1"


def test_fleet_status_honors_the_control_addr_override_and_the_unknown_verdict() -> (
    None
):
    text = (REPO_ROOT / "scripts" / "fleet_status.sh").read_text()
    assert "ini_get qserver control_addr" in text
    assert "readiness UNKNOWN" in text
    # exit 6 (unknown) is its own case — never the ready branch, never a bare ✓.
    assert "6)" in text
