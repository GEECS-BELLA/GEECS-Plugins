"""Pin ``scripts/qserver_probe.py``'s verdict — a running manager is not a ready one (#793).

The probe's 0MQ calls need ``bluesky-queueserver``; the reduction from the
manager's replies to ready / notready / down is pure and pinned here from
the root env without it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
        for n in ("noscan", "optimize", "run_action", "scan", "scan_request")
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
        "CLOSED" in line
        and "allowed plans" in line
        and "qserver environment open" in line
    )


def test_open_environment_with_zero_plans_is_not_ready() -> None:
    code, line = summarize(_status(), {"success": True, "plans_allowed": {}}, "primary")
    assert code == 5
    assert "0 plans" in line and "primary" in line


def test_no_answer_is_down() -> None:
    code, line = summarize(None, None, "primary")
    assert code == 1
    assert line.startswith("down ")


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
