"""A manager that runs scans in memory — for tests, and for ``geecs-scanner --demo``.

:class:`DemoQueueClient` implements the :class:`~geecs_bluesky.qs_client.QueueClient`
protocol over a list.  A submitted item starts when the queue is idle,
advances one shot per :meth:`step` (a background thread calls it every
``period`` seconds; tests pass ``period=0`` and step by hand), pauses at a
step boundary, stops gracefully, finishes into the history with the
manager's ``result`` shape — and emits the same bluesky documents the real
worker would (start / descriptor / event / stop) into a
:class:`~geecs_scanner.service.streams.ProgressCache`, so the SSE stream
and the page behave exactly as they will against the worker.

:class:`DemoResolver` holds three real :class:`geecs_schemas.Preset`
documents, a scan-variable catalog with a pseudo entry (listed, refused,
as on the real branch), trigger profiles and optimizer-config names.

:func:`demo_preflight` validates by the real :func:`expand_preset` and
returns one fixed operator question, so the acknowledgement loop can be
exercised without a CA gateway.

Nothing here is a default in code for a deployment: the demo is chosen by
a flag, and every name in it is sample content.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from geecs_scanner.service.streams import ProgressCache

_DEMO_DEVICES = [
    "UC_ALineEBeam3",
    "UC_TC_Phosphor",
    "UC_UndulatorRad2",
    "U_ICT",
    "U_HP_Daq",
    "U_S1H",
    "U_Hexapod",
]


def demo_presets() -> list[Any]:
    """Three presets in the real schema."""
    from geecs_schemas import Preset

    return [
        Preset.model_validate(
            {
                "name": "jet_pressure_sweep",
                "description": "Jet pressure 2–5 psi, ALine3 + phosphor, 10 shots/step",
                "trigger_profile": "standard_1hz",
                "devices": [
                    {
                        "device": "UC_ALineEBeam3",
                        "save_images": True,
                        "essential": True,
                    },
                    {
                        "device": "UC_TC_Phosphor",
                        "save_images": True,
                        "essential": False,
                    },
                    {"device": "U_ICT", "save_images": False, "essential": True},
                ],
                "plan": {
                    "name": "scan",
                    "args": ["Jet pressure", 2.0, 5.0, 7],
                    "kwargs": {"shots_per_step": 10, "acquisition": "strict"},
                },
            }
        ),
        Preset.model_validate(
            {
                "name": "eb_align_1hz",
                "description": "Alignment count, cameras only",
                "trigger_profile": "standard_1hz",
                "devices": [
                    {
                        "device": "UC_ALineEBeam3",
                        "save_images": True,
                        "essential": True,
                    },
                    {
                        "device": "UC_UndulatorRad2",
                        "save_images": True,
                        "essential": True,
                    },
                ],
                "plan": {
                    "name": "count",
                    "args": [],
                    "kwargs": {"num": 20, "acquisition": "strict"},
                },
            }
        ),
        Preset.model_validate(
            {
                "name": "background_dark",
                "description": "Darks with the shutter closed",
                "trigger_profile": "no_gas",
                "background": True,
                "devices": [
                    {
                        "device": "UC_ALineEBeam3",
                        "save_images": True,
                        "essential": True,
                    },
                ],
                "plan": {
                    "name": "count",
                    "args": [],
                    "kwargs": {"num": 50, "acquisition": "strict"},
                },
            }
        ),
    ]


class DemoResolver:
    """An in-memory stand-in for ``ConfigsRepoResolver``."""

    def __init__(self) -> None:
        self._presets = {p.name: p for p in demo_presets()}

    def list_presets(self) -> list[str]:
        """Preset names."""
        return sorted(self._presets)

    def resolve_preset(self, name: str) -> Any:
        """One preset, or ``KeyError``."""
        return self._presets[name]

    def list_trigger_profiles(self) -> list[str]:
        """Trigger-profile names."""
        return ["standard_1hz", "no_gas", "hexapod_slow"]

    def list_optimizer_configs(self) -> list[str]:
        """Optimizer-config names (listed; nothing submits one yet)."""
        return ["xopt_beam_charge"]

    def action_plan_registry(self) -> dict[str, Any]:
        """Action plans by name (none in the demo)."""
        return {}

    def scan_variable_catalog(self) -> Any:
        """A catalog with three plain entries and one pseudo entry."""
        from geecs_schemas import ScanVariables

        return ScanVariables.model_validate(
            {
                "variables": {
                    "Jet pressure": {
                        "target": "U_HP_Daq:Jet pressure",
                        "kind": "setpoint",
                    },
                    "S1H current": {"target": "U_S1H:current", "kind": "setpoint"},
                    "Hexapod X": {"target": "U_Hexapod:xpos", "kind": "motor"},
                    "Gas jet 2-axis": {
                        "kind": "pseudo",
                        "mode": "absolute",
                        "targets": [
                            {"target": "U_Hexapod:xpos", "forward": "x"},
                            {"target": "U_Hexapod:ypos", "forward": "0.5 * x"},
                        ],
                    },
                }
            }
        )


def demo_preflight(
    preset: Any, experiment: str, *, client: Any = None, catalog: Any = None
) -> Any:
    """Validate by the real expansion; ask one fixed question."""
    from geecs_bluesky.qs_client import (
        PreflightQuestion,
        PreflightReport,
        expand_preset,
    )

    report = PreflightReport()
    try:
        expand_preset(preset, catalog=catalog)
        report.outcomes.append(("validate", "passed", ""))
    except Exception as exc:  # noqa: BLE001 — the refusal text is the message
        report.refusal = str(exc)
        return report
    report.outcomes.append(("worker_ready", "passed", "demo manager"))
    report.questions.append(
        PreflightQuestion(
            check="gateway_liveness",
            title="UC_TC_Phosphor reports Disconnected on the gateway",
            message="One CA read, 0.8 s ago. Its frames would be missing rows in the s-file.",
        )
    )
    return report


class DemoQueueClient:
    """A fake RE Manager: a queue, a running item, a history, documents out.

    Parameters
    ----------
    streams : ProgressCache, optional
        Where the documents and console lines go.
    period : float
        Seconds between shots when self-driving; ``0`` means step by hand.
    first_scan : int
        The scan number the first run claims.
    user : str
        What the fake manager records as the submitting user — the real
        client stamps the identity ``make_queue_client`` was built with.
    """

    info_addr: Optional[str] = "demo"
    doc_addr: Optional[str] = "demo"

    def __init__(
        self,
        streams: Optional[ProgressCache] = None,
        *,
        period: float = 0.0,
        first_scan: int = 46,
        clock: Callable[[], float] = time.time,
        user: str = "geecs-scanner demo",
    ) -> None:
        self.streams = streams or ProgressCache()
        self.streams.mark_available(True, "demo")
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: list[dict] = []
        self._history: list[dict] = []
        self._running: Optional[dict] = None
        self._re_state = "idle"
        self._pause_requested = False
        self._shots = 0
        self._total = 0
        self._per_step = 1
        self._scan_number = first_scan
        self._run_uid = ""
        self._desc_uid = ""
        self._user = user
        if period > 0:
            threading.Thread(
                target=self._drive, args=(period,), name="demo-manager", daemon=True
            ).start()

    # ---------------------------------------------------------- protocol

    def status(self) -> Any:
        """A connected snapshot of the fake manager."""
        from geecs_bluesky.qs_client import QueueStatus

        with self._lock:
            return QueueStatus(
                connected=True,
                re_state=self._re_state,
                manager_state="idle" if self._running is None else "executing_queue",
                worker_exists=True,
                items_in_queue=len(self._queue),
                running_item_uid=self._running["item_uid"] if self._running else None,
                worker_environment_state="idle"
                if self._running is None
                else "executing_plan",
            )

    def readiness(self, expected_plans: str | Sequence[str] | None = None) -> Any:
        """Always ready."""
        from geecs_bluesky.qs_client import readiness_verdict

        return readiness_verdict(
            self.status(), self.allowed_plan_names(), expected_plans
        )

    def allowed_plan_names(self) -> list[str]:
        """The real plan list."""
        from geecs_bluesky.plan_names import GEECS_PLAN_NAMES

        return list(GEECS_PLAN_NAMES)

    def allowed_device_names(self) -> list[str]:
        """The demo devices and their common children."""
        names: list[str] = []
        for d in _DEMO_DEVICES:
            names += [
                d,
                f"{d}.scalars",
                f"{d}.current",
                f"{d}.xpos",
                f"{d}.ypos",
                f"{d}.jet_pressure",
            ]
        return names

    def submit_plan(
        self,
        name: str,
        *,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        clear_pending: bool = False,
    ) -> Any:
        """Queue an item and start the queue if idle."""
        from geecs_bluesky.qs_client import SubmitResult

        if name not in self.allowed_plan_names():
            return SubmitResult(ok=False, message=f"plan {name!r} is not allowed")
        item = {
            "name": name,
            "args": list(args),
            "kwargs": dict(kwargs or {}),
            "item_type": "plan",
            "user": self._user,
            "item_uid": uuid.uuid4().hex[:12],
        }
        with self._lock:
            # The real client's guard (client._submit_item): an item already
            # WAITING refuses the add unless the caller clears it first — the
            # failed-item-at-front trap. The running item does not count.
            if self._queue and not clear_pending:
                return SubmitResult(
                    ok=False,
                    message=(
                        f"{len(self._queue)} item(s) already queued (a failed item "
                        "returns to the queue front) — clear before resubmitting"
                    ),
                    pending_items=[dict(i) for i in self._queue],
                )
            if clear_pending:
                self._queue.clear()
            self._queue.append(item)
            self.streams.push_console_line(
                f"queue: added {name} ({item['item_uid']}) by {self._user}"
            )
            if self._running is None:
                self._start_next()
        return SubmitResult(ok=True, message="queued", item_uid=item["item_uid"])

    def submit_preset(
        self,
        preset: Any,
        *,
        catalog: Optional[Mapping[str, Any]] = None,
        md: Optional[Mapping[str, Any]] = None,
        clear_pending: bool = False,
    ) -> Any:
        """Expand with the real expansion, then queue."""
        from geecs_bluesky.qs_client import expand_preset

        item = expand_preset(preset, catalog=catalog, md=md)
        return self.submit_plan(
            item.name, args=item.args, kwargs=item.kwargs, clear_pending=clear_pending
        )

    def request_pause(self) -> tuple[bool, str]:
        """Pause at the next step boundary."""
        with self._lock:
            if self._running is None:
                return False, "nothing is running"
            self._pause_requested = True
            self.streams.push_console_line(
                "request_pause (deferred): pausing at the next step boundary"
            )
            return True, "pause requested"

    def request_resume(self) -> tuple[bool, str]:
        """Resume a paused plan."""
        with self._lock:
            if self._re_state != "paused":
                return False, "not paused"
            self._re_state = "running"
            self._pause_requested = False
            self.streams.push_console_line("resumed")
            return True, "resumed"

    def stop_scan(self) -> tuple[bool, str]:
        """Stop the running plan gracefully."""
        with self._lock:
            if self._running is None:
                return False, "nothing is running"
            self.streams.push_console_line("stop_scan: stopping after the current shot")
            self._finish("stopped", "stopped by operator")
            return True, "stopped"

    def queue_items(self) -> list[dict]:
        """Waiting items, front first."""
        with self._lock:
            return [dict(i) for i in self._queue]

    def history_items(self) -> list[dict]:
        """Finished items, oldest first."""
        with self._lock:
            return [dict(i) for i in self._history]

    def running_item(self) -> Optional[dict]:
        """The running item, or ``None``."""
        with self._lock:
            return dict(self._running) if self._running else None

    def clear_queue(self) -> tuple[bool, str]:
        """Drop every waiting item."""
        with self._lock:
            n = len(self._queue)
            self._queue.clear()
            self.streams.push_console_line(f"clear_queue: {n} item(s) removed")
            return True, f"{n} item(s) removed"

    def close(self) -> None:
        """Nothing to release."""

    # ---------------------------------------------------------- simulation

    def step(self) -> None:
        """Advance one shot (or start the next item when idle)."""
        with self._lock:
            if self._running is None:
                if self._queue:
                    self._start_next()
                return
            if self._re_state != "running":
                return
            self._shots += 1
            self.streams.on_document(
                "event",
                {
                    "descriptor": self._desc_uid,
                    "seq_num": self._shots,
                    "time": self._clock(),
                },
            )
            boundary = self._shots % self._per_step == 0
            if boundary:
                step = self._shots // self._per_step
                self.streams.push_console_line(
                    f"step {step}/{max(1, self._total // self._per_step)} · {self._per_step} shots"
                )
            if self._shots >= self._total:
                self._finish("completed", "")
            elif boundary and self._pause_requested:
                # The manager's word, not a document: re_state says paused;
                # the progress picture keeps saying running, as on the worker.
                self._re_state = "paused"
                self._pause_requested = False
                self.streams.push_console_line("paused at step boundary")

    def _drive(self, period: float) -> None:
        while True:
            time.sleep(period)
            try:
                self.step()
            except Exception:  # noqa: BLE001 — a demo thread never dies loudly
                pass

    def _start_next(self) -> None:
        from geecs_scanner.service.summaries import summarize_item

        item = self._queue.pop(0)
        self._running = item
        self._re_state = "running"
        self._pause_requested = False
        self._scan_number += 1
        summary = summarize_item(item)
        self._per_step = summary.shots_per_step or 1
        self._total = summary.planned_shots or self._per_step
        self._shots = 0
        self._run_uid = uuid.uuid4().hex
        self._desc_uid = uuid.uuid4().hex
        md = (
            item["kwargs"].get("md")
            if isinstance(item["kwargs"].get("md"), dict)
            else {}
        )
        # A count's total is its num: num_points × 1, the way the worker's
        # start document spells it; a stepped plan is steps × shots_per_step.
        is_count = item["name"] == "count"
        self.streams.on_document(
            "start",
            {
                "uid": self._run_uid,
                "scan_number": self._scan_number,
                "plan_name": item["name"],
                "num_points": self._total if is_count else summary.steps,
                "shots_per_step": 1 if is_count else self._per_step,
                "time": self._clock(),
                "geecs": dict(md.get("geecs") or {}),
            },
        )
        self.streams.on_document(
            "descriptor",
            {"uid": self._desc_uid, "run_start": self._run_uid, "name": "primary"},
        )
        self.streams.push_console_line(
            f"Scan {self._scan_number:03d} claimed · {summary.text} · submitted by {item['user']}"
        )

    def _finish(self, exit_status: str, msg: str) -> None:
        item = self._running
        assert item is not None
        # RunEngine.stop() marks the run SUCCESSFUL (bluesky: "mark it as
        # successful (not aborted)"); only abort/halt write "abort"/"fail".
        # The manager's history says "stopped"; the documents say success.
        self.streams.on_document(
            "stop",
            {
                "run_start": self._run_uid,
                "exit_status": "success"
                if exit_status in ("completed", "stopped")
                else "abort",
                "reason": msg,
                "time": self._clock(),
            },
        )
        item = dict(item)
        item["result"] = {
            "exit_status": exit_status,
            "msg": msg,
            "time_start": self._clock(),
            "time_stop": self._clock(),
            "scan_ids": [self._scan_number],
            "run_uids": [self._run_uid],
        }
        self._history.append(item)
        self._running = None
        self._re_state = "idle"
        self.streams.push_console_line(
            f"Scan {self._scan_number:03d} {exit_status} after {self._shots} shots"
        )
        # The manager keeps going: the next waiting item starts at once.
        if self._queue:
            self._start_next()
