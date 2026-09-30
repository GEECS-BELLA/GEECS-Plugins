"""Run a ``.himg`` folder job in a child process, streaming progress back.

A conversion, a compaction or a restore streams every frame of a scan —
tens of gigabytes — through whichever process runs it, for minutes.  Run
inside a web service that is a wedge waiting to happen (one core busy, the
page cache charged to the service's cgroup, no progress on the page — the
portal's 1806-shot conversion of 2026-09-29).  So a host that is not a
shell runs the job here, in a child interpreter, and reads a line-oriented
event stream from it:

- ``{"event": "progress", "done": n, "total": m, "phase": "verifying"}``
  after each frame (the :data:`~geecs_data_utils.io.himg_stack.Progress`
  callback, relayed);
- ``{"event": "log", "level": 20, "name": "...", "message": "..."}`` for
  every log record the job emits, re-logged in the parent under the same
  logger name and level, so a captured run log reads as if in-process;
- ``{"event": "report", "kind": "HimgCompactReport", "fields": {...}}``
  once, at the end — rebuilt into the same dataclass the in-process
  function returns;
- ``{"event": "error", "type": "HimgStackExists", "message": "...",
  "attrs": {...}}`` when the job raises one of this package's errors —
  re-raised as that class in the parent, attributes included.

The child is ``python -m geecs_data_utils.io.himg_worker <job json>`` with
the parent's own interpreter (``sys.executable``) and environment, so it
imports exactly the code the parent would have run.  Anything else the
child writes (a warning on stderr, an uncaught traceback) is logged by
the parent line by line and becomes an :class:`HimgStackError` if the
child exits without a report.

Nothing here creates a directory or touches a file: the job functions in
:mod:`~geecs_data_utils.io.himg_stack` and
:mod:`~geecs_data_utils.io.himg_compact` do the work and own the guards.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from geecs_data_utils.io import himg_compact, himg_stack
from geecs_data_utils.io.himg_stack import HimgStackError, Progress

logger = logging.getLogger(__name__)

__all__ = ["JOB_COMMANDS", "main", "run_himg_job", "run_job_in_process"]

#: The jobs the worker knows, by the ``command`` key of a job.
JOB_COMMANDS = ("convert", "verify", "compact", "restore")

_REPORTS: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        himg_stack.HimgStackReport,
        himg_stack.HimgVerifyReport,
        himg_compact.HimgCompactReport,
        himg_compact.HimgRestoreReport,
    )
}
_ERRORS: dict[str, type[HimgStackError]] = {
    cls.__name__: cls
    for cls in (
        HimgStackError,
        himg_stack.NoHimgFiles,
        himg_stack.HimgStackExists,
        himg_stack.HimgStampsUnavailable,
        himg_stack.HimgVerificationFailed,
        himg_stack.HimgSourcesDeleted,
        himg_compact.NoHimgStack,
        himg_compact.HimgFolderActive,
        himg_compact.HimgStackIncomplete,
        himg_compact.HimgSourceChanged,
    )
}
#: Exception attributes worth carrying across the process boundary.
_ERROR_ATTRS = ("stack_path", "mismatches", "missing", "changed")


# ---------------------------------------------------------------- the jobs


def run_job_in_process(job: dict, *, progress: Optional[Progress] = None):
    """Run one job here and return its report — the child's body, usable directly.

    Parameters
    ----------
    job : dict
        ``{"command": ..., "device_dir": ...}`` plus the command's options:
        ``convert`` takes ``device`` (the stamp attribute's device name),
        ``rows_path`` (the scan's scalar table for legacy names),
        ``verify``, ``overwrite``, ``compression_level``; ``verify`` takes
        ``against_files``; ``compact`` takes ``min_age`` and
        ``require_closed``; ``restore`` takes nothing more.
    progress : callable, optional
        Passed straight to the job function.
    """
    command = job.get("command")
    device_dir = Path(job["device_dir"])
    if command == "convert":
        rows = None
        rows_path = job.get("rows_path")
        if rows_path:
            from geecs_data_utils.data.sfile import read_sfile

            rows = read_sfile(Path(rows_path))
        return himg_stack.convert_himg_folder(
            device_dir,
            rows=rows,
            device=job.get("device"),
            verify=job.get("verify", True),
            overwrite=job.get("overwrite", False),
            compression_level=job.get(
                "compression_level", himg_stack.DEFAULT_COMPRESSION_LEVEL
            ),
            progress=progress,
        )
    if command == "verify":
        return himg_stack.verify_himg_stack(
            himg_stack.stack_path_for(device_dir),
            against_files=job.get("against_files", False),
            progress=progress,
        )
    if command == "compact":
        return himg_compact.compact_himg_folder(
            device_dir,
            min_age=job.get("min_age", himg_compact.MIN_SOURCE_AGE_S),
            require_closed=job.get("require_closed", True),
            progress=progress,
        )
    if command == "restore":
        return himg_compact.restore_himg_folder(device_dir, progress=progress)
    raise ValueError(f"unknown himg job {command!r}; known: {JOB_COMMANDS}")


# --------------------------------------------------------------- the child


class _EventLogHandler(logging.Handler):
    """Relay the child's log records to the parent as ``log`` events."""

    def __init__(self, emit_event: Callable[[dict], None]):
        super().__init__(level=logging.DEBUG)
        self._emit_event = emit_event

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_event(
                {
                    "event": "log",
                    "level": record.levelno,
                    "name": record.name,
                    "message": record.getMessage(),
                }
            )
        except Exception:  # noqa: BLE001 — a broken record must not kill the job
            self.handleError(record)


def _wire_report(report) -> dict:
    fields = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(report).items()
    }
    return {"event": "report", "kind": type(report).__name__, "fields": fields}


def _wire_error(exc: HimgStackError) -> dict:
    attrs: dict[str, Any] = {}
    for name in _ERROR_ATTRS:
        value = getattr(exc, name, None)
        if value is None:
            continue
        attrs[name] = str(value) if isinstance(value, Path) else list(value)
    return {
        "event": "error",
        "type": type(exc).__name__,
        "message": str(exc),
        "attrs": attrs,
    }


def _die_with_parent() -> None:
    """Ask the kernel to SIGTERM this child when its parent dies (Linux only).

    Under systemd the unit's cgroup kill covers a portal that stops; this
    covers a parent that dies any other way, so a compaction never runs
    on unattended after the host that asked for it is gone.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):  # no libc, no prctl: the cgroup is the net
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The child: run the job in ``argv[0]`` (JSON) and stream events on stdout."""
    _die_with_parent()
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m geecs_data_utils.io.himg_worker '<job json>'",
            file=sys.stderr,
        )
        return 2
    out = sys.stdout

    def emit(event: dict) -> None:
        out.write(json.dumps(event) + "\n")
        out.flush()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(_EventLogHandler(emit))
    try:
        job = json.loads(args[0])
        report = run_job_in_process(
            job,
            progress=lambda done, total, phase: emit(
                {"event": "progress", "done": done, "total": total, "phase": phase}
            ),
        )
    except HimgStackError as exc:
        emit(_wire_error(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 — every outcome becomes an event
        emit(_wire_error(HimgStackError(f"{type(exc).__name__}: {exc}")))
        return 1
    emit(_wire_report(report))
    return 0


# -------------------------------------------------------------- the parent


def _unwire_report(event: dict):
    cls = _REPORTS.get(event.get("kind", ""))
    if cls is None:
        raise HimgStackError(f"worker returned an unknown report {event.get('kind')!r}")
    fields = dict(event.get("fields", {}))
    for field in dataclasses.fields(cls):
        annotation = str(field.type)
        if "Path" in annotation and isinstance(fields.get(field.name), str):
            fields[field.name] = Path(fields[field.name])
        elif "tuple" in annotation and isinstance(fields.get(field.name), list):
            fields[field.name] = tuple(fields[field.name])
    return cls(**fields)


def _unwire_error(event: dict) -> HimgStackError:
    cls = _ERRORS.get(event.get("type", ""), HimgStackError)
    exc = cls.__new__(cls)
    Exception.__init__(exc, event.get("message", "worker failed"))
    for name, value in (event.get("attrs") or {}).items():
        if name == "stack_path":
            value = Path(value)
        elif isinstance(value, list):
            value = tuple(value)
        setattr(exc, name, value)
    return exc


def run_himg_job(
    job: dict,
    *,
    progress: Optional[Progress] = None,
    python: Optional[str] = None,
):
    """Run *job* in a child interpreter; return its report or raise its error.

    Parameters
    ----------
    job : dict
        As for :func:`run_job_in_process`.
    progress : callable, optional
        Called with each ``progress`` event the child sends.
    python : str, optional
        The interpreter; the parent's own by default.

    Raises
    ------
    HimgStackError
        The job's own error (its class, message and attributes carried
        over), or the child exiting without a report.
    """
    command = [
        python or sys.executable,
        "-m",
        "geecs_data_utils.io.himg_worker",
        json.dumps(job),
    ]
    report = error = None
    noise: list[str] = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    ) as child:
        assert child.stdout is not None
        for raw in child.stdout:
            line = raw.rstrip("\n")
            event = _parse_event(line)
            if event is None:
                noise.append(line)
                logger.warning("himg worker: %s", line)
                continue
            kind = event.get("event")
            if kind == "progress":
                if progress is not None:
                    progress(
                        int(event["done"]), int(event["total"]), str(event["phase"])
                    )
            elif kind == "log":
                logging.getLogger(str(event.get("name", __name__))).log(
                    int(event.get("level", logging.INFO)), "%s", event.get("message")
                )
            elif kind == "report":
                report = _unwire_report(event)
            elif kind == "error":
                error = _unwire_error(event)
        code = child.wait()
    if error is not None:
        raise error
    if report is None:
        tail = " | ".join(noise[-5:]) if noise else "no output"
        raise HimgStackError(f"himg worker exited with {code} and no report ({tail})")
    return report


def _parse_event(line: str) -> Optional[dict]:
    """The event a line carries, or ``None`` for anything that is not one."""
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) and "event" in event else None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
