"""Hardware-free preview, isolated behind time, memory and response budgets."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time

import psutil
from geecs_schemas import Sweep
from pydantic import BaseModel

from .errors import ScannerError

MAX_INPUT_BYTES = 256 * 1024
MAX_VALUES = 250_000
MAX_RESPONSE_VALUES = 10_000
MAX_RSS = 256 * 1024 * 1024
TIMEOUT = 8.0
_SLOTS = threading.BoundedSemaphore(2)


class TrajectoryAxisOut(BaseModel):
    """One axis, in its requested frame; relative values are offsets."""

    axis: str
    relative: bool
    positions: list[float]


class TrajectoryOut(BaseModel):
    """Sampled display coordinates, with full trajectory size disclosed."""

    axes: list[TrajectoryAxisOut]
    indices: list[int]
    total_steps: int
    sampled: bool


def preview(sweep: Sweep) -> TrajectoryOut:
    """Expand without consulting a worker, namespace, catalog or gateway.

    Curved patterns have no algebraic size; run upstream's geometry in a
    disposable process. Two concurrent previews at most, with wall time and
    resident memory monitored by the parent. Oversized results are refused,
    never truncated into an executable trajectory.
    """
    encoded = sweep.model_dump_json()
    if len(encoded.encode()) > MAX_INPUT_BYTES:
        raise ScannerError(
            "invalid_request", "Trajectory input exceeds the 256 KiB preview budget."
        )
    count = sweep.n_steps()
    if count is not None and count * len(sweep.axis_references()) > MAX_VALUES:
        raise ScannerError(
            "invalid_request",
            "Trajectory exceeds the 250,000-coordinate preview budget. Reduce the point count.",
        )
    if not _SLOTS.acquire(blocking=False):
        raise ScannerError("policy_refusal", "Preview is busy; try again shortly.")
    try:
        with subprocess.Popen(
            [sys.executable, "-m", "geecs_scanner.service.trajectory"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as child:
            monitor = psutil.Process(child.pid)
            deadline = time.monotonic() + TIMEOUT
            try:
                pending = encoded
                while True:
                    try:
                        output, error = child.communicate(input=pending, timeout=0.05)
                        break
                    except subprocess.TimeoutExpired:
                        pending = None
                        if time.monotonic() >= deadline:
                            raise ScannerError(
                                "invalid_request",
                                "Pattern exceeded the preview time budget. Use a larger radial step or fewer points.",
                            )
                        try:
                            if monitor.memory_info().rss > MAX_RSS:
                                raise ScannerError(
                                    "invalid_request",
                                    "Pattern exceeded the preview memory budget. Use fewer points.",
                                )
                        except psutil.NoSuchProcess:
                            pass
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate()
            if child.returncode:
                raise ScannerError(
                    "invalid_request",
                    "Trajectory calculation failed: "
                    + (error.strip()[-400:] or "preview process stopped"),
                )
            result = json.loads(output)
            if "error" in result:
                raise ScannerError("invalid_request", result["error"])
            return TrajectoryOut.model_validate(result)
    finally:
        _SLOTS.release()


def _expand(sweep: Sweep) -> TrajectoryOut:
    from geecs_bluesky.trajectory import sweep_to_cycler

    points = sweep_to_cycler(sweep, str)
    axes = sweep.axis_references()
    count = len(points)
    if count * len(axes) > MAX_VALUES:
        raise ValueError("Trajectory exceeds the 250,000-coordinate preview budget.")
    limit = min(2000, MAX_RESPONSE_VALUES // len(axes))
    if limit < 2:
        raise ValueError("Too many axes for the preview response budget.")
    stride = max(1, math.ceil((count - 1) / (limit - 1)))
    indices = list(range(0, count, stride))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    values = points.by_key()
    return TrajectoryOut(
        axes=[
            TrajectoryAxisOut(
                axis=a.axis,
                relative=a.relative,
                positions=[values[a.axis][i] for i in indices],
            )
            for a in axes
        ],
        indices=indices,
        total_steps=count,
        sampled=len(indices) != count,
    )


if __name__ == "__main__":
    try:
        result = _expand(Sweep.model_validate_json(sys.stdin.read(MAX_INPUT_BYTES + 1)))
        print(result.model_dump_json())
    except (ValueError, ArithmeticError, MemoryError) as exc:
        print(
            json.dumps(
                {"error": str(exc) or "Trajectory exceeds available preview memory."}
            )
        )
