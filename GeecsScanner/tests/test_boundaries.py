"""The scanner is a peer client of the queueserver; it imports nothing above or beside it.

Never the portal, the logbook or the console (peers), never the engine's
plans/devices (worker internals — the client seam is ``geecs_bluesky.qs_client``
plus the resolver and the plan-name tuple), and no facility literal in code.
"""

from __future__ import annotations

import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "geecs_scanner"

_FORBIDDEN_IMPORTS = [
    r"^\s*(from|import)\s+geecs_portal\b",
    r"^\s*(from|import)\s+geecs_logbook\b",
    r"^\s*(from|import)\s+geecs_console\b",
    r"^\s*(from|import)\s+geecs_mcp\b",
    r"^\s*from\s+geecs_bluesky\.plans\b",
    r"^\s*from\s+geecs_bluesky\.devices\b",
    r"^\s*from\s+geecs_bluesky\.run_engine\b",
    r"^\s*from\s+geecs_bluesky\.namespace\b",
    r"^\s*(from|import)\s+bluesky\.plans\b",
]

_FACILITY_LITERALS = [
    r"192\.168\.\d+\.\d+",
    r"\bUndulator\b",
    r"/home/\w+",
    r"America/Los_Angeles",
]


def _sources() -> list[Path]:
    return sorted(_PKG.rglob("*.py"))


def test_no_peer_or_engine_imports() -> None:
    offenders = []
    for path in _sources():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if any(re.search(p, line) for p in _FORBIDDEN_IMPORTS):
                offenders.append(f"{path.relative_to(_PKG)}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_no_facility_literal_in_code() -> None:
    offenders = []
    for path in _sources():
        text = path.read_text()
        for pattern in _FACILITY_LITERALS:
            for m in re.finditer(pattern, text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(_PKG)}:{line}: {m.group(0)}")
    assert not offenders, "\n".join(offenders)


def test_service_layer_imports_no_web_framework() -> None:
    for path in sorted((_PKG / "service").rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            assert not re.match(
                r"^\s*(from|import)\s+(fastapi|starlette|uvicorn)\b", line
            ), f"{path.relative_to(_PKG)}:{n}: {line.strip()}"
