"""The scan-log list: the variables every scan's logbook entry shows at its start.

One list per experiment, shared by every operator of this scanner. The page
edits it; :meth:`ScannerService.submit` stamps it into each submission's
``md["geecs"]["scan_log_variables"]``; the worker copies it into the scan's
``ScanInfo`` ini (``[Scan Log]``); the logbook reads the values from the
s-file's first row. Nothing here reads a value.

Storage is a JSON file in the scanner's own state directory (systemd's
``StateDirectory=geecs-scanner``), written atomically. Not the configs
repo: an operator list edited daily would keep that checkout dirty, and
once committed it would block ``git pull`` on the host.

Edits are optimistic: every answer carries ``version`` (a hash of the list,
so a hand edit of the file is a new version too), and a save names the
version it was made from; a stale one is refused with the current list.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import (
    LoggedVariableOut,
    LoggedVariablesOut,
    ScanLogListOut,
)

logger = logging.getLogger(__name__)

#: The file in the state directory.
FILE_NAME = "scan_log_variables.json"
#: A bound on the list, not a policy: the logbook shows them as columns.
MAX_VARIABLES = 64


def default_state_dir() -> Path:
    """Systemd's ``$STATE_DIRECTORY``, else ``~/.local/state/geecs-scanner``."""
    state = os.environ.get("STATE_DIRECTORY", "").split(":")[0]
    if state:
        return Path(state)
    return Path.home() / ".local" / "state" / "geecs-scanner"


def version_of(variables: Sequence[str]) -> str:
    """The list's content hash — what a save must name."""
    payload = json.dumps(list(variables)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def normalise(variables: Sequence[Any]) -> list[str]:
    """Strip, drop blanks and duplicates (first wins); refuse a malformed name."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in variables:
        if not isinstance(raw, str):
            raise ScannerError("invalid_request", f"not a variable name: {raw!r}")
        name = raw.strip()
        if not name:
            continue
        device, sep, variable = name.partition(":")
        if not sep or not device.strip() or not variable.strip():
            raise ScannerError(
                "invalid_request", f"{name!r} is not a Device:Variable name"
            )
        if name not in seen:
            seen.add(name)
            out.append(name)
    if len(out) > MAX_VARIABLES:
        raise ScannerError(
            "invalid_request",
            f"{len(out)} variables; the scan-log list holds at most {MAX_VARIABLES}",
        )
    return out


class ScanLogListStore(Protocol):
    """Where the list lives: a file in production, memory in demo."""

    def load(self) -> list[str]:
        """The stored list (empty when none was ever saved)."""

    def save(self, variables: list[str]) -> None:
        """Replace the stored list."""


class FileListStore:
    """The list as ``{"variables": [...]}`` in one JSON file, replaced atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[str]:
        """The stored list; a missing file is an empty list, a broken one an error."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        try:
            document = json.loads(text)
            names = document["variables"]
            if not isinstance(names, list):
                raise TypeError("variables is not a list")
            return [str(n) for n in names]
        except (ValueError, KeyError, TypeError) as exc:
            raise ScannerError(
                "internal_error", f"{self.path} is not a scan-log list: {exc}"
            ) from exc

    def save(self, variables: list[str]) -> None:
        """Write through a temp file beside the target and ``os.replace``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.stem}-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"variables": variables}, fh, indent=1)
                fh.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


class MemoryListStore:
    """The demo's list, in memory."""

    def __init__(self, variables: Sequence[str] = ()) -> None:
        self._variables = list(variables)

    def load(self) -> list[str]:
        """The list."""
        return list(self._variables)

    def save(self, variables: list[str]) -> None:
        """Replace the list."""
        self._variables = list(variables)


class ScanLogList:
    """Read and edit the list under one lock of its own (never the client's)."""

    def __init__(self, store: ScanLogListStore) -> None:
        self._store = store
        self._lock = threading.Lock()

    def get(self) -> ScanLogListOut:
        """The current list and its version."""
        with self._lock:
            names = self._store.load()
        return ScanLogListOut(variables=names, version=version_of(names))

    def put(self, variables: Sequence[Any], version: str) -> ScanLogListOut:
        """Replace the list if *version* is current; else refuse with the current one."""
        names = normalise(variables)
        with self._lock:
            current = self._store.load()
            if version != version_of(current):
                raise ScannerError(
                    "conflict",
                    "the scan-log list changed since this page loaded it",
                    current=ScanLogListOut(
                        variables=current, version=version_of(current)
                    ).model_dump(),
                )
            self._store.save(names)
        logger.info("scan-log list saved: %s", ", ".join(names) or "(empty)")
        return ScanLogListOut(variables=names, version=version_of(names))

    def for_submission(self) -> list[str]:
        """The list to stamp into a scan; a store failure stamps nothing."""
        try:
            with self._lock:
                return self._store.load()
        except (ScannerError, OSError) as exc:  # a list is never worth a refused scan
            logger.warning(
                "scan-log list unreadable; scan submitted without it: %s", exc
            )
            return []


def build_logged_variables(
    all_by_device: Mapping[str, Sequence[str]],
    logged_by_device: Mapping[str, Sequence[str]],
) -> list[LoggedVariableOut]:
    """Every variable as ``Device:Variable``, logged ones first, then by name."""
    items = [
        LoggedVariableOut(
            name=f"{device}:{variable}",
            logged=variable in set(logged_by_device.get(device, ())),
        )
        for device, variables in all_by_device.items()
        for variable in variables
    ]
    items.extend(
        LoggedVariableOut(name=f"{device}:{variable}", logged=True)
        for device, variables in logged_by_device.items()
        if device not in all_by_device
        for variable in variables
    )
    items.sort(key=lambda item: (not item.logged, item.name.lower()))
    return items


class DbLoggedVariables:
    """The experiment's variables and their ``get='yes'`` flag, read once and kept.

    The same cached read as :class:`~geecs_scanner.service.settables.DbSettables`
    (:class:`~geecs_scanner.service.settables.CachedDbRead`): the roster changes
    with a DB edit, which restarts the gateway anyway; a failed read is
    reported and not cached.
    """

    def __init__(self, experiment: str, *, db: Any = None) -> None:
        from geecs_scanner.service.settables import CachedDbRead

        self._read = CachedDbRead(
            lambda db: build_logged_variables(
                db.get_all_experiment_variables(experiment),
                db.get_subscribed_variables(experiment),
            ),
            what="logged variables",
            db=db,
        )

    def logged_variables(self) -> LoggedVariablesOut:
        """The cached list, or one pair of DB reads to fill it."""
        items, detail = self._read.get()
        return LoggedVariablesOut(items=items or [], source="db", detail=detail)
