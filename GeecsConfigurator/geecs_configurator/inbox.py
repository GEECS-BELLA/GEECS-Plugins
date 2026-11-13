"""The inbox: files the worker writes, committed to main by the service.

Nothing writes into the live checkout (git refuses to fast-forward over an
uncommitted file, identical bytes included), so the worker's one write,
``measure_shot_offsets`` saving ``shot_offsets.yaml``, lands in the inbox:
``<state>/inbox/`` mirrors the configs root, and the worker's write root
points at it. The worker writes atomically (a temporary file renamed over
the target), so a file at its path is always whole.

A reconcile, per inbox file:

1. **Takes it** by renaming it into ``.claimed/`` as ``<name>.new``
   (atomic). A newer measurement the worker writes meanwhile lands at the
   original path, a new file, untouched by this publish.
2. Checks it is a valid document of its kind; an invalid one is moved to
   ``.rejected/`` (kept, logged), never committed, and supersedes nothing.
3. A valid one becomes **the claim** (``<name>``), replacing an older claim
   still waiting from a failed push: the newer measurement supersedes it,
   so an older is never published after a newer.
4. Commits the claim's bytes in a staging worktree on top of main,
   authored as the worker, and pushes: **the measurement wins** if main
   changed the same file meanwhile (logged). The live checkout
   fast-forwards, and the claim is removed. On a failed push it stays, and
   the inbox backs off (5 s doubling to 5 min) before the next try, so an
   unreachable origin does not hold the repository's lock every tick.

It loops until the inbox is empty, so a measurement that arrived during a
publish is published by the same reconcile, after the older one.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass

import yaml
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, FileStorage
from pydantic import ValidationError

from geecs_configurator.repository import ConfigsRepository, GitError, Identity

logger = logging.getLogger(__name__)

#: The kinds the worker writes. Anything else in the inbox is left alone and logged.
INBOX_KINDS = ("shot_offsets",)
CLAIMED, REJECTED = ".claimed", ".rejected"
#: After a failed publish: the first wait, and the most it doubles to (seconds).
BACKOFF_S, BACKOFF_MAX_S = 5.0, 300.0
#: Per inbox: (monotonic time before which no publish is tried, failures in a row).
_backoff: dict[Path, tuple[float, int]] = {}


def reset_backoff(repo: ConfigsRepository) -> None:
    """Forget the failures: the next reconcile tries at once."""
    _backoff.pop(repo.inbox, None)


@dataclass(frozen=True)
class Published:
    """One inbox file's outcome: committed (``commit`` None when it changed nothing), or rejected."""

    path: str
    outcome: str  # "committed", "nothing to do", "rejected", "failed"
    kind: str = "shot_offsets"
    commit: str | None = None
    detail: str = ""


def _claimed_name(rel: str) -> str:
    return rel.replace("/", "%")


def inbox_paths(repo: ConfigsRepository, experiments: list[str]) -> dict[str, str]:
    """Each inbox kind's file per experiment: path below the configs root -> kind."""
    out = {}
    for kind in INBOX_KINDS:
        storage = KIND_STORAGE[kind]
        if not isinstance(storage, FileStorage):
            continue
        pattern = storage.path_pattern()
        for experiment in experiments:
            out[pattern.replace("<experiment>", experiment)] = kind
    return out


def prepare(repo: ConfigsRepository, experiments: list[str]) -> None:
    """Create the inbox's folders for every experiment (the worker refuses to create its experiment folder)."""
    for rel in inbox_paths(repo, experiments):
        (repo.inbox / rel).parent.mkdir(parents=True, exist_ok=True)
    for sub in (CLAIMED, REJECTED):
        (repo.inbox / sub).mkdir(parents=True, exist_ok=True)


def pending(repo: ConfigsRepository, experiments: list[str]) -> bool:
    """Whether anything waits in the inbox (no git, no network: the timer's cheap check)."""
    claimed = repo.inbox / CLAIMED
    return any(
        (repo.inbox / rel).is_file() for rel in inbox_paths(repo, experiments)
    ) or (claimed.is_dir() and any(claimed.iterdir()))


def _valid(kind: str, data: bytes) -> str | None:
    """Why the bytes are not a valid document of ``kind``, or ``None``."""
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
        SCHEMA_REGISTRY[kind].model_validate(raw)
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        return f"does not parse: {exc}"
    except ValidationError as exc:
        return f"does not validate: {exc.errors()[0]['msg']}"
    return None


def reconcile(repo: ConfigsRepository, experiments: list[str], author: Identity) -> list[Published]:  # fmt: skip
    """Publish every inbox file to main, oldest first, until the inbox is empty or a push fails."""
    out: list[Published] = []
    until, failures = _backoff.get(repo.inbox, (0.0, 0))
    if time.monotonic() < until:
        return out
    with repo.lock:
        for rel, kind in inbox_paths(repo, experiments).items():
            while True:
                result = _publish_one(repo, rel, kind, author)
                if result is None:
                    break
                out.append(result)
                if result.outcome == "failed":
                    wait = min(BACKOFF_MAX_S, BACKOFF_S * 2**failures)
                    _backoff[repo.inbox] = (time.monotonic() + wait, failures + 1)
                    return out
    reset_backoff(repo)
    return out


def _publish_one(repo: ConfigsRepository, rel: str, kind: str, author: Identity) -> Published | None:  # fmt: skip
    src, claimed = repo.inbox / rel, repo.inbox / CLAIMED / _claimed_name(rel)
    incoming = claimed.with_name(claimed.name + ".new")
    if (
        not incoming.exists() and src.exists()
    ):  # a leftover .new (a crash) goes first: it is older
        os.replace(src, incoming)
    if incoming.exists():
        why = _valid(kind, incoming.read_bytes())
        if (
            why is not None
        ):  # rejected, superseding nothing: an older valid claim still publishes
            rejected = repo.inbox / REJECTED / f"{_claimed_name(rel)}.{os.stat(incoming).st_mtime_ns}"  # fmt: skip
            os.replace(incoming, rejected)
            logger.error("%s from the inbox %s; kept as %s, not committed", rel, why, rejected.name)  # fmt: skip
            return Published(rel, "rejected", kind=kind, detail=why)
        if claimed.exists():
            logger.info("%s: an unpublished older measurement is superseded by a newer one", rel)  # fmt: skip
        os.replace(incoming, claimed)
    if not claimed.exists():
        return None
    data = claimed.read_bytes()
    try:
        repo.fetch()
        base = repo.remote_head()
        sha = repo.commit_file(rel, data, author=author, message=f"{kind} {rel}: measured by the worker, committed via configurator\n", base=base)  # fmt: skip
    except GitError as exc:
        logger.warning(
            "%s from the inbox was not published (kept, retried): %s", rel, exc
        )
        return Published(rel, "failed", kind=kind, detail=str(exc))
    claimed.unlink()
    return Published(
        rel, "committed" if sha else "nothing to do", kind=kind, commit=sha
    )
