"""The inbox: files the worker writes, committed to main by the service.

Nothing writes into the live checkout (git refuses to fast-forward over an
uncommitted file, identical bytes included), so the worker's one write,
``measure_shot_offsets`` saving ``shot_offsets.yaml``, lands in the inbox:
``<state>/inbox/`` mirrors the configs root, and the worker's write root
points at it. The worker writes atomically (a temporary file renamed over
the target), so a file at its path is always whole.

A reconcile, per inbox file:

1. **Claims it** by renaming it into ``.claimed/`` (atomic). A newer
   measurement the worker writes meanwhile lands at the original path, a
   new file, untouched by this publish.
2. Checks it is a valid document of its kind; an invalid one is moved to
   ``.rejected/`` (kept, logged) and never committed.
3. Commits the worker's bytes in a staging worktree on top of main,
   authored as the worker, and pushes: **the measurement wins** if main
   changed the same file meanwhile (logged). The live checkout
   fast-forwards.
4. Removes the claimed file. On a failed push it stays claimed; the next
   reconcile publishes it, unless a newer measurement is waiting at the
   original path, which supersedes it (the older is dropped, never
   published after the newer).

It loops until the inbox is empty, so a measurement that arrived during a
publish is published by the same reconcile, after the older one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import yaml
from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, FileStorage
from pydantic import ValidationError

from geecs_configurator.repository import ConfigsRepository, GitError, Identity

logger = logging.getLogger(__name__)

#: The kinds the worker writes. Anything else in the inbox is left alone and logged.
INBOX_KINDS = ("shot_offsets",)
CLAIMED, REJECTED = ".claimed", ".rejected"


@dataclass(frozen=True)
class Published:
    """One inbox file's outcome: committed (``commit`` None when it changed nothing), or rejected."""

    path: str
    outcome: str  # "committed", "nothing to do", "rejected", "failed"
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
    with repo.lock:
        for rel, kind in inbox_paths(repo, experiments).items():
            while True:
                result = _publish_one(repo, rel, kind, author)
                if result is None:
                    break
                out.append(result)
                if result.outcome == "failed":
                    return out
    return out


def _publish_one(repo: ConfigsRepository, rel: str, kind: str, author: Identity) -> Published | None:  # fmt: skip
    src, claimed = repo.inbox / rel, repo.inbox / CLAIMED / _claimed_name(rel)
    if claimed.exists() and src.exists():
        claimed.unlink()  # a newer measurement waits: the older is superseded
        logger.info(
            "%s: an unpublished older measurement is superseded by a newer one", rel
        )
    if not claimed.exists():
        if not src.exists():
            return None
        os.replace(src, claimed)
    data = claimed.read_bytes()
    why = _valid(kind, data)
    if why is not None:
        rejected = repo.inbox / REJECTED / f"{_claimed_name(rel)}.{os.stat(claimed).st_mtime_ns}"  # fmt: skip
        os.replace(claimed, rejected)
        logger.error("%s from the inbox %s; kept as %s, not committed", rel, why, rejected.name)  # fmt: skip
        return Published(rel, "rejected", detail=why)
    try:
        repo.fetch()
        base = repo.remote_head()
        sha = repo.commit_file(rel, data, author=author, message=f"{kind} {rel}: measured by the worker, committed via configurator\n", base=base)  # fmt: skip
    except GitError as exc:
        logger.warning(
            "%s from the inbox was not published (kept, retried): %s", rel, exc
        )
        return Published(rel, "failed", detail=str(exc))
    claimed.unlink()
    return Published(rel, "committed" if sha else "nothing to do", commit=sha)
