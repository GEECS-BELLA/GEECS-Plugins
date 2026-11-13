"""The commit log: what reached the configs repo, and whether it took effect.

One row per committed document (a save, an inbox publish, a foreign commit
the service applied or refused), kept as JSON lines in the state directory
and served on ``/log``. A row's ``state`` is one of four words:

- **applied**: the document's consumers have taken it up;
- **queued**: the apply waits for a moment (the worker idle and empty);
- **pending**: handed to someone named (the maintainer: a pull request, a
  restart, a refused commit to fix by PR);
- **nothing to do**: every reader reads per request.

**A landed commit is never written up as applied by that fact alone**: the
state is the apply seam's answer (:class:`Applier`). The seam is slice 3's
to implement (the worker's reopen, the archiver's onboard); until then
:class:`NoApplier` answers ``nothing to do`` where every consumer reads per
request and ``pending`` (the apply named) everywhere else, never
``applied``.

Rows are appended; a later line with the same ``id`` replaces the earlier
(how an apply will move a row from ``queued`` to ``applied``).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from geecs_configurator.kinds import CONSUMERS

State = Literal["applied", "queued", "pending", "nothing to do"]
STATES: tuple[str, ...] = ("applied", "queued", "pending", "nothing to do")
#: The kit's status word a state is drawn with (the chip): a dot and the state's own word beside it.
CHIP = {
    "applied": "ok",
    "queued": "queued",
    "pending": "degraded",
    "nothing to do": "unknown",
}


class Row(BaseModel):
    """One committed document and what became of it."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))  # fmt: skip
    #: saved (an operator commit), proposed (a pull request), published (the inbox), merged (a foreign commit applied), refused (one that failed the gate)
    source: Literal["saved", "proposed", "published", "merged", "refused"]
    who: str
    kind: str | None
    path: str
    commit: str | None = None
    link: str | None = None
    state: State
    detail: str = ""


class Answer(BaseModel):
    """The apply seam's answer for one committed document."""

    state: State
    detail: str = ""


class Applier(Protocol):
    """Makes a committed document take effect, or says when it will (slice 3 implements it)."""

    def apply(self, kind: str | None, path: str) -> Answer:
        """Apply (or queue, or hand off) the committed ``path`` of ``kind``; never claims what it did not do."""
        ...


class NoApplier:
    """No apply yet: ``nothing to do`` where every reader reads per request, else ``pending`` with the apply named."""

    def apply(self, kind: str | None, path: str) -> Answer:
        """The state a committed document is in while nothing applies it."""
        consumers = CONSUMERS.get(kind or "", [])
        verbs = sorted({c.apply for c in consumers if c.apply != "nothing to do"})
        if consumers and not verbs:
            return Answer(
                state="nothing to do", detail="every reader reads it per request"
            )
        return Answer(state="pending", detail=f"not applied: {'; '.join(verbs) or 'no reader known'} (applies arrive in slice 3)")  # fmt: skip


class CommitLog:
    """The JSON-lines log under the state directory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, row: Row) -> Row:
        """Append a row (or a newer version of one, by its ``id``)."""
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(row.model_dump_json() + "\n")
        return row

    def rows(self, limit: int = 200) -> list[Row]:
        """The newest rows first, each in its latest version; a line that does not parse is skipped."""
        if not self.path.is_file():
            return []
        latest: dict[str, Row] = {}
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                row = Row.model_validate(json.loads(line))
            except ValueError:
                continue
            latest[row.id] = row  # a newer version keeps the row's place
        return list(reversed(list(latest.values())))[:limit]

    def has_commit(self, commit: str, source: str) -> bool:
        """Whether a row of this source already names this commit (a refusal is logged once)."""
        return any(
            r.commit == commit and r.source == source for r in self.rows(limit=10_000)
        )
