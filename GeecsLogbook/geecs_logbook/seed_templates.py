"""Seed templates: the type buttons, read from a directory of markdown files.

A *type* of entry — a laser note, a shift handover, a fault — is a
template file in the configs checkout the portal already reads::

    logbook_templates/
      laser.md
      handover.md
      fault.md

Each file is a small front-matter header and a body::

    ---
    label: Laser
    colour: ok
    book: ops
    order: 10
    ---
    ### Laser
    #laser

The **body** is the prefill: pressing the button puts it in the composer
and the author types from there. It carries the type's ``#tag`` so a
button press and a typed tag are the same thing (see
:mod:`geecs_logbook.tags`). The entry records the file's stem as its
``template`` — provenance, never structure: the body is the author's from
the first save and a later edit to the file changes no stored entry.
Templates seed; they never enforce.

The header keys, all optional:

``label``
    The button's text. Default: the stem, title-cased.
``colour``
    A *theme token name* — one of :data:`TONES` — never a literal colour,
    so the button and the chip follow whichever palette the viewer picked
    (GeecsWebTheme's guard walks the stylesheet, not the configs repo, so
    the vocabulary is closed here). An unknown name falls back to
    ``accent`` with a warning rather than dropping the button.
``book``
    ``scans``, ``ops`` or ``both`` (default): which composers offer it.
``order``
    Sort key for the button row; ties by label.

Adding a file adds a button; no code change. The directory is on the
share, so it is read once at start and refreshed in the background when
stale — a page never waits on the share for its buttons, and a failed
refresh keeps the last good set.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from geecs_logbook.tags import parse_tags

logger = logging.getLogger(__name__)

#: The colour vocabulary a template may name. Each is a GeecsWebTheme token
#: with a ``.tone-<name>`` rule in ``scanlog.css`` (pinned by test), which
#: is how a name in a file becomes a colour on the page without a literal
#: anywhere.
TONES: tuple[str, ...] = (
    "accent",
    "ok",
    "warn",
    "crit",
    "agent",
    "muted",
    "trace-1",
    "trace-2",
    "trace-3",
    "trace-4",
)

#: Where the files live: this directory at the top of the configs checkout
#: (the portal derives it from its analysis tree's parent).
TEMPLATES_DIRNAME = "logbook_templates"

#: Template names an entry carries without a button: what the composer
#: sends when none was pressed, and two the pages render quietly. A file
#: with one of these stems is refused, or every hand-typed entry would
#: wear its chip.
RESERVED_NAMES: tuple[str, ...] = ("blank", "scan_note", "day_intro")

#: A template's name is its file stem, and it is stored on every entry that
#: started from it — so it is bounded like ``LogEntry.template``.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

#: How long a loaded set is served before a background refresh is kicked.
REFRESH_INTERVAL_S = 60.0

TemplateBook = Literal["scans", "ops", "both"]


class SeedTemplate(BaseModel):
    """One type button: what it is called, how it is coloured, what it inserts."""

    name: str = Field(description="The file stem; stored as the entry's template.")
    label: str = Field(description="Button text.")
    colour: str = Field("accent", description="A theme token name from TONES.")
    book: TemplateBook = Field("both", description="Which book's composers offer it.")
    order: int = Field(100, description="Button row sort key.")
    body: str = Field("", description="The prefill, with its #tag.")
    tags: list[str] = Field(
        default_factory=list,
        description="The tags the prefill carries — what the type chip already says.",
    )

    def offered_in(self, book: str) -> bool:
        """Whether this template belongs on ``book``'s composers."""
        return self.book == "both" or self.book == book


def parse_template(name: str, text: str) -> SeedTemplate:
    """Parse one template file's text.

    The header is ``key: value`` lines between two ``---`` lines, read
    without a YAML library so the format stays exactly what the docstring
    says. A file with no header is all body.
    """
    header: dict[str, str] = {}
    body = text
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body = "\n".join(lines[i + 1 :])
                break
            key, sep, value = line.partition(":")
            if sep and key.strip():
                header[key.strip().lower()] = value.strip()
        else:  # an opening fence with no close: the whole file is body
            header = {}
            body = text

    colour = header.get("colour", header.get("color", "accent")).lower()
    if colour not in TONES:
        logger.warning(
            "logbook template %s: colour %r is not a theme tone (%s); using accent",
            name,
            colour,
            ", ".join(TONES),
        )
        colour = "accent"
    book = header.get("book", "both").lower()
    if book not in ("scans", "ops", "both"):
        logger.warning(
            "logbook template %s: book %r is not scans/ops/both; offering in both",
            name,
            book,
        )
        book = "both"
    try:
        order = int(header.get("order", "100"))
    except ValueError:
        logger.warning(
            "logbook template %s: order %r is not a number", name, header["order"]
        )
        order = 100
    label = header.get("label") or name.replace("_", " ").replace("-", " ").title()
    body = body.lstrip("\n").rstrip() + ("\n" if body.strip() else "")
    return SeedTemplate(
        name=name,
        label=label,
        colour=colour,
        book=book,  # type: ignore[arg-type]
        order=order,
        body=body,
        tags=parse_tags(body),
    )


def load_templates(directory: Path) -> list[SeedTemplate]:
    """Read every ``*.md`` in ``directory``, sorted for the button row.

    A file whose stem is not a valid template name is skipped with a
    warning, and a ``README.md`` silently — the directory is meant to be
    browsed. A missing directory is an empty set, not an error: a site
    without templates has plain composers.
    """
    if not directory.is_dir():
        return []
    found: list[SeedTemplate] = []
    for path in sorted(directory.glob("*.md")):
        if path.stem.upper() == "README":  # a browsed directory has one
            continue
        if path.stem in RESERVED_NAMES:
            logger.warning(
                "logbook template %s: %r is reserved for entries with no template; skipped",
                path.name,
                path.stem,
            )
            continue
        if not _NAME.match(path.stem):
            logger.warning(
                "logbook template %s: name is not usable; skipped", path.name
            )
            continue
        try:
            found.append(parse_template(path.stem, path.read_text(encoding="utf-8")))
        except OSError as exc:
            logger.warning(
                "logbook template %s: unreadable (%s); skipped", path.name, exc
            )
    found.sort(key=lambda t: (t.order, t.label.lower(), t.name))
    return found


class PageSeeds(BaseModel):
    """What a page needs to draw type buttons and label stored entries.

    ``buttons`` is the row for one book's composers; ``labels`` maps every
    loaded template's name to itself, so an entry that started from a
    template offered in the *other* book (or one since re-scoped) still
    shows its chip; ``prefill`` is the one JSON block the editor reads;
    ``quiet`` are the names rendered without a chip.
    """

    buttons: list[SeedTemplate] = Field(default_factory=list)
    labels: dict[str, SeedTemplate] = Field(default_factory=dict)
    prefill: dict[str, str] = Field(default_factory=dict)
    quiet: tuple[str, ...] = RESERVED_NAMES


class SeedTemplates:
    """The current template set, refreshed in the background when stale.

    Parameters
    ----------
    directory : Path, optional
        Where the ``*.md`` files live. ``None`` means no templates at all —
        the set is empty and nothing is ever read.

    The first load is synchronous, so the process starts with its buttons;
    after that a request that finds the set older than
    :data:`REFRESH_INTERVAL_S` kicks one daemon thread to re-read the
    directory and returns the set it has. The share is never on a page's
    critical path, which is the month page's whole promise.
    """

    def __init__(self, directory: Optional[Path]) -> None:
        self.directory = Path(directory) if directory else None
        self._templates: list[SeedTemplate] = []
        self._loaded_at = 0.0
        self._lock = threading.Lock()
        self._refreshing = False
        if self.directory is not None:
            self._refresh()

    def _refresh(self) -> None:
        try:
            fresh = load_templates(self.directory)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — keep the last set; the share may be down
            logger.exception("logbook templates: refresh of %s failed", self.directory)
        else:
            with self._lock:
                self._templates = fresh
        finally:
            with self._lock:
                self._loaded_at = time.monotonic()
                self._refreshing = False

    def current(self) -> list[SeedTemplate]:
        """Return the loaded set, refreshing in the background if stale."""
        if self.directory is None:
            return []
        with self._lock:
            stale = time.monotonic() - self._loaded_at >= REFRESH_INTERVAL_S
            kick = stale and not self._refreshing
            if kick:
                self._refreshing = True
            templates = list(self._templates)
        if kick:
            threading.Thread(
                target=self._refresh, name="logbook-templates", daemon=True
            ).start()
        return templates

    def for_book(self, book: str) -> list[SeedTemplate]:
        """Return the templates offered on ``book``'s composers."""
        return [t for t in self.current() if t.offered_in(book)]

    def by_name(self) -> dict[str, SeedTemplate]:
        """Return the loaded set keyed by name, for labelling stored entries."""
        return {t.name: t for t in self.current()}

    def for_page(self, book: str) -> PageSeeds:
        """Return what a page for ``book`` needs, from one read of the set."""
        current = self.current()
        return PageSeeds(
            buttons=[t for t in current if t.offered_in(book)],
            labels={t.name: t for t in current},
            prefill={t.name: t.body for t in current},
        )
