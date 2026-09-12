"""LogEntry — one piece of human (or agent) commentary about a scan.

The scan logbook splits into two halves with completely different
properties. The *record* — scan number, parameters, shot counts, status —
is derived from the scan folder on every request and stored nowhere. The
*commentary* is the other half: what people wrote. It exists nowhere else,
which makes it the only irreplaceable thing in the system, and this module
is its shape.

It lives in ``geecs_schemas`` for the same reason ``ScanRequest`` does: the
logbook writes entries, GEECS-MCP validates them at the tool boundary when
an agent posts one, and a reader of the markdown mirror parses them back —
three packages that must agree on one definition without depending on each
other. This package depends on pydantic alone, so all three can.

The body is opaque
------------------
``body_md`` is one markdown string. Nothing in this repository parses it
for *structure*: the only reads are the renderer (to draw it) and the tag
scan (``#laser`` → ``tags``, at save). Templates supply its *initial
text* and stop there; the first save is copy-on-write and the text is
then entirely the author's.

That is a deliberate rejection of the obvious alternative — storing an
entry as structured fields named by a template's headings. Under that
design, editing a template retroactively hides or orphans historical
content, and every "can we add a field" becomes a migration. Here a
template can change five times a year and no stored entry notices.

Machine-authored structure has a home that does not compromise this:
``payload``, a kind-discriminated union carrying whatever an analysis or a
tool wants to make queryable, *alongside* the prose rather than instead of
it. Human writing never goes there.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

#: Bump when a stored entry needs migrating, not when a field is added —
#: every field below is optional or defaulted, so additive changes read old
#: rows unchanged.
LOG_ENTRY_SCHEMA_VERSION = 1


class Attachment(SchemaModel):
    """One file uploaded with an entry — a pasted screenshot, a PDF.

    The bytes live on disk — in the logbook's attachment store beside its
    database, and copied beside the mirrored markdown — never in the
    database. This is the manifest: what was uploaded, so that "stored but
    no longer referenced by any body" is computable and orphans can be
    found. The body stays authoritative for *display*; this is
    authoritative for *what exists*.

    Attributes
    ----------
    id : str
        Opaque identifier of this upload. The files of one entry share a
        directory named by the *entry's* id; this distinguishes uploads
        within it, and survives a rename of the file.
    filename : str
        Name as stored, so the entry's relative link is derivable.
    content_type : str
        The media type it is served as.
    size_bytes : int
        Stored size.
    uploaded_at : datetime
        When it was stored.
    """

    id: str = Field(description="Opaque id of this upload.")
    filename: str = Field(description="Name as stored on disk.")
    content_type: str = Field(description="Media type, e.g. image/png.")
    size_bytes: int = Field(ge=0, description="Stored size in bytes.")
    uploaded_at: datetime = Field(description="When the file was stored.")


class AnalysisPayload(SchemaModel):
    """Structured results an analysis wants to make queryable.

    Rides alongside the prose so that "show me every scan where the charge
    rolloff onset moved" is a query rather than a reading exercise. The
    analysis still writes prose in ``body_md``; this is the part a machine
    reads back.
    """

    kind: Literal["analysis"] = Field("analysis", description="Payload type tag.")
    analyzer: str = Field(
        description="Which analyzer produced this, e.g. Array1DScanAnalyzer."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Named scalar results. Free-form by design: an analyzer names its own.",
    )
    figures: list[str] = Field(
        default_factory=list,
        description="Paths to figures in the analysis tree, referenced not copied.",
    )


class ProblemPayload(SchemaModel):
    """A problem worth finding again later."""

    kind: Literal["problem"] = Field("problem", description="Payload type tag.")
    severity: Literal["note", "degraded", "blocked"] = Field(
        "note", description="How much it stopped the run."
    )
    devices: list[str] = Field(
        default_factory=list, description="Devices implicated, by GEECS name."
    )
    issue_url: Optional[str] = Field(
        None, description="Tracker link, when one was filed."
    )


#: Extension point. A new entry type is one model added here — no change to
#: the envelope, no migration, and every existing entry stays valid because
#: ``payload`` is optional. Same pattern as the analyzer specs in
#: :mod:`geecs_schemas.analysis`.
EntryPayload = Annotated[
    Union[AnalysisPayload, ProblemPayload],
    Field(discriminator="kind"),
]

#: Who wrote an entry, and how much a reader should trust it. An agent's
#: draft is not a reviewed observation, and a system that lets an agent
#: read its own unreviewed output back as fact will cite its own guesses.
EntryKind = Literal["note", "agent_analysis", "agent_draft"]

#: Which book an entry belongs to. The **scans** book is the curated
#: campaign record — the day document, anchored to scans; the **ops** book
#: is routine operations — laser notes, maintenance, shift handovers — read
#: by month. Same store, same shape; the author chooses the book by which
#: page they write from, never the timestamp.
Book = Literal["scans", "ops"]

#: ``draft`` renders with a "nobody has reviewed this" banner. Only a human
#: promotes a draft to ``kept``; an agent cannot promote its own. The store
#: enforces the birth half of that rule — an ``agent_*`` entry cannot be
#: *created* as ``kept`` — rather than this model, because a promoted agent
#: entry is a valid stored state and must read back.
EntryStatus = Literal["kept", "draft"]


class LogEntry(VersionedSchemaModel):
    """One entry in the scan logbook.

    Attributes
    ----------
    schema_version : int
        Format revision; see :data:`LOG_ENTRY_SCHEMA_VERSION`.
    entry_id : str
        Stable identifier, also the attachment directory segment.
    day : str
        The run day this entry belongs to, as ``YYYY-MM-DD``.
    book : Book
        ``scans`` (the campaign record, may be anchored to a scan) or
        ``ops`` (operations; day-level only).
    scan : int or None
        The scan it annotates, when the entry is about one. Mutually
        exclusive with ``after``; an entry with neither is a **day-level**
        entry — a note about the day rather than a scan, which is what a
        general logbook is mostly made of.
    after : int or None
        For an interscan entry, the scan number it follows. ``0`` means
        before the first scan of the day.
    author : str
        Who wrote it. ``"osprey"`` for the agent.
    kind : EntryKind
        Human note, agent analysis, or agent draft.
    status : EntryStatus
        ``draft`` until a human keeps it.
    tags : list of str
        Words the author marked with ``#`` in the body (``#laser``),
        parsed at save and kept here so a month can be filtered without
        reading every body. The body stays the truth: editing the tag out
        drops it from here on the next save. Templates seed the common
        ones, so a type button and a typed tag are the same thing.
    template : str
        Which seed template it started from. Metadata about provenance,
        never structure: the body may have diverged completely, and the
        entry keeps its template name because that is what the author
        reached for.
    body_md : str
        The entry itself. Opaque markdown; only rendered and tag-scanned.
    payload : EntryPayload or None
        Optional machine-authored structure. Never human prose.
    attachments : list of Attachment
        Files uploaded with this entry.
    created_at : datetime
        When it was first saved.
    edited_at : datetime or None
        When its *text* was last changed, if ever. What a reader is shown.
    edited_by : str or None
        Who last changed the text. ``author`` never changes — it is who
        wrote the entry, and part of the mirror file's stable name — so a
        colleague's typo fix is recorded here rather than by taking the
        entry over.
    updated_at : datetime
        When *anything* about it last changed — text, status, attachments,
        deletion. What a synchroniser asks for: "everything since" is a
        query on this, and a promotion or an upload that left ``edited_at``
        alone would otherwise be invisible to it.
    deleted_at : datetime or None
        Set instead of removing the row. A deleted entry is hidden from
        every listing, but the fact that it existed and was deleted is
        kept: a mirror or a downstream copy can only learn about a
        deletion it is able to see, and an accidental delete is then a
        field to clear rather than a loss.
    version : int
        Optimistic-lock counter, incremented on every save. A writer sends
        the version it read; a mismatch means someone else saved first.
    """

    schema_version: int = Field(
        LOG_ENTRY_SCHEMA_VERSION, description="Entry format revision."
    )

    entry_id: str = Field(description="Stable id; also the attachment directory.")
    day: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$", description="Run day, as YYYY-MM-DD."
    )
    book: Book = Field("scans", description="Campaign record or operations.")
    scan: Optional[int] = Field(
        None, ge=1, description="Scan annotated; none for a day-level entry."
    )
    after: Optional[int] = Field(
        None, ge=0, description="For an interscan entry, the scan it follows."
    )

    author: str = Field(min_length=1, description="Who wrote it.")
    kind: EntryKind = Field("note", description="Human note, or which agent output.")
    status: EntryStatus = Field("kept", description="Drafts await a human.")
    tags: list[str] = Field(
        default_factory=list, description="#tags parsed from the body at save."
    )
    template: str = Field("blank", description="Seed template it started from.")

    body_md: str = Field("", description="The entry. Opaque markdown.")
    payload: Optional[EntryPayload] = Field(
        None, description="Machine-authored structure, never human prose."
    )
    attachments: list[Attachment] = Field(
        default_factory=list, description="Files uploaded with the entry."
    )

    created_at: datetime = Field(description="First saved.")
    edited_at: Optional[datetime] = Field(None, description="Text last changed.")
    edited_by: Optional[str] = Field(None, description="Who last changed the text.")
    updated_at: datetime = Field(description="Anything last changed.")
    deleted_at: Optional[datetime] = Field(None, description="Tombstone.")
    version: int = Field(1, ge=1, description="Optimistic-lock counter.")

    @model_validator(mode="after")
    def _one_anchor_at_most(self) -> "LogEntry":
        if self.scan is not None and self.after is not None:
            raise ValueError("an entry is anchored to a scan or after one, not both")
        if self.book == "ops" and (self.scan is not None or self.after is not None):
            raise ValueError("an ops entry is about the day, not a scan")
        return self

    @property
    def is_interscan(self) -> bool:
        """Whether this entry sits between two scans rather than on one."""
        return self.after is not None

    @property
    def is_day_level(self) -> bool:
        """Whether this entry is about the day rather than any scan."""
        return self.scan is None and self.after is None

    @property
    def is_deleted(self) -> bool:
        """Whether this entry has been tombstoned."""
        return self.deleted_at is not None

    @property
    def anchor(self) -> str:
        """Return a stable, sortable key for where this entry belongs.

        Used to group entries onto the day document without the view
        needing to know the ``scan``/``after`` encoding: ``day`` for a
        day-level entry, ``scan-NNNN`` and ``after-NNNN`` otherwise.
        """
        if self.after is not None:
            return f"after-{self.after:04d}"
        if self.scan is None:
            return "day"
        return f"scan-{self.scan:04d}"
