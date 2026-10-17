"""The logbook domain over the logbook's and the portal's HTTP APIs.

One verb, ``post_logbook_entry`` (DESIGN.md §4.6): the entry is created
on the logbook's own ``POST /api/entries`` as an ``agent_analysis``
**draft** in the scans book (the logbook refuses a non-note entry born
``kept``: a person keeps it), and each figure is a reference to a portal
artifact that the **portal** fetches off the share and attaches through
its ``POST /api/run/{uid}/logbook`` send — the upload, re-read and
append live there once.  No image bytes pass through this server, and
nothing here validates the entry: the logbook does.

Order matters for recovery.  Every figure reference is checked and its
run resolved *before* the entry is written, so a bad reference refuses
the whole call with nothing written.  Once the entry exists the call is
a success whatever happens to the figures: each one's outcome is
reported on its own, and the ``entry_id`` is never lost to a failed
attachment — posting again would make a second entry.
"""

from __future__ import annotations

from typing import Any, Optional

from geecs_mcp import clients, config, errors, guard, tool_names
from geecs_mcp.analysis.tools import IMAGE_SUFFIXES, resolve_uid
from geecs_mcp.clients import ServiceRefusal, ServiceUnavailable
from geecs_mcp.identity import InvalidArgument, identity_for
from geecs_mcp.server import mcp

#: The book an agent writes to (DESIGN.md §1) and what its entries are.
BOOK = "scans"
ENTRY_KIND = "agent_analysis"
ENTRY_STATUS = "draft"


def _figure_refs(figures: Any, day: str) -> list[dict[str, Any]]:
    """The figure references, checked: ``scan_number``, ``day``, ``path``, ``caption``."""
    if figures is None:
        return []
    if not isinstance(figures, (list, tuple)):
        raise InvalidArgument(
            "figures is a list of {scan_number, path, day?, caption?} references"
        )
    refs = []
    for i, figure in enumerate(figures):
        if not isinstance(figure, dict):
            raise InvalidArgument(
                f"figures[{i}] is not a {{scan_number, path, day?, caption?}} object"
            )
        path = str(figure.get("path") or "").strip()
        if not path:
            raise InvalidArgument(
                f"figures[{i}] names no path (an artifact path from get_scan_figure)"
            )
        if not path.lower().endswith(IMAGE_SUFFIXES):
            raise InvalidArgument(
                f"figures[{i}] {path!r} is not an image: the logbook attaches "
                "figures only — quote a sidecar's numbers in body_md instead "
                "(get_analysis_artifact)"
            )
        refs.append(
            {
                "scan_number": figure.get("scan_number"),
                # A figure is the entry's day unless it says otherwise —
                # never the MCP host's "today".
                "day": str(figure.get("day") or day),
                "path": path,
                "caption": str(figure.get("caption") or ""),
            }
        )
    return refs


async def _attach(entry_id: str, ref: dict[str, Any], author: str) -> dict[str, Any]:
    """One figure through the portal's send: the outcome row, never a raise."""
    row: dict[str, Any] = {"path": ref["path"], "scan_number": ref["scan_number"]}
    try:
        reply = await clients.call(
            "portal",
            "POST",
            f"/api/run/{ref['uid']}/logbook",
            params={"day": ref["day"]},
            json={
                "entry": entry_id,
                "path": ref["path"],
                "caption": ref["caption"],
                "author": author,
            },
        )
    except ServiceRefusal as exc:
        return {
            **row,
            "attached": False,
            "error_kind": exc.kind,
            "message": exc.message,
        }
    except ServiceUnavailable as exc:
        return {
            **row,
            "attached": False,
            "error_kind": "service_unreachable",
            "message": exc.message,
            "service": exc.service,
        }
    row.update(attached=True, link=reply.get("link"), appended=reply.get("appended"))
    if reply.get("entry_id") and reply["entry_id"] != entry_id:
        # The entry was deleted between the create and this send: the
        # portal's send made a fresh one rather than lose the figure.
        row["entry_id"] = reply["entry_id"]
    return row


async def _post_logbook_entry(
    day: str,
    body_md: str,
    operator: str,
    scan: Optional[int],
    figures: Any,
    payload: Optional[dict[str, Any]],
) -> str:
    author = identity_for(operator)
    day = str(day or "").strip()
    refs = _figure_refs(figures, day)

    # Resolve every figure's run before anything is written.
    uids: dict[tuple[Any, str], tuple[str, str]] = {}
    for ref in refs:
        key = (ref["scan_number"], ref["day"])
        if key not in uids:
            uid, iso, _ = await resolve_uid(*key)
            uids[key] = (uid, iso)
        ref["uid"], ref["day"] = uids[key]

    entry = await clients.call(
        "logbook",
        "POST",
        "/api/entries",
        json={
            "day": day,
            "book": BOOK,
            "scan": scan,
            "author": author,
            "body_md": str(body_md or ""),
            "template": "blank",
            "kind": ENTRY_KIND,
            "status": ENTRY_STATUS,
            "payload": payload,
        },
    )
    entry_id = str(entry["entry_id"])
    attachments = [await _attach(entry_id, ref, author) for ref in refs]
    failed = sum(1 for row in attachments if not row["attached"])
    extra: dict[str, Any] = {}
    if failed:
        extra["note"] = (
            f"the entry exists — do not post it again; {failed} of "
            f"{len(attachments)} figure(s) were not attached (see attachments). "
            "Tell the operator: the draft can be completed on the logbook page."
        )
    return errors.make_ok(
        entry_id=entry_id,
        url=f"{config.service_url('logbook')}/entry/{entry_id}",
        day=entry.get("day", day),
        scan=entry.get("scan", scan),
        kind=entry.get("kind"),
        status=entry.get("status"),
        author=author,
        attachments=attachments,
        complete=not failed,
        **extra,
    )


@mcp.tool(name=tool_names.POST_LOGBOOK_ENTRY)
async def post_logbook_entry(
    day: str,
    body_md: str,
    operator: str,
    scan: Optional[int] = None,
    figures: Optional[list[dict[str, Any]]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> str:
    """Write an entry in the scans logbook — a DRAFT a person keeps afterwards.

    ``day`` is ``YYYY-MM-DD``; ``scan`` files it on that scan's card (omit
    for a day-level entry).  ``body_md`` is Markdown: put the script behind
    a quoted number inline as a fenced block.  ``operator`` names the
    person supervising; the entry's author is
    ``agent, admin supervision (<operator>)``.

    ``figures``: ``[{scan_number, path, day?, caption?}]`` — ``path`` is
    the ``figure`` that ``get_scan_figure`` returns (images only); the
    portal attaches each one, ``day`` defaulting to the entry's.  Every
    reference is resolved before anything is written, so a bad one
    refuses the call with nothing posted.

    ``payload``: optional ``{kind: "analysis", analyzer, metrics, figures}``
    — the queryable numbers; the logbook validates it.

    Returns ``entry_id``, ``url`` and one ``attachments`` row per figure
    (``attached``, ``link`` or ``error_kind``/``message``).  Once the entry
    exists the result is ok even if a figure failed (``complete: false``):
    do NOT post again — that makes a second entry.
    """
    return await guard.run_async(
        _post_logbook_entry, day, body_md, operator, scan, figures, payload
    )
