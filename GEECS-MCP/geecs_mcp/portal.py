"""Naming a scan to the Data Portal — shared by every domain that cites one.

The loop knows a scan by number and day; the portal's routes take a run
uid.  :func:`resolve_uid` turns one into the other through the portal's
``jump`` route and refuses a number the day does not hold (the route
otherwise answers the day's newest run, which must never be analysed or
cited by mistake).  The figure helpers name a run's images the way the
portal's analysis listing does, so ``get_scan_figure`` and
``post_logbook_entry`` agree on what a figure reference is.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional
from urllib.parse import urlencode

from geecs_mcp import clients, config
from geecs_mcp.clients import ServiceRefusal
from geecs_mcp.identity import InvalidArgument

#: The MCP's own routing rule between its two readers: a raster suffix is
#: a figure (get_scan_figure), anything else an artifact — the same set the
#: portal renders inline.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def day_or_today(day: Optional[str]) -> str:
    """The ISO day given, or this host's today."""
    text = str(day or "").strip()
    return text or date.today().isoformat()


async def resolve_uid(scan_number: Any, day: Optional[str]) -> tuple[str, str, int]:
    """``(uid, day, scan_number)`` through the portal's jump route, or a refusal."""
    try:
        number = int(scan_number)
    except (TypeError, ValueError):
        raise InvalidArgument(
            f"scan_number {scan_number!r} is not an integer"
        ) from None
    if number < 1:
        raise InvalidArgument("scan_number must be positive")
    iso = day_or_today(day)
    jump = await clients.call(
        "portal", "GET", f"/api/run/jump/{iso}", params={"prefer": number}
    )
    if not jump.get("matched") or not jump.get("uid"):
        raise ServiceRefusal(
            "not_found",
            f"no scan {number} on {iso} in the portal's catalog "
            f"({jump.get('runs', 0)} run(s) that day)",
            day=iso,
            scan_number=number,
        )
    return str(jump["uid"]), iso, number


async def analysis_listing(uid: str, day: str) -> dict:
    """The portal's analysis picture of one run (its Analysis tab's listing)."""
    return (
        await clients.call(
            "portal", "GET", f"/api/run/{uid}/analysis", params={"day": day}
        )
        or {}
    )


def artifact_url(uid: str, path: str, day: str) -> str:
    """The portal's own URL for one of a run's artifacts."""
    base = config.service_url("portal") or ""
    return f"{base}/run/{uid}/artifact?{urlencode({'path': path, 'day': day})}"


def is_image(path: str) -> bool:
    """Whether *path* is a figure by the suffix rule above."""
    return str(path).lower().endswith(IMAGE_SUFFIXES)


def figure_entries(listing: dict, uid: str, day: str) -> list[dict]:
    """Every image artifact in a listing: ``figure`` (its path), analyzer, kind, bin, url."""
    entries = []
    for analyzer in listing.get("analyzers") or []:
        for artifact in analyzer.get("artifacts") or []:
            path = artifact.get("path")
            if not isinstance(path, str) or not is_image(path):
                continue
            entries.append(
                {
                    "figure": path,
                    "analyzer": analyzer.get("id"),
                    "kind": artifact.get("kind"),
                    "bin": artifact.get("bin"),
                    "figure_url": artifact_url(uid, path, day),
                }
            )
    return entries
