"""Markdown to HTML for the logbook — one renderer, server-side, sanitized.

One renderer on purpose. The web view, the export and anything else that
shows an entry all go through this function, so they cannot disagree. A
client-side renderer for live preview would be a second implementation that
drifts; if live preview is wanted, it is a debounced call to the endpoint
that wraps this, not a port of it.

Three things happen here, in this order:

1. **CommonMark + GFM extras** via ``markdown-it-py``. Tables, strikethrough
   and task lists are enabled because people paste tables from spreadsheets
   and tick things off. Raw HTML in the source is *escaped*, never passed
   through — an agent writes into the same field a human does.
2. **Sanitize** with ``nh3``. Belt and braces over (1): even if the parser
   grew a hole, nothing script-shaped reaches the page.
3. **Rewrite attachment links.** The stored body says
   ``![trace](attachments/e7f2/jet.png)`` — relative, so the markdown file
   on the share resolves offline. The page needs the serving route, so the
   prefix is swapped at render time and never written back.

Callouts (``> [!WARNING]``) are recognised after sanitizing: the marker is
plain text that survives (2), so no allowlist has to be widened to carry a
class through. GitHub renders the same syntax, so the markdown mirror looks
right there too.
"""

from __future__ import annotations

import html
import re

import nh3
from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin

#: The four callout flavours, matching GitHub's.
CALLOUTS = ("NOTE", "TIP", "WARNING", "CAUTION")

#: nh3's defaults plus the task-list checkbox, which is the one form
#: element the markdown can produce. Read-only on the page (``disabled``
#: is what the plugin emits); nothing else from ``input`` is admitted.
_TAGS = nh3.ALLOWED_TAGS | {"input"}
_ATTRIBUTES = {**nh3.ALLOWED_ATTRIBUTES, "input": {"type", "checked", "disabled"}}

_md = (
    MarkdownIt("commonmark", {"html": False, "breaks": False, "typographer": False})
    .enable(["table", "strikethrough"])
    .use(tasklists_plugin)
)

#: ``> [!WARNING]`` renders as a blockquote whose first paragraph starts
#: with the marker. Matched on sanitized output; non-greedy to the closing
#: tag, so a nested blockquote inside a callout is not supported.
_CALLOUT = re.compile(
    r"<blockquote>\s*<p>\[!(NOTE|TIP|WARNING|CAUTION)\]\s*(?:<br\s*/?>)?\s*(.*?)</blockquote>",
    re.S,
)

#: Attachment references as the store writes them.
_ATTACHMENT_REF = re.compile(r'(src|href)="attachments/')

#: Two or more images in a row, each its own paragraph — what a run of
#: pasted screenshots or published figures renders as. They become one
#: grid, the plot-table LogMaker's ``gdoc_slot`` numbering used to fake:
#: no slot assignment, no ceiling at four, no collisions.
_IMAGE_RUN = re.compile(r"(?:<p>\s*<img\b[^>]*>\s*</p>\s*){2,}")
_IMAGE = re.compile(r"<img\b[^>]*>")


def render_markdown(body_md: str, *, attachment_base: str | None = None) -> str:
    """Render an entry body to safe HTML.

    Parameters
    ----------
    body_md : str
        The stored body, verbatim.
    attachment_base : str, optional
        The attachment serving route, e.g. ``/log/attachments``; a
        relative ``attachments/<entry_id>/<file>`` reference becomes
        ``/log/attachments/<entry_id>/<file>``. When absent
        they are left relative, which is what an offline export wants.

    Returns
    -------
    str
        Sanitized HTML.
    """
    rendered = _md.render(body_md or "")
    clean = nh3.clean(
        rendered, tags=_TAGS, attributes=_ATTRIBUTES, link_rel="noopener noreferrer"
    )
    clean = _callouts(clean)
    clean = _image_grids(clean)
    if attachment_base:
        base = attachment_base.rstrip("/")
        clean = _ATTACHMENT_REF.sub(rf'\1="{html.escape(base)}/', clean)
    return clean


def _image_grids(fragment: str) -> str:
    """Gather consecutive image paragraphs into a ``figgrid``."""

    def swap(match: re.Match[str]) -> str:
        imgs = "".join(
            f"<figure>{img}</figure>" for img in _IMAGE.findall(match.group(0))
        )
        return f'<div class="figgrid">{imgs}</div>'

    return _IMAGE_RUN.sub(swap, fragment)


def _callouts(fragment: str) -> str:
    """Turn marker-led blockquotes into labelled callout boxes."""

    def swap(match: re.Match[str]) -> str:
        flavour = match.group(1).lower()
        body = match.group(2).strip()
        if not body.startswith("<"):
            body = "<p>" + body
        return (
            f'<div class="callout callout-{flavour}">'
            f'<div class="callout-label">{flavour}</div>{body}</div>'
        )

    return _CALLOUT.sub(swap, fragment)
