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


def _plain(token) -> str:
    """The readable text of one inline token.

    An image's ``content`` IS its alt, and a line break inside a block is a
    child with no content — so both are handled here rather than by joining
    on content alone, which glued a wrapped callout's two lines together.
    """
    return "".join(
        " " if child.type in {"softbreak", "hardbreak"} else child.content
        for child in (token.children or [])
        if child.type in {"text", "code_inline", "softbreak", "hardbreak", "image"}
    ).strip()


def summarize(body_md: str, limit: int = 120) -> str:
    """A one-line stand-in for an entry, for when it is collapsed.

    Every entry in every book is collapsible, which only works if the shut
    state says something worth reading — otherwise a closed note is an
    author and a timestamp, and the reader opens all of them to find one.

    **Derived, never asked for.** A title field would make the writer name
    a thing before they could type it, and would be empty for every entry
    already written — the ceremony this package refuses elsewhere (a day is
    a query, tags come out of the body, nobody declares anything). Derived
    but steerable: start with a markdown heading and it becomes the summary.

    It reads the **token stream**, not the raw text. A first version lived
    in ``geecs_schemas`` and stripped ``` `*_~ ``` with a regex, which
    turned ``~20 mJ, jitter ~3%`` into ``20 mJ, jitter 3%`` and
    ``3*10^18 W/cm2`` into ``310^18`` — a single ``~`` is not markdown at
    all and a single ``*`` is not emphasis, but both are ordinary lab
    notation. Asking the parser that already renders the body avoids
    guessing: emphasis that *is* emphasis loses its markers, and arithmetic
    keeps its characters.

    Fenced code is skipped (a summary reading ``import os`` is worse than
    none) and a callout keeps its text but loses its ``[!NOTE]`` marker,
    which the chip beside it already shows.
    """
    in_table_cell = False
    first_cell = ""
    for token in _md.parse(body_md or ""):
        # A table cell's contents is an ordinary inline token, so the first
        # one wins and a summary reads "Parameter" or "Date" — the header of
        # the toolbar's own table skeleton, or of a pasted spreadsheet.
        # Content-free, and worse than empty because it looks like a summary.
        if token.type in {"th_open", "td_open"}:
            in_table_cell = True
        elif token.type in {"th_close", "td_close"}:
            in_table_cell = False
        if token.type != "inline" or not token.content.strip():
            continue
        if in_table_cell:
            # Remembered, not discarded. The toolbar's table button and the
            # spreadsheet paste both leave the cursor ONE newline below the
            # block, and markdown-it absorbs a sentence typed there as another
            # row — so skipping cells outright summarised such a note as
            # nothing at all. Prose still wins; a table-only note gets its
            # first cell, which is thin but true.
            if not first_cell:
                first_cell = _plain(token)
            continue
        # A line break inside one block is a child token carrying no content,
        # so joining on content alone glues the lines together — a wrapped
        # callout came out as "Jet pressure driftingchecked it".
        text = re.sub(r"^\[!\w+\]\s*", "", _plain(token))  # callout flavour marker
        if text:
            return text[: limit - 1] + "\u2026" if len(text) > limit else text
    return first_cell
