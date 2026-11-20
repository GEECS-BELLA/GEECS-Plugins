"""Send a rendered plot to the scan logbook, over HTTP like any client.

The portal is a *view* of a scan and the logbook is a peer service, so a
plot travels between them the way anything else does: an HTTP call to the
logbook's public write verbs. Nothing here imports GeecsLogbook (the
package edge in the root ``CLAUDE.md`` stands) — this module knows three
URLs and the shapes they take.

Why the portal sends and not the browser
----------------------------------------
Clipboard *image* writes are a secure-context privilege, so on a
plain-HTTP deployment the Plot tab cannot put a PNG on the clipboard at
all — the button downloads instead (see ``DEPLOYMENT.md``). This path
does not care about the page's origin: the browser hands the bytes to the
portal, and the portal talks to the logbook server-to-server. That also
keeps the logbook free of CORS headers, since no cross-origin request is
ever made.

The four calls
--------------
The logbook stores an attachment *against an entry*, so there is nothing
to upload until an entry exists (``POST /api/entries/{id}/attachments``
404s otherwise — the composer solves the same problem with its
``ensureEntry``). One send is therefore:

1. ``POST /api/entries`` — create, on this day and scan, in the scans
   book.  Skipped when appending to an entry that already exists.
2. ``POST /api/entries/{id}/attachments`` — the PNG; the reply carries
   the ``attachments/<entry_id>/<file>`` link the body must use, because
   the store claims the free filename and may not get the one we asked
   for.
3. ``GET /api/entries/{id}`` — the current body and version.  The upload
   in (2) bumped the version, so the value read before it is stale.
4. ``PATCH /api/entries/{id}`` — the body with the image appended,
   against that version.

The settings a plot was drawn with
----------------------------------
A plot sent from the Plot tab comes with the settings set it was drawn
with, as the page read it at send time. It is stored **in the entry**, as
a second attachment (``<image name>-settings.json``, the
:func:`frozen_settings` document), so the entry carries the settings
themselves and not the name of a set that changes with its owner's next
pick. One more upload between (2) and (3), and one line of the body: the
``Plot settings as sent:`` paragraph at the top of the entry, which links
each plot's document (above the images, like the portal link, so it never
splits their run). A logbook too old to take JSON answers 415: the image
is still sent, and the result says the settings were not kept.

Why append rather than replace
------------------------------
A second plot sent **from the same browser** joins the first *in one
entry*: the logbook's renderer turns two or more images in a row into a
figure grid (``geecs_logbook.render``), which is the layout that used to
be faked by LogMaker's ``gdoc_slot`` numbering. So the body grows by one
image paragraph per send, and the portal link stays on the first line
where it does not break the run of images.

"From the same browser", not "for the same scan": the entry to append to
is whatever the caller passes as ``entry_id``, which the page remembers
in ``localStorage``. Nothing here asks the logbook which entry a scan
already has — see this package's ``CLAUDE.md`` for why that was rejected
rather than overlooked. A 409 means someone was editing the
entry between (3) and (4); the append is re-read and retried once, which
is safe precisely because it is an append and never a replace.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

#: Kept under the logbook's own 20 MiB attachment cap so an oversized
#: plot is refused here, with our error message, rather than after an
#: upload that crosses the network to be rejected.
MAX_PLOT_BYTES = 8 * 1024 * 1024

#: Server-to-server on the same host in every deployment we run; long
#: enough for a mirror write to the share, short enough that a wedged
#: logbook does not hold a portal worker thread.
TIMEOUT_SECONDS = 15.0

#: A settings set is a few kilobytes; this is room for one a hundred times
#: that, and a refusal for anything that is not a settings set.
MAX_SETTINGS_BYTES = 256 * 1024

#: The body paragraph that links each plot's settings document.
SETTINGS_LINE = "Plot settings as sent:"


class LogbookUnreachable(RuntimeError):
    """The logbook did not answer at all — down, or the URL is wrong."""


class LogbookRefused(RuntimeError):
    """The logbook answered, and said no.

    Not a dataclass on purpose: a dataclass exception never runs
    ``Exception.__init__``, so ``args`` comes out empty and the class
    becomes unhashable.

    Attributes
    ----------
    status : int
        The status the logbook returned.
    detail : str
        Its own explanation, passed through rather than reworded — it
        knows why (an unsupported type, an entry that vanished) and the
        portal does not.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"logbook returned {status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class SendResult:
    """What a completed send produced, for the page to act on."""

    entry_id: str
    #: True when the image joined an entry that already existed.
    appended: bool
    #: The ``attachments/<entry_id>/<file>`` reference now in the body.
    link: str
    #: The settings document's reference, when settings were given and the
    #: logbook kept them; ``None`` otherwise.
    settings_link: Optional[str] = None


def _detail_of(response: httpx.Response) -> str:
    """The logbook's ``detail`` string, or the bare status line."""
    try:
        body = response.json()
    except ValueError:
        return response.reason_phrase or str(response.status_code)
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):  # the logbook's richer errors
        detail = detail.get("message") or str(detail)
    return str(detail or response.reason_phrase or response.status_code)


def _checked(response: httpx.Response) -> dict:
    """The decoded body of a 2xx reply, or :class:`LogbookRefused`.

    A success that is not JSON is the peer's trouble, not the caller's:
    ``--logbook-url`` pointing at a front door or the wrong service gets
    an HTML maintenance page with a 200 on it, and letting
    ``json.JSONDecodeError`` (a ``ValueError``) escape would surface that
    to the user as "the image is malformed".
    """
    if response.status_code >= 400:
        raise LogbookRefused(response.status_code, _detail_of(response))
    try:
        body = response.json()
    except ValueError as exc:
        raise LogbookRefused(
            502, f"the logbook answered {response.status_code} but not JSON"
        ) from exc
    if not isinstance(body, dict):
        raise LogbookRefused(502, "the logbook answered with an unexpected shape")
    return body


def _field(body: dict, key: str) -> object:
    """One expected key of a reply, or :class:`LogbookRefused`.

    Portal and logbook are separate services on separate release
    cadences, and the fleet has run them from different branches. A
    reply missing a key it used to carry is version skew — a 502, not a
    ``KeyError`` traceback out of a portal request thread.
    """
    if key not in body:
        raise LogbookRefused(502, f"the logbook reply carries no {key!r}")
    return body[key]


#: What the Plot tab hands over: ``Plotly.toImage`` returns exactly this
#: shape, so the page forwards it rather than re-encoding.
_PNG_DATA_URL = "data:image/png;base64,"


def decode_png_data_url(image: str) -> bytes:
    """The PNG bytes inside a ``data:image/png;base64,…`` URL.

    Only PNG is accepted, and only base64 — the logbook's attachment
    store keys its extension off the content type, so a JPEG smuggled
    under a PNG prefix would be stored under the wrong name. Anything
    else is the caller's error, not a server fault.

    Raises
    ------
    ValueError
        The prefix is missing or the payload is not valid base64.
    """
    if not image.startswith(_PNG_DATA_URL):
        raise ValueError("expected a data:image/png;base64 URL")
    try:
        return base64.b64decode(image[len(_PNG_DATA_URL) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"the image is not valid base64: {exc}") from exc


def image_markdown(caption: str, link: str) -> str:
    """One image paragraph, as the composer would have written it.

    ``caption`` becomes the alt text, so a note read without images (the
    markdown mirror on the share, a screen reader) still says which plot
    this was.
    """
    safe = caption.replace("[", "(").replace("]", ")").strip() or "plot"
    return f"![{safe}]({link})"


def frozen_settings(
    settings: dict, *, name: str, image: str, day: str, scan: int, source_url: str
) -> bytes:
    """The document stored beside a plot: the settings it was drawn with.

    ``settings`` is the set as it was at send time, whole, so the plot can
    be drawn again after the set itself has changed; ``set`` is only the
    name it had. ``image`` is the plot's own attachment link, which is what
    pairs the two. The plotter's ``PlotterSettings`` is the shape of
    ``settings``, but nothing here reads it: the document is kept as sent.
    """
    document = {
        "kind": "plot_settings",
        "image": image,
        "day": day,
        "scan": scan,
        "set": name,
        "source_url": source_url,
        "settings": settings,
    }
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()


def with_settings_link(body_md: str, label: str, link: str) -> str:
    """``body_md`` with one more link on its settings line.

    The line is a paragraph of its own at the top of the entry, made on
    the first plot that has settings: above the images, where it cannot
    split the run the renderer grids. Later plots extend it in place.
    """
    item = f"[{_plain(label)}]({link})"
    lines = body_md.split("\n")
    for index, line in enumerate(lines):
        if line.startswith(SETTINGS_LINE):
            lines[index] = f"{line.rstrip()} · {item}"
            return "\n".join(lines)
    rest = body_md.lstrip("\n")
    head = f"{SETTINGS_LINE} {item}\n"
    return f"{head}\n{rest}" if rest.strip() else head


def _plain(text: str) -> str:
    """Link text that cannot close its own brackets."""
    return text.replace("[", "(").replace("]", ")")


def appended_body(body_md: str, block: str) -> str:
    """``body_md`` with ``block`` as a new trailing paragraph.

    A blank line between paragraphs is what makes consecutive images a
    *run* the renderer can grid, so the separator is not cosmetic.
    """
    existing = body_md.rstrip()
    return f"{existing}\n\n{block}\n" if existing else f"{block}\n"


def send_plot(
    *,
    base_url: str,
    day: str,
    scan: int,
    author: str,
    png: bytes,
    caption: str,
    source_url: str = "",
    entry_id: Optional[str] = None,
    filename: str = "plot.png",
    client: Optional[httpx.Client] = None,
    content_type: str = "image/png",
    entry_required: bool = False,
    settings: Optional[dict] = None,
    settings_name: str = "",
) -> SendResult:
    """Put one PNG into the scans book, on ``day`` against ``scan``.

    Parameters
    ----------
    base_url : str
        The logbook's absolute base URL, e.g. ``http://host:8400``.
    day, scan : str, int
        Where the entry belongs: an ISO date and the scan number. The
        logbook files the entry against that scan's card.
    author : str
        Who is sending. The logbook requires a name on every entry and
        does not invent one.
    png : bytes
        The rendered image.
    caption : str
        Alt text — what the plot shows.
    source_url : str, optional
        The portal URL that produced it. Written once, on the line above
        the images, so re-plotting is a click; the portal's page state
        lives in its URL, so this link restores the analysis, not just
        the scan. Ignored when appending (the line is already there).
    entry_id : str, optional
        Append to this entry instead of creating one. An entry that has
        since been deleted is not an error — a fresh one is created, and
        the result says so — unless ``entry_required``.
    entry_required : bool, optional
        The caller made ``entry_id`` itself and wants the image there or
        nowhere: a missing entry is :class:`LogbookRefused` (404), never a
        fresh one. A fresh one is a ``note`` born ``kept`` — wrong for an
        agent's draft, and one per figure when the caller's logbook is not
        this one.
    filename : str, optional
        The name offered to the store, which may claim a different one.
    client : httpx.Client, optional
        Injected in tests; a short-lived client is made otherwise.
    settings : dict, optional
        The settings set the plot was drawn with, as it was at send time.
        Stored in the entry beside the image (:func:`frozen_settings`).
    settings_name : str, optional
        The name that set had; recorded, never dereferenced.

    Returns
    -------
    SendResult

    Raises
    ------
    ValueError
        The image is empty or over :data:`MAX_PLOT_BYTES`, or the settings
        are not plain JSON or over :data:`MAX_SETTINGS_BYTES`.
    LogbookUnreachable
        No answer from the logbook.
    LogbookRefused
        An answer that was not a success.
    """
    if not png:
        raise ValueError("nothing to send: the rendered plot is empty")
    if len(png) > MAX_PLOT_BYTES:
        raise ValueError(
            f"plot is {len(png) // 1024} KiB, over the "
            f"{MAX_PLOT_BYTES // (1024 * 1024)} MiB send cap"
        )

    if settings is not None:
        try:
            size = len(json.dumps(settings, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"the settings are not plain JSON: {exc}") from exc
        if size > MAX_SETTINGS_BYTES:
            raise ValueError(
                f"the settings are {size // 1024} KiB, over the "
                f"{MAX_SETTINGS_BYTES // 1024} KiB cap"
            )

    base = base_url.rstrip("/")
    owned = client is None
    http = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        return _send(
            http,
            base=base,
            day=day,
            scan=scan,
            author=author,
            png=png,
            caption=caption,
            source_url=source_url,
            entry_id=entry_id,
            filename=filename,
            content_type=content_type,
            entry_required=entry_required,
            settings=settings,
            settings_name=settings_name,
        )
    except httpx.HTTPError as exc:
        raise LogbookUnreachable(f"no answer from the logbook at {base}") from exc
    finally:
        if owned:
            http.close()


def _create_entry(
    http: httpx.Client, *, base: str, day: str, scan: int, author: str, opening: str
) -> str:
    """Make the entry this scan's plots will live in; return its id."""
    created = _checked(
        http.post(
            f"{base}/api/entries",
            json={
                "day": day,
                "book": "scans",
                "scan": scan,
                "author": author,
                "body_md": opening,
                "template": "blank",
            },
        )
    )
    return str(_field(created, "entry_id"))


def _send(
    http: httpx.Client,
    *,
    base: str,
    day: str,
    scan: int,
    author: str,
    png: bytes,
    caption: str,
    source_url: str,
    entry_id: Optional[str],
    filename: str,
    content_type: str = "image/png",
    entry_required: bool = False,
    settings: Optional[dict] = None,
    settings_name: str = "",
) -> SendResult:
    """The four calls, with the create skipped when appending."""
    # The link goes in at creation, above the images: a paragraph
    # *between* two images would split the run the renderer grids.
    opening = f"[Plotted in the data portal]({source_url})\n" if source_url else ""

    appended = entry_id is not None
    if entry_id is None:
        entry_id = _create_entry(
            http, base=base, day=day, scan=scan, author=author, opening=opening
        )

    try:
        uploaded = _checked(
            http.post(
                f"{base}/api/entries/{entry_id}/attachments",
                files={"file": (filename, png, content_type)},
            )
        )
    except LogbookRefused as exc:
        if not (appended and exc.status == 404) or entry_required:
            raise
        # The entry the browser remembered is gone (discarded, deleted).
        # That is a stale pointer, not a failure to send: start a new one.
        logger.info("logbook entry %s is gone; creating a fresh one", entry_id)
        return _send(
            http,
            base=base,
            day=day,
            scan=scan,
            author=author,
            png=png,
            caption=caption,
            source_url=source_url,
            entry_id=None,
            filename=filename,
            content_type=content_type,
            settings=settings,
            settings_name=settings_name,
        )

    link = str(_field(uploaded, "link"))
    settings_link = None
    if settings is not None:
        settings_link = _upload_settings(
            http,
            base=base,
            entry_id=entry_id,
            link=link,
            document=frozen_settings(
                settings,
                name=settings_name,
                image=link,
                day=day,
                scan=scan,
                source_url=source_url,
            ),
        )
    _append_image(
        http,
        base=base,
        entry_id=entry_id,
        author=author,
        block=image_markdown(caption, link),
        settings=(PurePosixPath(link).stem, settings_link) if settings_link else None,
    )
    return SendResult(
        entry_id=entry_id, appended=appended, link=link, settings_link=settings_link
    )


def _upload_settings(
    http: httpx.Client, *, base: str, entry_id: str, link: str, document: bytes
) -> Optional[str]:
    """Store the settings document beside its image; its link, or ``None``.

    Named after the image the store actually kept, so the pair reads as
    one in a listing. A 415 is a logbook that does not take JSON yet (one
    deployed before this): the image is already stored and is still sent,
    without its settings, and the caller is told so. Any other refusal is
    the send's.
    """
    name = f"{PurePosixPath(link).stem}-settings.json"
    try:
        uploaded = _checked(
            http.post(
                f"{base}/api/entries/{entry_id}/attachments",
                files={"file": (name, document, "application/json")},
            )
        )
    except LogbookRefused as exc:
        if exc.status != 415:
            raise
        logger.warning(
            "the logbook at %s does not store plot settings (415); "
            "the plot is sent without them",
            base,
        )
        return None
    return str(_field(uploaded, "link"))


def _append_image(
    http: httpx.Client,
    *,
    base: str,
    entry_id: str,
    author: str,
    block: str,
    settings: Optional[tuple[str, str]] = None,
) -> None:
    """Read the body, add the image paragraph, write it back.

    With *settings* (a label and the settings document's link), the same
    write adds that link to the body's settings line.

    Retried once on a conflict: between the read and the write someone
    may have saved the entry from the logbook, and the right answer is to
    append to *their* text rather than to the copy we read. Only an
    append is safe to retry this way — a replace would eat their edit.
    """
    for attempt in (1, 2):
        current = _checked(http.get(f"{base}/api/entries/{entry_id}"))
        # `_field`, never `.get(…, "")`: this is the read half of a
        # read-modify-write, and a default would turn "the body did not
        # come back" into "the body is empty" — which the PATCH below
        # then makes true, erasing whatever someone had written.  An
        # entry legitimately HAS an empty body (one created with no
        # source_url), and that arrives as a present key, so nothing is
        # lost by demanding the key.  Missing or not a string is skew,
        # and skew must stop the write rather than complete it.
        body_md = _field(current, "body_md")
        if not isinstance(body_md, str):
            raise LogbookRefused(
                502, "the logbook reply's 'body_md' is not text; refusing to overwrite"
            )
        if settings is not None:
            body_md = with_settings_link(body_md, *settings)
        patch = http.patch(
            f"{base}/api/entries/{entry_id}",
            json={
                "body_md": appended_body(body_md, block),
                "editor": author,
                "expected_version": _field(current, "version"),
            },
        )
        if patch.status_code != 409:
            _checked(patch)
            return
        logger.info(
            "logbook entry %s moved under us (attempt %d); re-reading",
            entry_id,
            attempt,
        )
    raise LogbookRefused(409, "the entry is being edited — try again")
