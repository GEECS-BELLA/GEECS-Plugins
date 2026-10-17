"""Send a rendered plot into the scan's logbook entry."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from geecs_portal import analysis_runs, logbook_send
from geecs_portal.routes.common import _run_day
from geecs_portal.state import PortalState, get_state

router = APIRouter()


class PlotToLogbook(BaseModel):
    """One image on its way to the scan logbook, from one of two sources.

    ``image``: a rendered plot as a data URL rather than multipart form
    data — the page already holds one (``Plotly.toImage`` returns it)
    and form parsing would pull in ``python-multipart``, which this
    package does not otherwise need. Base64 costs a third in size, which
    a plot PNG can afford.

    ``path``: an artifact an analysis run produced, named as the run
    listing names it (relative to ``analysis/ScanNNN/``). The portal
    reads it off the share itself, under the artifact route's
    containment rule, so a client that has only a reference — the MCP
    server posting an agent's figures — never moves image bytes.
    Exactly one of the two is given.
    """

    #: The logbook requires a name on every entry and invents none.
    #: ``\S`` because a blank one survives ``min_length`` and is then the
    #: logbook's 422 — our malformed request arriving as the peer's fault.
    author: str = Field(min_length=1, max_length=120, pattern=r"\S")
    #: ``data:image/png;base64,…`` — the only data-URL form accepted.
    image: str = Field("", max_length=12_000_000)
    #: An analysis artifact's listing path (``UC_Crop/.../summary.png``);
    #: raster images only, the artifact route's inline set.
    path: str = Field("", max_length=1_000)
    #: Alt text: what the plot shows.
    caption: str = Field("", max_length=300)
    #: The portal URL that made it — the page state IS the analysis.
    source_url: str = Field("", max_length=4_000)
    #: Append to this entry when the page already made one for the scan.
    #: Anchored to the logbook's id alphabet (``uuid4().hex[:12]``): the
    #: value becomes a path segment in the URLs we build, and a ``/`` or a
    #: ``..`` in it would address some other route under that base.
    entry: str = Field("", max_length=64, pattern=r"^[0-9a-f]*$")

    @model_validator(mode="after")
    def _one_source(self) -> "PlotToLogbook":
        if bool(self.image) == bool(self.path):
            raise ValueError("give exactly one of 'image' (a data URL) or 'path'")
        return self


def _artifact_image(
    portal: PortalState, uid: str, day: str, relative: str
) -> tuple[bytes, str, str]:
    """(bytes, filename, content type) of one raster artifact, or 404/400.

    The same gate and containment as ``GET /run/{uid}/artifact``: the
    feature must be on, the path must resolve inside the scan's own
    analysis folder. Only the inline raster set is sent — the logbook
    stores images and PDFs, and a sidecar table is not a figure.
    """
    portal.analysis_available()
    _, _, analysis_folder, _ = portal.analysis_context(uid, day)
    file = analysis_runs.contained_artifact(analysis_folder, relative)
    if file is None:
        raise HTTPException(status_code=404, detail="no such artifact")
    if file.suffix.lower() not in analysis_runs.INLINE_IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"{relative!r} is not a raster image artifact",
        )
    content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
    try:
        data = file.read_bytes()
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"artifact unreadable: {exc}"
        ) from exc
    return data, file.name, content_type


@router.post("/api/run/{uid}/logbook", status_code=201)
def run_logbook_send(
    uid: str,
    payload: PlotToLogbook,
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Put one image — a rendered plot, or a run's artifact — into this scan's entry.

    The portal's **third** write verb (after the analysis runs and the
    config editor) and, like them, an explicit act: nothing is sent
    that a person (or, through the MCP server, an agent) did not ask
    for.  It writes to the logbook's own service through that service's
    public API, and touches neither the scans tree nor any portal state
    (an artifact source is READ off the analysis tree, nothing more).

    Ladder: no absolute ``--logbook-url`` / not this experiment / no
    scan number or day → 404 (there is no entry for it to join) · a
    malformed image → 400 · an artifact path that does not resolve →
    404, one that is not a raster image → 400 · the logbook unreachable
    → 503 · the logbook refusing → its own status for the verdicts that
    are about this payload (409/413/415), else 502.  The reply carries
    the attachment ``link`` the body now shows, so a caller appending
    more can cite it.
    """
    detail = portal.load_run(uid)
    # _run_day, not _resolved_folder: the folder is not wanted, and
    # resolving one stats the SMB share.  A logbook write touches the
    # scans mount not at all.
    run_day = _run_day(detail, day)
    if not portal.logbook_sendable(detail, run_day):
        raise HTTPException(
            status_code=404, detail="no logbook entry this scan could join"
        )
    scan = detail.summary.scan_number
    if payload.path:
        png, name, content_type = _artifact_image(portal, uid, day, payload.path)
        filename = f"scan{scan:03d}-{name}"
    else:
        try:
            png = logbook_send.decode_png_data_url(payload.image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        filename, content_type = f"scan{scan:03d}-plot.png", "image/png"
    try:
        result = logbook_send.send_plot(
            base_url=portal.logbook_send_base,
            day=run_day.isoformat(),
            scan=scan,
            author=payload.author.strip(),
            png=png,
            caption=payload.caption,
            source_url=payload.source_url,
            entry_id=payload.entry or None,
            filename=filename,
            content_type=content_type,
            client=portal.logbook_client,
        )
    except ValueError as exc:  # the image itself: too big, or empty
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except logbook_send.LogbookUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except logbook_send.LogbookRefused as exc:
        # 409/413/415 are verdicts on what we sent and mean the same
        # to our caller; anything else is the peer's own trouble, and
        # a peer's 500 must not read as ours.
        status = exc.status if exc.status in (409, 413, 415) else 502
        raise HTTPException(status_code=status, detail=exc.detail) from exc
    return JSONResponse(
        {
            "entry_id": result.entry_id,
            "appended": result.appended,
            "url": f"{portal.logbook_send_base}/entry/{result.entry_id}",
            "link": result.link,
        },
        status_code=201,
    )
