"""Send a rendered plot into the scan's logbook entry."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from geecs_portal import logbook_send
from geecs_portal.routes.common import _run_day
from geecs_portal.state import PortalState, get_state

router = APIRouter()


class PlotToLogbook(BaseModel):
    """One rendered plot on its way to the scan logbook.

    The image arrives as a data URL rather than as multipart form data:
    the page already holds one (``Plotly.toImage`` returns it) and form
    parsing would pull in ``python-multipart``, which this package does
    not otherwise need. Base64 costs a third in size, which a plot PNG
    can afford.
    """

    #: The logbook requires a name on every entry and invents none.
    #: ``\S`` because a blank one survives ``min_length`` and is then the
    #: logbook's 422 — our malformed request arriving as the peer's fault.
    author: str = Field(min_length=1, max_length=120, pattern=r"\S")
    #: ``data:image/png;base64,…`` — the only form accepted.
    image: str = Field(max_length=12_000_000)
    #: Alt text: what the plot shows.
    caption: str = Field("", max_length=300)
    #: The portal URL that made it — the page state IS the analysis.
    source_url: str = Field("", max_length=4_000)
    #: Append to this entry when the page already made one for the scan.
    #: Anchored to the logbook's id alphabet (``uuid4().hex[:12]``): the
    #: value becomes a path segment in the URLs we build, and a ``/`` or a
    #: ``..`` in it would address some other route under that base.
    entry: str = Field("", max_length=64, pattern=r"^[0-9a-f]*$")


@router.post("/api/run/{uid}/logbook", status_code=201)
def run_logbook_send(
    uid: str,
    payload: PlotToLogbook,
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Put one rendered plot into this scan's entry in the logbook.

    The portal's **third** write verb (after the analysis runs and the
    config editor) and, like them, an explicit act: nothing is sent
    that a person did not click.  It writes to the logbook's own
    service through that service's public API, and touches neither the
    scans tree nor any portal state.

    Ladder: no absolute ``--logbook-url`` / not this experiment / no
    scan number or day → 404 (there is no entry for it to join) · a
    malformed image → 400 · the logbook unreachable → 503 · the
    logbook refusing → its own status for the verdicts that are about
    this payload (409/413/415), else 502.
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
    try:
        png = logbook_send.decode_png_data_url(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            filename=f"scan{scan:03d}-plot.png",
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
        },
        status_code=201,
    )
