"""Analysis runs: list and start one analyzer on one scan, serve its files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from geecs_portal import analysis_runs, resources
from geecs_portal.state import PortalState, get_state

router = APIRouter()


@router.get("/api/run/{uid}/analysis")
def run_analysis_list(
    uid: str, day: str = "", portal: PortalState = Depends(get_state)
) -> JSONResponse:
    """The scan's analyzers: applicability, job record, files on disk.

    ``applicable`` = the diagnostic's data device (``scan.device``,
    else its name) has a data folder in this scan. Every loadable
    diagnostic is listed regardless, so a device-less one is still
    reachable; the tab collapses the inapplicable ones.
    """
    portal.analysis_available()
    detail, folder, analysis_folder, tag = portal.analysis_context(uid, day)
    devices = set(resources.image_devices(folder))
    try:
        present = {p.name for p in folder.iterdir() if p.is_dir()}
    except OSError:
        present = set()
    jobs = portal.runner.jobs_for(uid)
    analyzers = []
    for name, info in portal.processing_infos().items():
        job = jobs.get(name)
        # What the tab shows: everything on disk under the output dir
        # (summaries + per-bin visuals, classified server-side) plus
        # any label the finished job returned that is not a file —
        # described (servable / inline / kind / bin) so the page
        # never guesses from a path's shape.
        files = analysis_runs.list_artifacts(analysis_folder, info.output_name)
        shown = list(files)
        if job is not None and job.state == analysis_runs.DONE:
            shown += [a for a in job.artifacts if a not in files]
        analyzers.append(
            {
                "id": name,
                "device": info.device,
                "applicable": info.device in devices or info.device in present,
                "output_dir": info.output_name,
                "destructive": info.destructive,
                "job": job.to_json() if job is not None else None,
                "files": files,
                "artifacts": analysis_runs.describe_artifacts(
                    analysis_folder, shown, known_files=set(files)
                ),
            }
        )
    running = portal.runner.running_for(uid)
    return JSONResponse(
        {
            "analyzers": analyzers,
            "running": running.analyzer_id if running is not None else None,
            # The number the start endpoint's confirm check compares
            # against (parsed from the resolved folder, like the run's
            # tag) — the page asks for this one, never a guess.
            "scan_number": tag.number,
        },
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/api/run/{uid}/analysis", status_code=202)
def run_analysis_start(
    uid: str,
    analyzer: str,
    day: str = "",
    confirm: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Start one analyzer on this scan (202 + the fresh job record).

    Ladder: feature off / extra missing 404 · unknown or unloadable
    diagnostic 404 · folder unresolvable 404 · a destructive kind
    without ``confirm=<this scan's number>`` 400 (the tab asks for
    it; the gate is here so no client runs a delete unasked) · a job
    already running for this scan 409 (its record in the body).
    Build + run + cleanup happen on the worker thread, so everything
    past this point — config errors included — lands in the record
    as ``failed``.
    """
    portal.analysis_available()
    info = portal.processing_infos().get(analyzer)
    if info is None:
        raise HTTPException(status_code=404, detail=f"no diagnostic: {analyzer!r}")
    _, _, analysis_folder, tag = portal.analysis_context(uid, day)
    if info.destructive and confirm.strip() != str(tag.number):
        raise HTTPException(
            status_code=400,
            detail=f"{analyzer!r} deletes data: confirm with this scan's "
            "number (confirm=<scan number>) to run it",
        )
    config_dir = Path(portal.processing_config_dir)

    def run(progress: analysis_runs.ProgressSink) -> Optional[list]:
        # The opt-in reaches the factory only past the confirm check
        # above — the one place a destructive kind gets built here.
        return analysis_runs.run_scan_analyzer(
            portal.factory,
            analyzer,
            config_dir,
            tag,
            progress=progress,
            allow_destructive=info.destructive,
        )

    try:
        job = portal.runner.start(uid, analyzer, run, relative_to=analysis_folder)
    except analysis_runs.RunInProgress as exc:
        return JSONResponse(
            {
                "detail": "a run is in progress for this scan",
                "job": exc.job.to_json(),
            },
            status_code=409,
        )
    except RuntimeError as exc:  # shutting down
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(job.to_json(), status_code=202)


@router.get("/run/{uid}/artifact")
def run_artifact(
    uid: str, path: str, day: str = "", portal: PortalState = Depends(get_state)
) -> FileResponse:
    """Serve one file a run produced, from the scan's analysis folder only.

    Containment is the contract (:func:`analysis_runs.contained_artifact`):
    anything that resolves outside the scan's own analysis folder is
    a 404, same as a missing file. Same feature gate as the run
    endpoints. Raster images render inline; every other type is a
    download (``attachment`` + ``nosniff``) — the share is writable
    by many hands, and a planted HTML/SVG must never execute in the
    portal's (or, behind the OSPREY proxy, OSPREY's) origin.
    ``no-cache``: re-runs overwrite by name.
    """
    portal.analysis_available()
    _, _, analysis_folder, _ = portal.analysis_context(uid, day)
    file = analysis_runs.contained_artifact(analysis_folder, path)
    if file is None:
        raise HTTPException(status_code=404, detail="no such artifact")
    headers = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}
    inline = file.suffix.lower() in analysis_runs.INLINE_IMAGE_SUFFIXES
    return FileResponse(
        file,
        headers=headers,
        content_disposition_type="inline" if inline else "attachment",
        filename=file.name,
    )
