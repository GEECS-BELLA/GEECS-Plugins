"""Export legacy GEECS scalar files from a Bluesky run.

Downstream GEECS analysis (ScanAnalysis, optimization, log tooling) consumes
the legacy on-disk scalar files the original scanner wrote:

- ``scans/ScanNNN/ScanDataScanNNN.txt`` — tab-separated scalar summary.
- ``analysis/sNNN.txt`` — a copy of the same table; the *mutable* one analysis
  code appends to.

:func:`write_scalar_files` writes both from a run's start document and its
row stream as a DataFrame — the worker's s-file callback feeds it straight
from the live documents at the stop document;
:func:`write_scalar_files_from_tiled` reads a recorded run back from Tiled
by ``uid`` first (an offline re-export).  The Bluesky event stream names
scalar columns ``<ophyd>-<safe_var>`` (e.g. ``uc_wavemeter-wavelength_nm``),
an irreversible mangling of the original GEECS ``Device Variable``; the
run's start document carries a ``geecs_scalar_headers`` map (event key →
``Device Variable``) recorded at scan time so the original headers can be
recovered.  See ``GeecsBluesky/EVENT_SCHEMA.md``.

**Rows are not always events.**  A strict run records every essential
device per shot, so its rows are the ``primary`` events.  A *gated* run's
``primary`` is datum-only — the frames and their per-frame scalars live in
each camera's stack — and its rows are the per-shot sampler's ``shots``
events; a *non-essential* camera streams into its own ``<name>_stream``
whichever mode the run used.  Either way the per-frame columns are joined
onto the rows by offset-corrected stamp
(:mod:`geecs_data_utils.shot_join`, ``Planning/native_bluesky/08_gated_batch.md``
§4.5): one s-file row per essential shot, orphan frames left in the stack.

This module is a **consumer** of scan folders: it writes into an
already-claimed ``scans/ScanNNN/`` folder but never creates one (the
cross-package "analysis code never creates scan folders" invariant).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from geecs_data_utils.shot_join import (
    DEFAULT_SHOT_PERIOD_S,
    SHOTS_STREAM,
    FrameColumns,
    clock_device,
    join_frames_to_shots,
    row_windows,
    shot_clock_column,
)
from geecs_data_utils.tiled_catalog import read_tiled_config

logger = logging.getLogger(__name__)

# Derived companion columns (event-schema v1) normally kept out of the legacy
# scalar file.  Emission is driven entirely by ``geecs_scalar_headers`` (any
# event-stream column not named there is dropped), so this set only documents
# the intent — it is not an active filter.  Exception: ``-acq_timestamp`` is
# deliberately surfaced (added to a device's headers) for file/image-saving
# devices so saved files tie back to scan rows; it stays out for pure-scalar
# devices.
_COMPANION_SUFFIXES = (
    "-t0_acq_timestamp",
    "-shot_id",
    "-shot_offset",
    "-valid",
    "-nonscalar_save_path",
)


def build_legacy_scalar_dataframe(
    start_doc: dict[str, Any],
    primary_df: pd.DataFrame,
    frames: Sequence[FrameColumns] = (),
    *,
    drain_offsets: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build a legacy-format scalar DataFrame from a run's start doc + rows.

    Pure transform (no I/O) so it is unit-testable without a live Tiled server.

    Parameters
    ----------
    start_doc:
        The run's start-document metadata.  Must contain ``geecs_scalar_headers``
        (event key → ``"Device Variable"``); ``scan_number`` is used for the
        ``scan`` column when present, ``shot_clock`` / ``detectors`` to pick
        the rows' stamp column and ``shot_period`` to size the join window.
    primary_df:
        The run's row stream as a DataFrame, with Bluesky ophyd-named columns
        (``<ophyd>-<safe_var>``, companion columns, ``bin_number``, …): a
        strict run's ``primary`` events, a gated run's ``shots`` events.
    frames:
        Per-frame columns of each datum-only stream source (a gated run's
        cameras, a non-essential camera), joined onto the rows by
        offset-corrected stamp.  A column a row already carries is never
        overwritten — a strict run's essential camera is read per shot, and
        the event row is the authority.
    drain_offsets:
        Object name → its ``drain_offset``, seconds — the clock device's and
        every frame source's alike, from the streams' descriptor
        configuration.  **One** map for both sides of the join, so the
        worker and the offline re-export cannot correct by different
        amounts; a name it does not carry means ``0.0``.

    Returns
    -------
    pandas.DataFrame
        Columns ``Bin #``, ``scan``, ``<Device Variable>…`` (in
        ``geecs_scalar_headers`` order), ``Shotnumber``.  The legacy
        ``Elapsed Time`` column is intentionally omitted.
    """
    headers: dict[str, str] = dict(start_doc.get("geecs_scalar_headers") or {})
    rows = (
        join_frame_columns(start_doc, primary_df, frames, drain_offsets=drain_offsets)
        if len(frames)
        else primary_df
    )
    n_rows = len(rows)
    out = pd.DataFrame(index=range(n_rows))

    # Row-identity columns.
    if "bin_number" in rows.columns:
        out["Bin #"] = rows["bin_number"].to_numpy()
    else:
        out["Bin #"] = 1
    out["scan"] = start_doc.get("scan_number", 0)

    # Device data columns: rename via the header map, preserving its order.
    # Only keys actually present in the rows are emitted; everything not
    # in the map (companion columns, row-identity columns) is dropped.
    for event_key, legacy_header in headers.items():
        if event_key in rows.columns:
            out[legacy_header] = rows[event_key].to_numpy()
        else:
            logger.debug("geecs_scalar_headers key %r absent from the rows", event_key)

    _warn_unnamed_frame_columns(frames, headers)
    out["Shotnumber"] = range(1, n_rows + 1)
    return out


def _warn_unnamed_frame_columns(
    frames: Sequence[FrameColumns], headers: Mapping[str, str]
) -> None:
    """Warn about a joined column no ``geecs_scalar_headers`` key claims.

    The "a gated row carries the same columns as a strict row" promise
    rests on one mangling (``geecs_core.pv_naming.normalize_component``)
    producing the same spelling on the gateway's attribute name and on the
    namespace's event key.  When it does not, the only symptom is a column
    quietly missing from the s-file — so say so, in the GEECS vocabulary
    where the source knows it (``FrameColumns.raw_names``).  The plugin's
    own ``frame_recv_timestamp`` is never an s-file column and is not a
    drift signal.
    """
    for source in frames:
        unnamed = [
            key
            for key in source.columns
            if key not in headers and not key.endswith("-frame_recv_timestamp")
        ]
        if unnamed:
            logger.warning(
                "%s: %d per-frame column(s) no scalar header names, left out of the "
                "s-file: %s",
                source.object_name,
                len(unnamed),
                ", ".join(
                    f"{key} (GEECS {source.raw_names[key]!r})"
                    if key in source.raw_names
                    else key
                    for key in sorted(unnamed)
                ),
            )


def join_frame_columns(
    start_doc: dict[str, Any],
    rows: pd.DataFrame,
    frames: Sequence[FrameColumns],
    *,
    drain_offsets: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Add each frame source's per-frame columns to *rows*, joined by stamp.

    One row per shot, always: a frame with no row inside the join window is
    an orphan and is dropped here (it stays in the stack and in Tiled —
    ``08_gated_batch.md`` §4.5, Sam's answer to §6 Q4), and a row with no
    frame from a source gets ``NaN`` in that source's columns.  Orphans and
    duplicates are logged per source; for a gated run's *essential* cameras
    both are zero by construction (the batch trims to the quota), so a
    non-zero count there is worth the warning it gets.

    Parameters
    ----------
    start_doc:
        The run's start document (``shot_clock``, ``detectors``, ``shot_period``).
    rows:
        The run's row stream (one row per shot).
    frames:
        The datum-only stream sources to join.
    drain_offsets:
        Object name → its ``drain_offset``, seconds, for the clock device
        and every source alike.

    Returns
    -------
    pandas.DataFrame
        A copy of *rows* with the joined columns appended.
    """
    if not len(rows) or not len(frames):
        return rows
    clock = shot_clock_column(start_doc, list(rows.columns))
    if clock is None:
        logger.warning(
            "run %s has no stamp column in its rows; %d frame stream(s) not joined",
            start_doc.get("uid"),
            len(frames),
        )
        return rows
    offsets = dict(drain_offsets or {})
    shot_stamps = rows[clock].to_numpy(dtype=float)
    period = float(start_doc.get("shot_period") or DEFAULT_SHOT_PERIOD_S)
    windows = row_windows(shot_stamps, period)
    clock_offset = float(offsets.get(clock_device(clock), 0.0))
    out = rows.copy()
    for source in frames:
        join = join_frames_to_shots(
            shot_stamps,
            source.stamps,
            windows=windows,
            shot_offset=clock_offset,
            frame_offset=float(offsets.get(source.object_name, 0.0)),
        )
        if join.orphans or join.contested or join.matched != len(rows):
            logger.warning(
                "%s: %d of %d frame(s) joined to %d row(s) on %s (windows "
                "±%.3f-%.3f s, drain %+.3f s vs the clock's %+.3f s) — %d orphan(s) "
                "left out of the s-file, %d duplicate(s) dropped",
                source.object_name,
                join.matched,
                len(source),
                len(rows),
                clock,
                float(windows.min()) if windows.size else 0.0,
                float(windows.max()) if windows.size else 0.0,
                float(offsets.get(source.object_name, 0.0)),
                clock_offset,
                len(join.orphans),
                len(join.contested),
            )
        for key, values in source.columns.items():
            if key in out.columns:
                continue  # the event row is the authority for its own column
            out[key] = _gathered(values, join.frame_for_shot)
    return out


def _gathered(values: Any, indices: Sequence[int | None]) -> np.ndarray:
    """*values* gathered at *indices*, ``NaN`` where a row has no frame."""
    array = np.asarray(values, dtype=float)
    out = np.full(len(indices), np.nan, dtype=float)
    for row, index in enumerate(indices):
        if index is not None and 0 <= index < array.size:
            out[row] = array[index]
    return out


def _resolve_output_paths(start_doc: dict[str, Any]) -> Optional[tuple[Path, Path]]:
    """Resolve ``(ScanDataScanNNN.txt, sNNN.txt)`` paths from the start doc.

    Returns ``None`` (and logs) when the scan folder is missing from the start
    doc or does not exist on disk — this module never creates scan folders.
    """
    scan_folder = start_doc.get("scan_folder")
    if not scan_folder:
        logger.warning("Run has no scan_folder in start doc; cannot write s-file")
        return None
    folder = Path(scan_folder)
    if not folder.is_dir():
        logger.warning(
            "Scan folder %s does not exist; refusing to create it (invariant)",
            folder,
        )
        return None

    scan_number = start_doc.get("scan_number")
    if scan_number is None:
        # Fall back to the trailing digits of the folder name (ScanNNN).
        scan_number = int("".join(ch for ch in folder.name if ch.isdigit()) or "0")

    scan_txt = folder / f"ScanData{folder.name}.txt"
    # analysis/ is the sibling of scans/ under the day folder: replace the
    # "scans" path component with "analysis" and drop the ScanNNN leaf.
    parts = list(folder.parts)
    parts[-2] = "analysis"
    analysis_dir = Path(*parts[:-1])
    analysis_dir.mkdir(exist_ok=True)  # day folder exists; only analysis/ is new
    sfile_txt = analysis_dir / f"s{int(scan_number)}.txt"
    return scan_txt, sfile_txt


def read_run_rows(run: Any) -> tuple[pd.DataFrame, str]:
    """The row stream of a recorded run: ``primary``'s events, else ``shots``.

    A gated run's ``primary`` holds only the cameras' frames (no table
    part), and its per-shot rows are the sampler's ``shots`` events — so
    the re-export of a gated run reads the same rows the worker wrote its
    s-file from.  Only the table parts are read either way
    (:func:`geecs_data_utils.tiled_catalog.read_primary_scalars`): an
    array part of a stream is never downloaded.

    Parameters
    ----------
    run :
        The run's Tiled node.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        The rows and the stream they came from (``""`` when the run has none).
    """
    from geecs_data_utils.tiled_catalog import read_primary_scalars

    for stream in ("primary", SHOTS_STREAM):
        try:
            node = run[stream]
        except KeyError:
            continue
        rows = read_primary_scalars(node)
        if rows is not None and len(rows):
            return rows, stream
    return pd.DataFrame(), ""


def read_frame_columns(run: Any, row_stream: str) -> list[FrameColumns]:
    """Per-frame columns of every stream of a recorded run except the rows'.

    A stream other than the one the rows came from carries no per-shot row
    of its own: a gated run's ``primary`` and every non-essential camera's
    ``<name>_stream``.  Its per-frame attribute columns are 1-D arrays
    named ``<device>-hdf-<variable>-<suffix>``
    (``io.scan_stack.parse_attribute_name``), read **by name** — the frame
    stack itself, the only multi-dimensional part, is never touched, and a
    stream with no such part (``baseline``) contributes nothing.  The
    stamps come back in the stacks' Unix epoch and are converted to the
    rows' LabVIEW epoch here.

    The test is deliberately "not the row stream" rather than "has no table
    part": the two must agree with :func:`read_run_rows`, which falls
    through a stream whose table is *empty*, and a gated ``primary`` that
    carries an empty table part would otherwise lose every camera column
    with no warning at all.  Joining a stream that does have rows would be
    harmless anyway — the row is the authority for a column it carries.

    Parameters
    ----------
    run :
        The run's Tiled node.
    row_stream :
        The stream the rows came from (:func:`read_run_rows`) — skipped.

    Returns
    -------
    list of FrameColumns
        One entry per device, in stream then device order.
    """
    from geecs_data_utils.io.scan_stack import (
        LABVIEW_EPOCH_OFFSET,
        parse_attribute_name,
    )
    from geecs_data_utils.shot_join import frame_columns_from_attributes

    out: list[FrameColumns] = []
    for stream in sorted(run):
        if stream == row_stream:
            continue
        node = run[stream]
        per_device: dict[str, dict[str, Any]] = {}
        for part in node.get_contents():
            parsed = parse_attribute_name(str(part))
            if parsed is None:
                continue  # the frame stack (or a part of another shape)
            per_device.setdefault(parsed[0], {})[str(part)] = node.base[part].read()
        for device, attributes in sorted(per_device.items()):
            columns = frame_columns_from_attributes(
                device,
                attributes,
                labview_epoch_offset=LABVIEW_EPOCH_OFFSET,
            )
            if columns is not None:
                out.append(columns)
    return out


def read_drain_offsets(run: Any) -> dict[str, float]:
    """``{object name: drain offset}`` over every stream of a recorded run.

    One map for the whole run, so the clock device's offset is found
    wherever its stream happens to be — the gap that made an offline
    re-export correct only one side of the join.
    """
    offsets: dict[str, float] = {}
    for stream in run:
        try:
            offsets.update(_descriptor_drain_offsets(run[stream]))
        except Exception:  # noqa: BLE001 - a stream without usable metadata
            logger.debug("stream %r has no readable configuration", stream)
    return offsets


def _descriptor_drain_offsets(node: Any) -> dict[str, float]:
    """``{object name: drain offset}`` from one stream node's descriptor configuration.

    A Tiled stream node carries the descriptor's ``configuration`` block at
    the top of its own metadata (verified against the lab catalog,
    2026-09-12: ``{object: {"data": {"<object>-drain_offset": …}}}``); a
    writer that nests the whole descriptor list under ``descriptors``
    instead is read as well.  Missing either way means ``0.0``, which is
    what every offset reads until the sync calibration (``03`` §11.7,
    §4.F) sets them.
    """
    metadata = node.metadata
    blocks = [metadata.get("configuration") or {}]
    blocks += [
        descriptor.get("configuration") or {}
        for descriptor in metadata.get("descriptors") or ()
    ]
    offsets: dict[str, float] = {}
    for block in blocks:
        for name, config in block.items():
            value = (config.get("data") or {}).get(f"{name}-drain_offset")
            if value is not None:
                offsets[str(name)] = float(value)
    return offsets


def _fetch_run(uid: str, tiled_uri: str, tiled_api_key: Optional[str]):
    """Return ``(start_doc, rows, frames, drain_offsets)`` for *uid* from Tiled."""
    try:
        from tiled.client import from_uri
    except ImportError as exc:  # pragma: no cover - exercised only without tiled
        raise RuntimeError(
            "tiled is not installed; install the 'tiled' extra "
            "(pip install 'geecs-data-utils[tiled]') to export scalar files"
        ) from exc

    client = from_uri(tiled_uri, api_key=tiled_api_key)
    run = client[uid]
    start_doc = dict(run.metadata.get("start") or {})
    rows, row_stream = read_run_rows(run)
    return (
        start_doc,
        rows,
        read_frame_columns(run, row_stream),
        read_drain_offsets(run),
    )


def write_scalar_files(
    start_doc: dict[str, Any],
    primary_df: pd.DataFrame,
    frames: Sequence[FrameColumns] = (),
    *,
    drain_offsets: Mapping[str, float] | None = None,
) -> Optional[tuple[Path, Path]]:
    """Write the legacy scalar files for one run from its documents.

    Parameters
    ----------
    start_doc:
        The run's start document (``scan_folder``, ``scan_number``,
        ``geecs_scalar_headers``).
    primary_df:
        The run's row stream as a DataFrame (one row per shot, Bluesky
        ophyd-named columns): a strict run's ``primary`` events, a gated
        run's ``shots`` events.
    frames:
        Per-frame columns of the run's datum-only stream sources, joined
        onto the rows by stamp (:func:`join_frame_columns`).
    drain_offsets:
        Object name → its ``drain_offset``, seconds (:func:`join_frame_columns`).

    Returns
    -------
    tuple[Path, Path] or None
        ``(scan_data_txt_path, sfile_txt_path)`` on success, or ``None`` when
        the scan folder is absent (see :func:`_resolve_output_paths`) or the
        run has no scalar rows.
    """
    paths = _resolve_output_paths(start_doc)
    if paths is None:
        return None
    scan_txt, sfile_txt = paths

    df = build_legacy_scalar_dataframe(
        start_doc, primary_df, frames, drain_offsets=drain_offsets
    )
    if df.empty:
        logger.warning(
            "Run %s produced no scalar rows; nothing written", start_doc.get("uid")
        )
        return None

    df.to_csv(scan_txt, sep="\t", index=False)
    df.to_csv(sfile_txt, sep="\t", index=False)
    logger.info("Wrote legacy scalar files: %s and %s", scan_txt, sfile_txt)
    return scan_txt, sfile_txt


def write_scalar_files_from_tiled(
    uid: str,
    *,
    tiled_uri: Optional[str] = None,
    tiled_api_key: Optional[str] = None,
) -> Optional[tuple[Path, Path]]:
    """Read a Bluesky run from Tiled and write the legacy scalar files.

    Works for either shape of run: a strict run's ``primary`` events, or a
    gated run's ``shots`` events joined to the cameras' per-frame columns
    (:func:`read_run_rows`, :func:`read_frame_columns`) — so re-exporting a
    gated run offline reproduces the s-file the worker wrote.

    Parameters
    ----------
    uid:
        The Bluesky run uid (the start-document ``uid``).
    tiled_uri, tiled_api_key:
        Tiled connection details.  When omitted, read from the ``[tiled]``
        section of ``~/.config/geecs_python_api/config.ini``.

    Returns
    -------
    tuple[Path, Path] or None
        ``(scan_data_txt_path, sfile_txt_path)`` on success, or ``None`` when
        the scan folder is absent (see :func:`_resolve_output_paths`).

    Raises
    ------
    RuntimeError
        If ``tiled`` is not installed or no Tiled URI can be resolved.
    """
    if tiled_uri is None:
        tiled_uri, tiled_api_key = read_tiled_config()
    if not tiled_uri:
        raise RuntimeError(
            "No Tiled URI given and none found in "
            "~/.config/geecs_python_api/config.ini [tiled]"
        )

    start_doc, rows, frames, offsets = _fetch_run(uid, tiled_uri, tiled_api_key)
    return write_scalar_files(start_doc, rows, frames, drain_offsets=offsets)
