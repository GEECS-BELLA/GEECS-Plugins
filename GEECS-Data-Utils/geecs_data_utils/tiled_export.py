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

**Rows are not always events.**  A strict run's rows are the ``primary``
events (an earlier strict run's carry its waited-on devices too; a
shot-table run's carry none).  A *gated* run's
``primary`` is datum-only — the frames and their per-frame scalars live in
each camera's stack — and its rows are the per-shot sampler's ``shots``
events; a streamed device records into its own ``<name>_stream`` (every
per-shot device of a shot-table run, a gated run's devices not required)
— a plugin camera's frames, or one event per
stamp a triggered device without a plugin published.  Either way the
per-frame columns are joined onto the rows by offset-corrected stamp
through the shot table (:mod:`geecs_data_utils.shot_table`, which also
reads a shot-table run, ``layout: "shot-table/1"``): one s-file row per
row of the run, orphan frames left in the stack.

This module is a **consumer** of scan folders: it writes into an
already-claimed ``scans/ScanNNN/`` folder but never creates one (the
cross-package "analysis code never creates scan folders" invariant).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from geecs_data_utils.shot_join import (
    SHOTS_STREAM,
    FrameColumns,
    device_stream,
)
from geecs_data_utils.shot_table import (
    COUNTER_KEYS,
    LAYOUT_KEY,
    WITNESS_ATTRIBUTE_KEY,
    DeviceRecords,
    device_records,
    per_shot_event_streams,
    shot_table,
)
from geecs_data_utils.data.sfile import scan_data_txt_path_for
from geecs_data_utils.tiled_catalog import read_tiled_config
from geecs_data_utils.tiled_schema import ANNOUNCED_SHOT_COLUMN, SFILE_ANNOUNCED_SHOT

logger = logging.getLogger(__name__)

#: A per-shot source: a device's records with their stream, or its bare
#: columns (joined by stamp, as a ``<name>_stream`` is).
Source = DeviceRecords | FrameColumns

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
    frames: Sequence[Source] = (),
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
        Each per-shot source outside the rows (a gated run's cameras, a
        streamed camera or event stream, a shot-table run's device
        streams), joined onto the rows by the shot table
        (:func:`join_frame_columns`).  A column a row already carries is never
        overwritten — an earlier strict run's camera was read per shot, and
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
        ``geecs_scalar_headers`` order), ``Announced Shot`` (scan-mode runs
        only), ``Shotnumber``.  The legacy
        ``Elapsed Time`` column is intentionally omitted.
    """
    headers: dict[str, str] = dict(start_doc.get("geecs_scalar_headers") or {})
    rows = (
        join_frame_columns(start_doc, primary_df, frames, drain_offsets=drain_offsets)
        if len(frames)
        else primary_df
    )
    n_rows = len(rows)
    # Collected first, allocated once: inserting column by column into an
    # existing frame fragments it, and past ~100 columns pandas warns
    # ("DataFrame is highly fragmented") — an HTU s-file with its background
    # telemetry (GEECS-Plugins#1016) carries ~300.
    columns: dict[str, Any] = {}

    # Row-identity columns.
    if "bin_number" in rows.columns:
        columns["Bin #"] = rows["bin_number"].to_numpy()
    else:
        columns["Bin #"] = 1
    columns["scan"] = start_doc.get("scan_number", 0)

    # Device data columns: rename via the header map, preserving its order.
    # Only keys actually present in the rows are emitted; everything not
    # in the map (companion columns, row-identity columns) is dropped.
    for event_key, legacy_header in headers.items():
        if event_key in rows.columns:
            columns[legacy_header] = rows[event_key].to_numpy()
        else:
            logger.debug("geecs_scalar_headers key %r absent from the rows", event_key)

    _warn_unnamed_frame_columns(frames, headers)
    # Scan mode: the number in the devices' file names, which is not always
    # the row index (EVENT_SCHEMA.md, ``announced_shot``).
    if ANNOUNCED_SHOT_COLUMN in rows.columns:
        columns[SFILE_ANNOUNCED_SHOT] = rows[ANNOUNCED_SHOT_COLUMN].to_numpy()
    columns["Shotnumber"] = range(1, n_rows + 1)
    return pd.DataFrame(columns, index=range(n_rows))


def _warn_unnamed_frame_columns(
    frames: Sequence[Source], headers: Mapping[str, str]
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
    for source in (_records(frame).records for frame in frames):
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
    frames: Sequence[Source],
    *,
    drain_offsets: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Add each source's per-record columns to *rows*, through the shot table.

    One row per shot, always: a record no row claims is an extra and is
    dropped here (it stays in its stream and in Tiled — Sam, 2026-09-12),
    and a row with no record from a source gets ``NaN`` in that source's
    columns (:func:`geecs_data_utils.shot_table.shot_table`, either
    layout).  Extras and gaps are logged per source; for a gated run's
    *essential* cameras both are zero by construction (the batch trims to
    the quota), so a non-zero count there is worth the warning it gets.

    Parameters
    ----------
    start_doc:
        The run's start document (``shot_clock``, ``detectors``,
        ``shot_period``; ``layout`` for a shot-table run).
    rows:
        The run's row stream (one row per shot).
    frames:
        The sources to join; every one is joined by stamp, never by
        position (the row stream's own stacks are not passed here).
    drain_offsets:
        Object name → its ``drain_offset``, seconds, for the clock device
        and every source alike.

    Returns
    -------
    pandas.DataFrame
        *rows* with the joined columns appended.
    """
    if not len(rows) or not len(frames):
        return rows
    if (
        start_doc.get(LAYOUT_KEY)
        and not start_doc.get("shot_clock_column")
        and "fire_stamp" in rows.columns
    ):
        logger.warning(
            "run %s: a shot-table run naming no shot_clock_column; timed on fire_stamp",
            start_doc.get("uid"),
        )
        start_doc = {**start_doc, "shot_clock_column": "fire_stamp"}
    devices = [_records(source) for source in frames]
    table = shot_table(
        start_doc, rows, devices, row_stream="", drain_offsets=drain_offsets
    )
    if table.clock is None:
        logger.warning(
            "run %s has no stamp column in its rows; %d frame stream(s) not joined",
            start_doc.get("uid"),
            len(frames),
        )
        return rows
    for source in devices:
        assigned = table.record_for_shot[source.device]
        matched = sum(index is not None for index in assigned)
        extras = len(table.extras.get(source.device, ()))
        if extras or matched != len(rows):
            logger.warning(
                "%s: %d of %d frame(s) joined to %d row(s) on %s — %d orphan(s) "
                "left out of the s-file",
                source.device,
                matched,
                len(source.records),
                len(rows),
                table.clock,
                extras,
            )
    return table.rows


def _records(source: Source) -> DeviceRecords:
    """*source* as :class:`DeviceRecords`; bare columns stream as ``<name>_stream``."""
    if isinstance(source, DeviceRecords):
        return source
    return DeviceRecords(stream=device_stream(source.object_name), records=source)


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

    scan_txt = scan_data_txt_path_for(folder)
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
    """The columns of :func:`read_device_records`, without their streams."""
    return [source.records for source in read_device_records(run, row_stream)]


def read_device_records(
    run: Any, row_stream: str, *, stamps_only: bool = False
) -> list[DeviceRecords]:
    """Every per-shot source of a recorded run except the rows' own, with its stream.

    The Tiled adapter of :func:`geecs_data_utils.shot_table.shot_table`:
    each source's class and staleness from the start document's
    ``stream_meta``, its counter and witness attribute when the stream carries
    them (a bare ``NDArrayUniqueId`` / ``WitnessCount`` part belongs to
    the stream's one stamped device — an attribute stream's or an event
    stream's alike: a native camera's stream is an event stream of its
    frame stamps beside its stack's NDAttribute parts, or a datum-only
    stream whose bare parts carry the EPICS stamp pair, joined on that
    pair as the live s-file joins the stack).  A stream that
    publishes neither (a plugin stack carries no per-frame counter) has no
    audit here, as in the s-file.  *stamps_only* reads each device's stamps
    and counters and no scalar column (what a frame lookup needs).

    A stream other than the one the rows came from carries no per-shot row
    of its own: a gated run's ``primary`` and every streamed device's
    ``<name>_stream``.  A streamed device **without** a file plugin
    records an event per stamp it published instead of frames; its
    stream's table (named by the start document's ``per_shot_streams``, or
    a gated run's ``non_essential``) is read as join columns by :func:`_event_stream_columns`.  Otherwise
    its per-frame attribute columns are 1-D arrays named ``<device>-hdf-<variable>-<suffix>``
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

    Each device's columns are **truncated to the frame count Tiled reports
    for that device's stack** (its shape, read from the node's structure
    metadata — no download).  That count comes from the run's stream
    datums, which is exactly what the worker's live path truncates to, so
    the two paths agree whether or not the server happens to clip a 1-D
    attribute dataset to the datum range: a streamed camera's frames
    written between its last ``collect`` and its ``unstage`` reach neither
    s-file.

    Parameters
    ----------
    run :
        The run's Tiled node.
    row_stream :
        The stream the rows came from (:func:`read_run_rows`) — skipped.

    Returns
    -------
    list of DeviceRecords
        One entry per device, in stream then device order.
    """
    from geecs_data_utils.io.scan_stack import EPICS_TS_SEC, LABVIEW_EPOCH_OFFSET
    from geecs_data_utils.shot_join import (
        DEVICE_STREAM_SUFFIX,
        frame_columns_from_attributes,
    )

    start = dict((getattr(run, "metadata", None) or {}).get("start") or {})
    audit_keys = (*COUNTER_KEYS, WITNESS_ATTRIBUTE_KEY)
    event_streams = per_shot_event_streams(start)
    out: list[DeviceRecords] = []
    for stream in sorted(run):
        if stream == row_stream:
            continue
        node = run[stream]
        if stream in event_streams:
            columns = _event_stream_columns(event_streams[stream], node)
            if columns is not None:
                if stamps_only:
                    columns = _stamps_and_counters(columns, audit_keys)
                bare = {
                    part: node.base[part].read()
                    for part in map(str, node.get_contents())
                    if part in audit_keys
                }
                out.append(device_records(stream, columns, start, bare))
                continue
        contents = node.get_contents()
        per_device, bare = _attribute_parts(node, contents, audit_keys, stamps_only)
        if not per_device and EPICS_TS_SEC in bare:
            # A native camera's stream: its NDAttribute parts carry bare names,
            # the EPICS stamp pair among them, and belong to the stream's device.
            device = event_streams.get(
                stream, stream.removesuffix(DEVICE_STREAM_SUFFIX)
            )
            per_device = {device: dict(bare)}
        built: list[FrameColumns] = []
        for device, attributes in sorted(per_device.items()):
            columns = frame_columns_from_attributes(
                device,
                attributes,
                labview_epoch_offset=LABVIEW_EPOCH_OFFSET,
            )
            if columns is None:
                continue
            referenced = _referenced_frames(contents, device)
            if 0 <= referenced < len(columns):
                logger.info(
                    "%s/%s: %d frame(s) in the attribute arrays, %d referenced by "
                    "the stream's datums; the join uses the referenced ones",
                    stream,
                    device,
                    len(columns),
                    referenced,
                )
                columns = columns.truncated(referenced)
            built.append(columns)
        # A bare counter part is the stream's one stamped device's; with
        # more, whose it is would be a guess.
        bare = bare if len(built) == 1 else {}
        out += [device_records(stream, columns, start, bare) for columns in built]
    return out


def _attribute_parts(
    node: Any, contents: Mapping[str, Any], audit_keys: Sequence[str], stamps_only: bool
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``(device → its attribute parts, the bare audit parts)`` of one stream node.

    Only 1-D parts named ``<device>-hdf-<variable>-<suffix>`` are read (never
    the frame stack), and the bare audit parts with a native camera's EPICS
    stamp pair (``NDArrayEpicsTSSec`` / ``NDArrayEpicsTSnSec``); *stamps_only* keeps the stamps and audit columns
    (matched case- and separator-blind: a plugin suffix is lowercase).
    """
    from geecs_data_utils.io.scan_stack import (
        EPICS_TS_NSEC,
        EPICS_TS_SEC,
        parse_attribute_name,
    )
    from geecs_data_utils.shot_join import FRAME_STAMP_SUFFIX, _squash

    keep = {_squash(key) for key in (FRAME_STAMP_SUFFIX, *audit_keys)}

    per_device: dict[str, dict[str, Any]] = {}
    bare: dict[str, Any] = {}
    for part in map(str, contents):
        parsed = parse_attribute_name(part)
        if parsed is None:
            if part in audit_keys or part in (EPICS_TS_SEC, EPICS_TS_NSEC):
                bare[part] = node.base[part].read()
            continue  # the frame stack (or a part of another shape)
        if stamps_only and _squash(parsed[2]) not in keep:
            continue
        per_device.setdefault(parsed[0], {})[part] = node.base[part].read()
    return per_device, bare


def _stamps_and_counters(columns: FrameColumns, keys: Sequence[str]) -> FrameColumns:
    """*columns* with only its stamp and its audit columns (``<device>-<key>``, any case)."""
    from geecs_data_utils.shot_join import _squash

    name = columns.object_name
    kept = {_squash(f"{name}-{key}") for key in ("acq_timestamp", *keys)}
    return FrameColumns(
        object_name=name,
        stamps=columns.stamps,
        columns={k: v for k, v in columns.columns.items() if _squash(k) in kept},
    )


def _event_stream_columns(object_name: str, node: Any) -> FrameColumns | None:
    """A streamed device's event stream as join columns, or ``None``.

    The stream of a triggered device without a file plugin: one event per
    stamp it published (``shot_join.frame_columns_from_events``).  Its
    table part is read like the rows' (never an array part); a stream with
    no events still yields its numeric keys from the stream's
    ``data_keys`` metadata (the descriptor's), so the offline s-file
    carries the same all-``NaN`` columns the live one does.  ``None``
    (quietly) for a stream without the device's own ``<name>-acq_timestamp``
    — a plugin camera's datum-only stream, read from its attribute arrays
    instead.
    """
    from geecs_data_utils.shot_join import (
        ACQ_TIMESTAMP_SUFFIX,
        frame_columns_from_events,
        numeric_data_keys,
    )
    from geecs_data_utils.tiled_catalog import read_primary_scalars

    # The device's OWN stamp column is what makes it an event stream: a
    # plugin camera's datum stream carries per-frame numeric keys too
    # (``<name>-hdf-<variable>-frame_acq_timestamp``) but never this one.
    stamp = f"{object_name}{ACQ_TIMESTAMP_SUFFIX}"
    table = read_primary_scalars(node)
    if table is not None and len(table) and stamp in table.columns:
        return frame_columns_from_events(object_name, table.to_dict("records"))
    data_keys = (getattr(node, "metadata", None) or {}).get("data_keys") or {}
    keys = numeric_data_keys(data_keys)
    if stamp not in keys:
        return None
    return frame_columns_from_events(object_name, [], keys=keys)


def _referenced_frames(contents: Mapping[str, Any], device: str) -> int:
    """Frames the run's datums gave *device*, from its stack part's shape, or ``-1``.

    Tiled builds an external array's shape from the ``StreamDatum``
    ``indices``, not from the file, so the part named after the device
    reports the referenced count without any data being read.
    """
    item = contents.get(device)
    if item is None:
        return -1
    structure = ((item.get("attributes") or {}).get("structure")) or {}
    shape = structure.get("shape") or ()
    try:
        return int(shape[0])
    except (IndexError, TypeError, ValueError):
        return -1


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
    what every offset reads until the sync calibration sets them.
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
        read_device_records(run, row_stream),
        read_drain_offsets(run),
    )


def write_scalar_files(
    start_doc: dict[str, Any],
    primary_df: pd.DataFrame,
    frames: Sequence[Source] = (),
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
        The run's per-shot sources outside the rows, joined onto them
        (:func:`join_frame_columns`).
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
