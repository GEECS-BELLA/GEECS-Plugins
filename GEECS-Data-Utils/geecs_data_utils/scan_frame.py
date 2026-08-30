"""The union-with-provenance scan frame — one scan, every column source.

THE substrate of the analysis-tabs arc
(``Planning/data_portal/03_analysis_tabs_design.md``): a scan's scalar
columns from every provider, side by side, each column tagged with
where it came from.  Providers today: the Bluesky event table (a
``RunDetail`` from :mod:`geecs_data_utils.tiled_catalog`) and the
legacy s-file (:mod:`geecs_data_utils.data.sfile`); ephemeral analysis
results join later as a third tag, ``"computed"``.

Doctrine (settled with the owner, 2026-08-29/30): **union, never
reconciliation** — the two namespaces share no column names by
construction, near-duplicate quantities under different spellings are
kept as-is, and nothing ever reverse-maps through the PV-name
sanitizer.  Rows join on shot identity: ``Shotnumber`` (s-file) equals
``scan_event_index`` (event table), **both 1-based** — the plans
increment before emitting and the legacy exporter writes
``Shotnumber = 1..N`` over the same rows.

Read-only throughout (repo scan-folder invariant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from geecs_data_utils.tiled_catalog import RunDetail

logger = logging.getLogger(__name__)

#: Provenance tags a column can carry.
PROVENANCE_RUN = "run"
PROVENANCE_SFILE = "sfile"
PROVENANCE_COMPUTED = "computed"

#: Suffix applied to an s-file column whose name collides with an event
#: column (the namespaces are disjoint by construction, so a collision
#: is exceptional — the event column keeps the bare name).
_SFILE_COLLISION_SUFFIX = " (s-file)"


@dataclass(frozen=True)
class ProvenancedFrame:
    """One scan's union frame plus per-column provenance.

    Attributes
    ----------
    frame : pandas.DataFrame
        One row per shot (outer join over shot identity — a shot one
        provider missed carries NaN on that provider's columns).
    provenance : dict of str to str
        ``column name -> "run" | "sfile" | "computed"``.  Carried
        explicitly rather than in ``DataFrame.attrs`` (attrs are
        silently dropped by many pandas operations).
    """

    frame: "pd.DataFrame"
    provenance: "dict[str, str]" = field(default_factory=dict)

    def columns(self, source: Optional[str] = None) -> "list[str]":
        """Column names, optionally restricted to one provenance tag."""
        if source is None:
            return [str(c) for c in self.frame.columns]
        return [c for c, s in self.provenance.items() if s == source]


def _event_shot_key(frame: "pd.DataFrame") -> "pd.Series":
    """The event table's 1-based shot key, synthesized when absent.

    ``scan_event_index`` is guaranteed by the event schema; the fallback
    mirrors ``tiled_export``'s own ``Shotnumber = 1..N`` synthesis so a
    schema-less injected frame still joins positionally.
    """
    import pandas as pd

    from geecs_data_utils.tiled_schema import SHOT_INDEX_COLUMN

    if SHOT_INDEX_COLUMN in frame.columns:
        return frame[SHOT_INDEX_COLUMN].astype("Int64")
    logger.warning(
        "event frame lacks %s — joining on positional 1..N", SHOT_INDEX_COLUMN
    )
    return pd.Series(range(1, len(frame) + 1), index=frame.index, dtype="Int64")


def scan_frame(
    detail: "Optional[RunDetail]" = None,
    scan_folder: Optional[Path] = None,
    *,
    sfile_path: Optional[Path] = None,
) -> ProvenancedFrame:
    """Build the union-with-provenance frame for one scan.

    Providers are independently optional so every era of data works:
    a Bluesky run with no s-file yet, a legacy scan with no run, or the
    normal case of both.

    Parameters
    ----------
    detail : RunDetail, optional
        The catalog's loaded run (the event-table provider).  Its
        ``data`` may be ``None`` (no primary stream).
    scan_folder : Path, optional
        The scan's ``scans/ScanNNN`` folder; when given, the s-file is
        looked up by convention (:func:`sfile_path_for_scan`) — a
        missing or non-canonical path degrades to run-only with a log
        line, never an error.
    sfile_path : Path, optional
        Explicit s-file path (overrides the *scan_folder* convention;
        tests and non-canonical layouts).

    Returns
    -------
    ProvenancedFrame
        The union frame.  With both providers absent (or empty), the
        frame is empty with empty provenance.

    Notes
    -----
    On the exceptional exact-name collision the event column keeps the
    bare name and the s-file column is stored as
    ``"{name} (s-file)"`` — provenance records both.
    """
    import pandas as pd

    event = None if detail is None else detail.data
    sfile_frame = None
    resolved_sfile = sfile_path
    if resolved_sfile is None and scan_folder is not None:
        try:
            resolved_sfile = sfile_path_for_scan(scan_folder)
        except ValueError as exc:
            logger.info("no conventional s-file path: %s", exc)
    if resolved_sfile is not None:
        try:
            sfile_frame = read_sfile(resolved_sfile)
        except FileNotFoundError:
            logger.info("no s-file at %s — run-only frame", resolved_sfile)
        except Exception as exc:  # noqa: BLE001 — a corrupt s-file must not sink the run side
            logger.warning("unreadable s-file %s: %s", resolved_sfile, exc)

    if event is None and sfile_frame is None:
        return ProvenancedFrame(frame=pd.DataFrame(), provenance={})

    if sfile_frame is None:
        provenance = {str(c): PROVENANCE_RUN for c in event.columns}
        return ProvenancedFrame(frame=event.copy(), provenance=provenance)

    if event is None:
        provenance = {str(c): PROVENANCE_SFILE for c in sfile_frame.columns}
        return ProvenancedFrame(frame=sfile_frame.copy(), provenance=provenance)

    # Both providers: outer join on shot identity (1-based == 1-based).
    event_cols = [str(c) for c in event.columns]
    renames = {
        c: f"{c}{_SFILE_COLLISION_SUFFIX}"
        for c in sfile_frame.columns
        if str(c) in event_cols
    }
    if renames:
        logger.info("s-file column collisions renamed: %s", sorted(renames))
        sfile_frame = sfile_frame.rename(columns=renames)

    left = event.copy()
    right = sfile_frame.copy()
    left["_shot_key"] = _event_shot_key(event)
    shot_col = (
        f"Shotnumber{_SFILE_COLLISION_SUFFIX}"
        if "Shotnumber" in renames
        else "Shotnumber"
    )
    try:
        if shot_col in right.columns:
            # pd.to_numeric first: a parses-but-corrupt key cell (stray
            # text, non-integral float) must degrade like any other
            # corrupt s-file, never sink the run side.
            right["_shot_key"] = pd.to_numeric(right[shot_col], errors="raise").astype(
                "Int64"
            )
        else:
            logger.warning("s-file lacks Shotnumber — joining on positional 1..N")
            right["_shot_key"] = pd.Series(
                range(1, len(right) + 1), index=right.index, dtype="Int64"
            )
    except (TypeError, ValueError) as exc:
        logger.warning("unusable s-file shot key (%s) — run-only frame", exc)
        provenance = {str(c): PROVENANCE_RUN for c in event.columns}
        return ProvenancedFrame(frame=event.copy(), provenance=provenance)

    # One row per shot is the contract: duplicate keys (corruption or
    # hand edits — producers write unique 1..N) keep the FIRST row,
    # matching the shared keep-first join doctrine. NA keys collapse
    # too (pandas treats <NA> duplicates as equal) — blank-Shotnumber
    # rows aren't shots, so one survivor is the honest rendering.
    dupes = right["_shot_key"].duplicated(keep="first")
    if bool(dupes.any()):
        logger.warning(
            "s-file has %d duplicate Shotnumber row(s) — keeping first",
            int(dupes.sum()),
        )
        right = right[~dupes]

    merged = left.merge(right, on="_shot_key", how="outer", sort=True)
    merged = merged.drop(columns="_shot_key")

    provenance = {str(c): PROVENANCE_RUN for c in event.columns}
    provenance.update({str(c): PROVENANCE_SFILE for c in sfile_frame.columns})
    return ProvenancedFrame(frame=merged, provenance=provenance)
