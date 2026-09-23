"""Resolve completed-scan shot rows to native files or capture-stack frames.

The existing ScanAnalysis mapping rules live here so future scan runners and
readers can share them without importing analysis. This module only discovers
references: image/trace readers still own loading and decoding. Callers must
wait for a scan to close before opening its HDF5 stacks across SMB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from geecs_data_utils.native_files import (
    filename_timestamp_regex,
    legacy_filename_regex,
    probe_native_file,
    timestamp_key,
)
from geecs_data_utils.io.scan_stack import (
    ShotRef,
    find_stack_file,
    frame_index_for_timestamp,
    read_stack_timestamps,
    stack_frame_index_map,
)
from geecs_data_utils.tiled_schema import normalize_token

logger = logging.getLogger(__name__)


class StackMappingUnavailable(LookupError):
    """A stack-only input has no capture frames that can be joined to its rows."""


def map_shot_files(
    directory: Path,
    rows: pd.DataFrame,
    *,
    device: str,
    file_tail: str,
    prefer_stack: bool = False,
    stacks_only: bool = False,
    file_device: str | None = None,
) -> dict[int, Path]:
    """Resolve shot numbers to read-only input references without creating folders.

    The rows carry ``Shotnumber`` and optionally the device's ``acq_timestamp``
    and ``valid`` columns. Timestamps are in LabVIEW seconds, as in an s-file.
    Capture references are ``ShotRef`` Path subclasses carrying a frame index;
    native references are plain Paths. No frame arrays are read by this helper.

    A partial timestamp join is final; only a zero join may try legacy
    shot-number filenames. Stack preference falls back to native files when
    no stack frames match, unless ``stacks_only`` is set. Stack-only inputs
    must also request stack preference. Missing directories return an empty
    map; no raw scan or output directories are ever created.
    """
    if stacks_only and not prefer_stack:
        raise ValueError("stacks_only requires prefer_stack")
    if "Shotnumber" not in rows:
        raise ValueError("Shot rows require a Shotnumber column")
    mapper = _ShotFileMapper(
        directory=Path(directory),
        rows=rows,
        device=device,
        file_tail=file_tail,
        prefer_stack=prefer_stack,
        stacks_only=stacks_only,
        file_device=file_device,
    )
    mapper._build_data_file_map()
    return mapper.paths


@dataclass
class _ShotFileMapper:
    """One mapping pass; mutable work state never escapes except the result map."""

    directory: Path
    rows: pd.DataFrame
    device: str
    file_tail: str
    prefer_stack: bool
    stacks_only: bool
    file_device: str | None
    paths: dict[int, Path] = field(default_factory=dict)

    def _build_data_file_map(self) -> None:
        """
        Build a mapping from shot number to data file path.

        Three strategies: the capture-stack join (config-selected via
        ``data_format="device_hdf5"``, with unconditional fallback), then
        two selected automatically by the metadata present:

        - **acq_timestamp join** (Bluesky-produced scans): when the auxiliary
          frame carries this device's ``acq_timestamp`` column, each shot's
          file is identified by the device's own per-shot timestamp — the
          device names its natively saved files with the same value it
          streams into the event row, so the join is a deterministic lookup
          (canonicalised to integer milliseconds), never a filename guess.
          The join is strictly per-device: this device's column against this
          device's folder; other devices' clocks never enter it.
        - **shot-number filenames** (MC-produced scans): the legacy pattern
          ``Scan<scan_number>_<device_subject>_<shot_number><file_tail>``,
          e.g. ``Scan012_UC_ALineEBeam3_005.png``.

        The presence of an ``acq_timestamp`` column selects the timestamp
        join first, but it is not conclusive on its own: the legacy scanner
        force-appends ``acq_timestamp`` to every synchronous device, so
        MC-produced s-files carry the column while their data files are
        shot-number-named. The filename shape is the real discriminator —
        if the timestamp join maps **zero** files while the auxiliary frame
        expects shots, we fall back to shot-number mapping. A *partial*
        timestamp map (at least one file joined) never falls back: partial
        maps are legitimate for Bluesky scans with invalid rows, and mixing
        strategies would risk attaching wrong files.

        Only files whose suffix + format matches ``file_tail`` exactly are
        included in either strategy.
        """
        self.paths = {}

        # Check if data directory exists
        if not self.directory.exists():
            logger.warning(
                f"Data directory does not exist: {self.directory}. "
                f"Skipping file mapping for device '{self.device}'."
            )
            return

        logger.info(f"self.file_tail: {self.file_tail}")

        # Opt-in capture-stack strategy (data_format="device_hdf5"): map
        # shots into the per-device HDF5 frame stack the PVA gateway's
        # file plugin writes. Zero mappings (no stack, no timestamp column, no joins)
        # fall back to the per-shot file strategies below — the old basis
        # keeps working unconditionally.
        if self.prefer_stack:
            if self._map_shots_from_stack():
                expected_shots = set(self.rows["Shotnumber"].values)
                for m in sorted(expected_shots - set(self.paths.keys())):
                    logger.warning(f"No stack frame found for shot {m}")
                return
            if self.stacks_only:
                # A stack-only reader cannot recover through native paths. Let
                # the caller distinguish absent data from successful analysis.
                raise StackMappingUnavailable(
                    f"'{self.device}' reads the capture stack only "
                    "and no stack could be mapped in "
                    f"{self.directory} — not falling back to per-shot "
                    "files, which that loader cannot read."
                )

        ts_column = self._acq_timestamp_column()
        if ts_column is not None:
            logger.info("Mapping files by device acq_timestamp (column %r)", ts_column)
            self._map_files_by_acq_timestamp(ts_column)
            if not self.paths and len(self.rows) > 0:
                # Legacy MC/GUI scans carry the acq_timestamp column too
                # (device_manager force-appends it), but their files are
                # shot-number-named — the timestamp join then matches
                # nothing. Zero joins with shots expected means the column
                # lied about the file naming; trust the filenames instead.
                logger.info(
                    "acq_timestamp column %r present but no timestamp-named "
                    "files matched in %s; falling back to legacy shot-number "
                    "filename mapping for device '%s'.",
                    ts_column,
                    self.directory,
                    self.device,
                )
                self._map_files_by_shot_number()
        else:
            logger.info("Mapping matched files")
            self._map_files_by_shot_number()

        expected_shots = set(self.rows["Shotnumber"].values)
        found_shots = set(self.paths.keys())
        for m in sorted(expected_shots - found_shots):
            logger.warning(f"No file found for shot {m}")

    def _legacy_filename_regex(self) -> re.Pattern[str]:
        """Compile the MC-convention filename pattern for this device's tail.

        The pattern is owned by :mod:`geecs_data_utils.native_files`.
        """
        return legacy_filename_regex(self.file_tail)

    def _map_files_by_shot_number(self) -> None:
        """Legacy strategy: parse shot numbers out of MC-convention filenames."""
        data_filename_regex = self._legacy_filename_regex()

        for file in self.directory.iterdir():
            if not file.is_file():
                continue

            m = data_filename_regex.match(file.name)
            if m:
                shot_num = int(m.group("shot_number"))
                if shot_num in self.rows["Shotnumber"].values:
                    self.paths[shot_num] = file
                    logger.info(f"Mapped file for shot {shot_num}: {file}")
            else:
                logger.debug(f"Filename {file.name} does not match expected pattern.")

    @staticmethod
    def _normalize_column_token(name: str) -> str:
        """Collapse a name to a matching token — the shared schema rule."""
        return normalize_token(name)

    def _acq_timestamp_column(self) -> Optional[str]:
        """Find this device's ``acq_timestamp`` column in the auxiliary frame.

        Recognises every spelling the column takes across data paths —
        ``"<Device> acq_timestamp"`` (s-file header), ``"<Device>:acq_timestamp"``
        (in-memory frame), ``"<device>-acq_timestamp"`` (raw event key) — by
        normalising both the device name and the column prefix to the same
        token. Returns ``None`` (→ legacy shot-number mapping) when absent.
        """
        if self.rows is None:
            return None
        device_token = self._normalize_column_token(self.device)
        for column in self.rows.columns:
            token = self._normalize_column_token(column)
            if token == f"{device_token}_acq_timestamp":
                return str(column)
        return None

    def _matching_valid_column(self) -> Optional[str]:
        """Find this device's ``valid`` column, if the frame carries one."""
        device_token = self._normalize_column_token(self.device)
        for column in self.rows.columns:
            if self._normalize_column_token(column) == f"{device_token}_valid":
                return str(column)
        return None

    def _map_files_by_acq_timestamp(self, ts_column: str) -> None:
        """Join shots to files via this device's own per-shot ``acq_timestamp``.

        The device stamps one double per acquisition: streamed into the event
        row and written into the native filename (``<name>_<timestamp><tail>``,
        milliseconds precision — the contract owned by
        :mod:`geecs_data_utils.native_files`). Both representations are
        canonicalised to integer milliseconds and joined exactly; the ±1 ms
        fallback below is float-formatting canonicalisation at the rounding
        boundary, not a physical tolerance window. Rows where the device's
        ``valid`` column is
        false are skipped — that device's frame belongs to a different
        physical shot, so "no file for this shot" is the correct answer.
        """
        data_dir = self.directory
        file_device = self.file_device or self.device

        # Directory listings over SMB can serve stale (cached) entries for
        # minutes after a file lands — long enough to hide files written
        # seconds ago during a live scan. The expected filename is fully
        # determined by the row timestamp, so probe it with a direct stat
        # (never served from the listing cache) first; the listing-based map
        # below is only a fallback for unconventional names.
        file_ts_regex = filename_timestamp_regex(self.file_tail)
        legacy_regex = self._legacy_filename_regex()
        files_by_ms: dict[int, Path] = {}
        has_legacy_named = False
        for file in data_dir.iterdir():
            if not file.is_file():
                continue
            m = file_ts_regex.search(file.name)
            if m:
                # keep-first on duplicate ms keys — the shared join contract
                # (same rule as stack_frame_index_map).
                files_by_ms.setdefault(timestamp_key(float(m.group("ts"))), file)
            elif legacy_regex.match(file.name):
                has_legacy_named = True

        # A listing of legacy shot-number files with no timestamp-named files
        # is a legacy scan: the per-row stat probes can never hit, and each
        # probe is a network round-trip (~seconds per device over SMB), so
        # skip straight to the zero-match fallback in _build_data_file_map.
        # An empty or ambiguous listing keeps probing — during a live Bluesky
        # scan a stale SMB listing can hide every file the probes would find.
        probe_expected_names = bool(files_by_ms) or not has_legacy_named
        if not probe_expected_names:
            logger.info(
                "Data directory %s contains only legacy shot-number-named "
                "files; skipping per-shot timestamp filename probes",
                data_dir,
            )

        valid_column = self._matching_valid_column()
        for _, row in self.rows.iterrows():
            shot_num = int(row["Shotnumber"])
            if valid_column is not None and not bool(row[valid_column]):
                continue
            ts = row[ts_column]
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ts) or ts <= 0:
                continue
            file = None
            if probe_expected_names:
                file = self._probe_expected_file(data_dir, file_device, ts)
            if file is None:
                # The shared candidate probe, generic over map values.
                file = frame_index_for_timestamp(files_by_ms, ts)
            if file is not None:
                self.paths[shot_num] = file
                logger.info(f"Mapped file for shot {shot_num}: {file}")

    def _map_shots_from_stack(self) -> bool:
        """Join shots into the capture frame stack, if one exists.

        Mirrors :meth:`_map_files_by_acq_timestamp`'s canonical-millisecond
        join: the stack stores Unix-epoch timestamps (the read side and its
        constants are ``geecs_data_utils.io.scan_stack``), converted here to
        the LabVIEW epoch of the auxiliary frame's ``acq_timestamp`` column.
        Each joined shot maps to a :class:`ShotRef` — a path into the stack
        carrying the frame index — which travels through the existing
        per-shot pipeline and is resolved to pixels by
        ``ImageAnalyzer.load_image``.

        Returns
        -------
        bool
            True when at least one shot mapped (the caller then skips the
            per-shot-file strategies); False means "no usable stack" and the
            caller falls back.
        """
        stack = find_stack_file(self.directory)
        if stack is None:
            logger.warning(
                "data_format='device_hdf5' but no capture stack in %s — "
                "falling back to per-shot files",
                self.directory,
            )
            return False
        ts_column = self._acq_timestamp_column()
        if ts_column is None:
            logger.warning(
                "Capture stack %s present but the auxiliary frame has no "
                "acq_timestamp column for %s — falling back to per-shot files",
                stack,
                self.device,
            )
            return False
        try:
            stack_ts = read_stack_timestamps(stack, labview_epoch=True)
        except Exception as exc:
            # ANY unreadable stack must fall back, not fail the task — the
            # dual-write PNGs are right there. Broad on purpose (review of
            # PR #693): a malformed-but-schema-valid stack can raise
            # TypeError (acq_timestamp as a group) or ValueError (string
            # dtype), not just OSError/KeyError, and the whole point of
            # this strategy is that its failure shapes are recoverable.
            logger.warning(
                "Capture stack %s unreadable (%s) — falling back to per-shot files",
                stack,
                exc,
            )
            return False
        # THE shared keep-first ms-key map (portal parity — one contract).
        frames_by_ms = stack_frame_index_map(stack_ts)
        valid_column = self._matching_valid_column()
        for _, row in self.rows.iterrows():
            shot_num = int(row["Shotnumber"])
            if valid_column is not None and not bool(row[valid_column]):
                continue
            try:
                ts = float(row[ts_column])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ts) or ts <= 0:
                continue
            idx = frame_index_for_timestamp(frames_by_ms, ts)
            if idx is not None:
                self.paths[shot_num] = ShotRef(stack, idx)
        if self.paths:
            logger.info(
                "Mapped %d shot(s) into capture stack %s",
                len(self.paths),
                stack,
            )
            return True
        logger.warning(
            "Capture stack %s joined zero shots (timestamp mismatch?) — "
            "falling back to per-shot files",
            stack,
        )
        return False

    def _probe_expected_file(
        self, data_dir: Path, file_device: str, acq_timestamp: float
    ) -> Optional[Path]:
        """Stat the timestamp-determined filename, if present.

        Delegates to the shared :func:`geecs_data_utils.probe_native_file`
        (direct stats bypassing stale SMB listing caches — the contract's
        one probe, shared with the data portal).
        """
        return probe_native_file(data_dir, file_device, self.file_tail, acq_timestamp)
