"""Background file-move worker for post-shot device file management.

After each shot, camera and spectrometer devices write files with a timestamp
encoded in the filename.  :class:`FileMover` matches those files against the
acquisition timestamps recorded in scalar data and moves them into the scan's
output directory with the standard ``Scan{NNN}_{device}_{shot}`` naming convention.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from geecs_data_utils import timestamp_from_filename
from geecs_scanner.utils.exceptions import DataFileError
from geecs_scanner.utils.retry import retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileSavePolicy:
    """File-saving behaviour for a device type."""

    expected_files: int = 1
    exact_name_match: bool = False
    variant_suffixes: tuple[str, ...] = ()


_DEFAULT_POLICY = FileSavePolicy()

_DEVICE_TYPE_POLICY: dict[str, FileSavePolicy] = {
    "FROG": FileSavePolicy(expected_files=2),
    "PicoscopeV2": FileSavePolicy(expected_files=2),
    "Thorlabs CCS175 Spectrometer": FileSavePolicy(expected_files=2),
    "RohdeSchwarz_RTA4000": FileSavePolicy(expected_files=2),
    "ThorlabsWFS": FileSavePolicy(expected_files=2),
    "MagSpecCamera": FileSavePolicy(
        exact_name_match=True, variant_suffixes=("-interp", "-interpSpec", "-interpDiv")
    ),
    "MagSpecStitcher": FileSavePolicy(
        exact_name_match=True, variant_suffixes=("-interpSpec", "-interpDiv")
    ),
}


@dataclass
class FileMoveTask:
    """Task definition for moving and renaming a device file after acquisition."""

    source_dir: Path
    target_dir: Path
    device_name: str
    device_type: (
        str  # TODO Device type is not required, consider making an Optional[str]
    )
    expected_timestamp: float
    shot_index: int
    random_part: Optional[str] = None
    suffix: Optional[str] = None
    new_name: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    files_found_so_far: int = 0


class FileMover:
    """Rename and relocate device files using background worker threads.

    Files are matched by acquisition timestamp: when a device saves a file after
    a shot, the filename encodes the system timestamp, which is also recorded in
    scalar data.  Workers extract that timestamp, compare it against the expected
    value, and rename the file to the scan's shot-number convention.

    ``save_local=True`` (the default) points devices at ``C:/SharedData`` on
    their host rather than the network share, avoiding write-time bottlenecks.

    Attributes
    ----------
    task_queue : queue.Queue
    stop_event : threading.Event
    workers : list[threading.Thread]
    file_check_counts : dict
        How many times each file has been examined; files exceeding the limit
        are moved to ``orphaned_files`` to prevent infinite re-checking.
    orphaned_files : set
    orphan_tasks : list
    processed_files : set
    scan_is_live : bool
    save_local : bool
    scan_number : int or None
    """

    def __init__(self, num_workers: int = 16) -> None:
        self.task_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers: list[threading.Thread] = []
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_func, daemon=True)
            worker.start()
            self.workers.append(worker)

        self.file_check_counts: dict = {}
        self.orphaned_files: set = set()
        self.orphan_tasks: list = []
        self.processed_files: set = set()
        self.scan_is_live = True
        self.save_local = True
        self.scan_number: Optional[int] = None
        logger.info("FileMover worker started.")

    def _worker_func(self) -> None:
        """Drain the task queue until stopped."""
        while True:
            try:
                task: Optional[FileMoveTask] = self.task_queue.get(timeout=1)
            except queue.Empty:
                if self.stop_event.is_set() and self.task_queue.empty():
                    break
                continue

            if task is None:
                continue

            try:
                self._process_task(task)
            except Exception:
                logger.exception("Error processing task")
            finally:
                self.task_queue.task_done()
        logger.info("FileMover worker stopped.")

    def _process_task(self, task: FileMoveTask) -> None:
        """Locate the file matching *task*'s timestamp and move it to the target dir."""
        if task.retry_count > 0:
            time.sleep(0.5)

        source_dir = task.source_dir
        target_dir = task.target_dir
        device_name = task.device_name
        device_type = task.device_type
        expected_timestamp = task.expected_timestamp
        shot_index = task.shot_index

        home_dir = source_dir.parent
        policy = _DEVICE_TYPE_POLICY.get(device_type or "", _DEFAULT_POLICY)

        try:
            if policy.exact_name_match:
                variant_dirs = [
                    d
                    for d in home_dir.iterdir()
                    if d.is_dir() and d.name == device_name
                ]
            else:
                variant_dirs = [
                    d
                    for d in home_dir.iterdir()
                    if d.is_dir() and d.name.startswith(device_name)
                ]
        except OSError as exc:
            raise DataFileError(
                f"Cannot read source directory {home_dir} for device {device_name}"
            ) from exc

        expected_file_count = policy.expected_files

        if not self.save_local:
            time.sleep(0.1)

        task_success = False
        found_files_count = 0
        for variant in variant_dirs:
            adjusted_target_dir = target_dir.parent / variant.name
            adjusted_target_dir.mkdir(parents=True, exist_ok=True)

            for file in variant.glob("*"):
                try:
                    if not file.is_file():
                        continue
                except OSError:
                    continue

                if file in self.orphaned_files:
                    if self.scan_is_live:
                        continue

                if file in self.processed_files:
                    continue

                if self.scan_is_live:
                    self.file_check_counts[file] = (
                        self.file_check_counts.get(file, 0) + 1
                    )
                    if self.file_check_counts[file] > 1:
                        logger.debug(
                            "File %s checked >1 times; marking as orphaned.", file
                        )
                        self.orphaned_files.add(file)
                        continue

                file_ts = timestamp_from_filename(file)
                if abs(file_ts - expected_timestamp) < 0.0011:
                    found_files_count += 1
                    task.files_found_so_far += 1

                    random_part = file.stem.replace(f"{device_name}_", "")
                    task.random_part = random_part

                    task.new_name = self._generate_device_shot_filename(
                        self.scan_number, device_name, shot_index
                    )

                    self._move_file(task, file, variant.name)

                    for suffix in policy.variant_suffixes:
                        task.suffix = suffix
                        self._process_variant_file(task)

                    if task.files_found_so_far >= expected_file_count:
                        task_success = True
                        break

            if task_success:
                break

        if not task_success:
            if self.scan_is_live and task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.debug(
                    "Re-queuing task for %s with timestamp %s (retry %d/%d)",
                    task.device_name,
                    task.expected_timestamp,
                    task.retry_count,
                    task.max_retries,
                )
                self.move_files_by_timestamp(task)
            else:
                logger.warning(
                    "Failed to find a file for %s with timestamp %s — will orphan.",
                    task.device_name,
                    task.expected_timestamp,
                )
                self.orphan_tasks.append(task)

    def _move_file(
        self, task: FileMoveTask, source_file: Path, new_device_name: str
    ) -> None:
        """Move *source_file* to the target dir, retrying on transient network-share errors."""
        target_dir = task.target_dir.parent / new_device_name
        target_dir.mkdir(parents=True, exist_ok=True)
        new_file_stem = (
            task.new_name
            if task.new_name
            else self._generate_device_shot_filename(
                self.scan_number, new_device_name, task.shot_index
            )
        )
        new_filename = new_file_stem + source_file.suffix
        dest_file = target_dir / new_filename
        try:
            retry(
                lambda: shutil.move(str(source_file), str(dest_file)),
                attempts=3,
                delay=0.5,
                backoff=2.0,
                catch=(OSError,),
                on_retry=lambda exc, n: logger.debug(
                    "Retry %d moving %s → %s: %s", n, source_file.name, dest_file, exc
                ),
            )
            self.processed_files.add(dest_file)
        except OSError as exc:
            raise DataFileError(
                f"Failed to move {source_file.name} → {dest_file} after retries"
            ) from exc

    def _process_variant_file(self, task: FileMoveTask) -> None:
        """Find and move the variant file (e.g. ``-interpSpec``) matching *task.random_part*."""
        if task.suffix is None or task.random_part is None:
            return

        variant_dir = task.source_dir.parent / f"{task.device_name}{task.suffix}"
        if not variant_dir.exists():
            return

        candidate = None
        for file in variant_dir.glob("*"):
            if file.is_file() and task.random_part in file.stem:
                candidate = file
                break

        if candidate is None:
            logger.warning(
                "No file found in %s containing %s.", variant_dir, task.random_part
            )
            return

        new_device_name = f"{task.device_name}{task.suffix}"
        self._move_file(task, candidate, new_device_name)

    @staticmethod
    def _generate_device_shot_filename(
        scan_number: int, device_name: str, shot_index: int
    ) -> str:
        """Return ``Scan{NNN}_{device}_{SSS}`` — the shot-number naming convention."""
        scan_number_str = str(scan_number).zfill(3)
        shot_number_str = str(shot_index).zfill(3)
        return f"Scan{scan_number_str}_{device_name}_{shot_number_str}"

    def move_files_by_timestamp(self, task: FileMoveTask) -> None:
        """Enqueue *task* for processing by a worker thread."""
        self.task_queue.put(task)

    def _post_process_orphaned_files(
        self, log_df: pd.DataFrame, device_save_paths_mapping: dict
    ) -> None:
        """Match unprocessed files on disk against log timestamps and re-queue them."""
        logger.info("looking to handle orphaned data files")
        tolerance = 0.0011
        for device_name, device_info in device_save_paths_mapping.items():
            source_dir = Path(device_info["source_dir"])
            target_dir = Path(device_info["target_dir"])
            device_type = device_info["device_type"]

            acq_col = f"{device_name} acq_timestamp"
            if acq_col not in log_df.columns:
                logger.warning(
                    "No acq_timestamp column for %s in log_df; skipping orphan sweep.",
                    device_name,
                )
                continue

            shot_timestamp_pairs = [
                (row["Shotnumber"], row[acq_col])
                for _, row in log_df.iterrows()
                if pd.notnull(row[acq_col])
            ]

            if not shot_timestamp_pairs:
                continue

            home_dir = source_dir.parent
            try:
                variant_dirs = [
                    d
                    for d in home_dir.iterdir()
                    if d.is_dir() and d.name.startswith(device_name)
                ]
            except OSError:
                logger.warning(
                    "Cannot iterate %s for orphan sweep of %s; skipping.",
                    home_dir,
                    device_name,
                )
                continue

            orphan_files = []
            for variant_dir in variant_dirs:
                for f in variant_dir.glob("*"):
                    if f.is_file() and f not in self.processed_files:
                        orphan_files.append(f)

            for file in orphan_files:
                file_ts = timestamp_from_filename(file)
                matched_shot = None

                for shot_number, ts in shot_timestamp_pairs:
                    if abs(file_ts - ts) < tolerance:
                        matched_shot = int(shot_number)
                        break

                if matched_shot is not None:
                    random_part = file.stem.replace(f"{device_name}_", "")
                    task = FileMoveTask(
                        source_dir=file.parent,
                        target_dir=target_dir,
                        device_name=device_name,
                        device_type=device_type,
                        expected_timestamp=file_ts,
                        shot_index=matched_shot,
                        random_part=random_part,
                    )
                    self.move_files_by_timestamp(task)
                else:
                    logger.warning(
                        "No matching shot number found for orphan file %s (timestamp %s)",
                        file,
                        file_ts,
                    )

    def _post_process_orphan_task(self) -> None:
        """Re-queue all previously failed tasks for a final move attempt."""
        orphan_snapshot = list(self.orphan_tasks)
        self.orphan_tasks.clear()
        for task in orphan_snapshot:
            task.retry_count = 0
            self.move_files_by_timestamp(task)

    def shutdown(self, wait: bool = True) -> None:
        """Signal workers to stop; if *wait*, drain the queue first."""
        self.stop_event.set()
        if wait:
            self.task_queue.join()
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()
        logger.debug("FileMover has been shut down gracefully.")
