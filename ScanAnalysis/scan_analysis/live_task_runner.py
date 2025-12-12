"""
Lightweight live runner for scan analysis using the new config + task queue.

Intended for script/notebook use (no GUI): watches a day's analysis folder for new
s-files, initializes per-analyzer status, rebuilds a priority worklist, and processes
the highest-priority queued tasks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional

from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.config_roots import image_analysis_config, scan_analysis_config
from scan_analysis.task_queue import (
    build_worklist,
    init_status_for_scan,
    load_analyzers_from_config,
    reset_status_for_scan,
    run_worklist,
)
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

logger = logging.getLogger(__name__)

SFILE_REGEX = re.compile(r"^s(?P<num>\d+)\.txt$", re.IGNORECASE)


def extract_scan_number(filename: str) -> Optional[int]:
    """Return scan number from an s-file name like s123.txt."""
    m = SFILE_REGEX.match(filename)
    if m:
        return int(m["num"])
    return None


class AnalysisEventHandler(FileSystemEventHandler):
    """Watchdog handler that enqueues ScanTags when new s-files appear."""

    def __init__(self, queue: Queue, base_tag: ScanTag):
        super().__init__()
        self.queue = queue
        self.base_tag = base_tag

    def on_created(self, event):
        """Enqueue a ScanTag when a new s-file appears."""
        if event.is_directory:
            return
        scan_number = extract_scan_number(Path(event.src_path).name)
        if scan_number is None:
            return
        tag = ScanTag(
            year=self.base_tag.year,
            month=self.base_tag.month,
            day=self.base_tag.day,
            number=scan_number,
            experiment=self.base_tag.experiment,
        )
        logger.info("New s-file detected: %s", event.src_path)
        self.queue.put(tag)


class LiveTaskRunner:
    """Watch for new scans, track status, and run analyzers by priority."""

    def __init__(
        self,
        experiment: str,
        date_tag: ScanTag,
        *,
        config_dir: Optional[Path] = None,
        image_config_dir: Optional[Path] = None,
    ):
        """Initialize runner with experiment, date, and optional config roots.

        Parameters
        ----------
        experiment : str
            Experiment name (e.g., 'Undulator').
        date_tag : ScanTag
            ScanTag with year/month/day/experiment (number ignored).
        config_dir : Path, optional
            Base dir for scan analysis configs (if None, uses scan_analysis_config.base_dir).
        image_config_dir : Path, optional
            Base dir for image analysis configs (if None, uses image_analysis_config.base_dir).
        """
        self.experiment = experiment
        self.date_tag = date_tag
        if config_dir:
            scan_analysis_config.set_base_dir(config_dir)
        if image_config_dir:
            image_analysis_config.set_base_dir(image_config_dir)

        self.analyzers = load_analyzers_from_config(
            experiment, config_dir=scan_analysis_config.base_dir
        )
        self.queue: Queue = Queue()
        self.observer = PollingObserver()
        self._reset_done_tags: set[tuple[str, int, int, int, int]] = set()

        # Watch today's analysis folder
        self.watch_folder = (
            ScanPaths.get_scan_folder_path(tag=date_tag).parents[1] / "analysis"
        )
        self.watch_folder.mkdir(parents=True, exist_ok=True)
        handler = AnalysisEventHandler(self.queue, date_tag)
        self.observer.schedule(handler, str(self.watch_folder))

    def start(self):
        """Start the filesystem observer."""
        self.observer.start()
        logger.info("Started watching %s", self.watch_folder)

    def stop(self):
        """Stop the filesystem observer."""
        self.observer.stop()
        self.observer.join()

    def process_new(
        self,
        base_directory: Optional[Path] = None,
        *,
        max_items: int = 1,
        dry_run: bool = False,
        rerun_completed: bool = False,
        rerun_failed: bool = True,
        rerun_only_ids: Optional[Iterable[str]] = None,
        rerun_skip_ids: Optional[Iterable[str]] = None,
    ):
        """
        Process up to max_items highest-priority queued tasks across known scans.

        Parameters
        ----------
        base_directory : Path, optional
            Root for data; defaults to configured base path.
        max_items : int
            Maximum tasks to run this call.
        dry_run : bool
            If True, update status but skip analyzer execution.
        rerun_completed : bool
            If True, reset done analyzers to queued once per discovered scan.
        rerun_failed : bool
            If True, requeue analyzers marked failed.
        rerun_only_ids : Iterable[str], optional
            If provided, only these analyzer ids are eligible for rerun/reset.
        rerun_skip_ids : Iterable[str], optional
            Analyzer ids to exclude from rerun/reset.
        """
        # Drain queue into a set of tags
        new_tags = []
        while not self.queue.empty():
            new_tags.append(self.queue.get())

        # Initialize status for new scans
        for tag in new_tags:
            init_status_for_scan(tag, self.analyzers, base_directory=base_directory)

        # Build worklist across all scans we know about (watch folder discovery)
        tags = self._discover_scan_tags(base_directory=base_directory)
        if rerun_completed:
            for t in tags:
                key = (t.experiment, t.year, t.month, t.day, t.number)
                if key not in self._reset_done_tags:
                    reset_status_for_scan(
                        t,
                        self.analyzers,
                        base_directory=base_directory,
                        states_to_reset=("done", "claimed"),
                        rerun_only_ids=rerun_only_ids,
                        rerun_skip_ids=rerun_skip_ids,
                    )
                    self._reset_done_tags.add(key)
        work = build_worklist(
            tags,
            self.analyzers,
            base_directory=base_directory,
            rerun_completed=False,  # rerun_completed is handled by reset above
            rerun_failed=rerun_failed,
            rerun_only_ids=rerun_only_ids,
            rerun_skip_ids=rerun_skip_ids,
        )
        if work:
            run_worklist(
                work[:max_items], base_directory=base_directory, dry_run=dry_run
            )

    def _discover_scan_tags(
        self, base_directory: Optional[Path] = None
    ) -> Iterable[ScanTag]:
        """Discover scans under the watch folder's parent (analysis siblings)."""
        # The s-files live under <date>/analysis/sXXX.txt; analysis is sibling to scans
        daily_scans = ScanPaths.get_daily_scan_folder(
            tag=self.date_tag, base_directory=base_directory
        )
        scan_root = daily_scans.parent / "analysis"
        tags = []
        if scan_root.exists():
            for f in scan_root.iterdir():
                if f.is_file():
                    num = extract_scan_number(f.name)
                    if num is not None:
                        tags.append(
                            ScanTag(
                                year=self.date_tag.year,
                                month=self.date_tag.month,
                                day=self.date_tag.day,
                                number=num,
                                experiment=self.experiment,
                            )
                        )
        return tags
