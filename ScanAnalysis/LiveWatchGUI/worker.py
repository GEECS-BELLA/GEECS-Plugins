"""Background worker thread for running LiveTaskRunner.

Wraps :class:`~scan_analysis.live_task_runner.LiveTaskRunner` in a
:class:`QThread` so the GUI remains responsive while the file-system
watcher and analysis loop run continuously.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from geecs_data_utils import ScanPaths, ScanTag
from scan_analysis.live_task_runner import LiveTaskRunner

logger = logging.getLogger(__name__)


def _stop_runner_with_timeout(runner: LiveTaskRunner, timeout: float = 5.0) -> None:
    """Stop a LiveTaskRunner with a timeout to prevent indefinite blocking.

    Calls ``observer.stop()`` (non-blocking signal) then ``observer.join()``
    with a bounded timeout.  Previous versions spawned a daemon thread for
    this, but that created orphaned threads that could emit Qt signals after
    the QThread had already finished — leading to intermittent GUI freezes.

    Parameters
    ----------
    runner : LiveTaskRunner
        The runner to stop.
    timeout : float
        Maximum time to wait for the observer thread to finish (seconds).
    """
    try:
        runner.observer.stop()  # non-blocking: signals the observer to exit
    except Exception as exc:
        logger.warning("Error calling observer.stop(): %s", exc)
        return

    try:
        runner.observer.join(timeout=timeout)
        if runner.observer.is_alive():
            logger.warning(
                "PollingObserver did not stop within %s seconds; "
                "abandoning join to avoid blocking the QThread",
                timeout,
            )
    except Exception as exc:
        logger.warning("Error joining observer: %s", exc)


@dataclass
class LiveWatchConfig:
    """All parameters needed to construct and run a LiveTaskRunner.

    Mirrors the constructor and ``process_new`` arguments so the GUI
    can build one of these from its widgets and hand it to the worker.
    """

    analyzer_group: str
    year: int
    month: int
    day: int
    start_scan_number: int = 0
    experiment: str = ""
    gdoc_enabled: bool = False
    document_id: Optional[str] = None
    config_dir: Optional[Path] = None
    image_config_dir: Optional[Path] = None

    # process_new kwargs
    max_items: int = 1
    dry_run: bool = False
    rerun_completed: bool = False
    rerun_failed: bool = False

    def to_scan_tag(self) -> ScanTag:
        """Build a :class:`ScanTag` from the stored date/experiment fields."""
        return ScanTag(
            year=self.year,
            month=self.month,
            day=self.day,
            number=self.start_scan_number,
            experiment=self.experiment or self.analyzer_group,
        )


class LiveWatchWorker(QThread):
    """QThread that owns and drives a :class:`LiveTaskRunner`.

    Signals
    -------
    status_changed : str
        Emitted when the runner state changes (``"running"``, ``"stopped"``,
        ``"error"``).
    error_occurred : str
        Emitted with an error message if the runner raises an exception.

    Parameters
    ----------
    config : LiveWatchConfig
        Configuration built from the GUI widgets.
    """

    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: LiveWatchConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop_event = threading.Event()
        self._runner: Optional[LiveTaskRunner] = None

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D401 – Qt naming convention
        """Thread body: create runner, start watcher, loop ``process_new``."""
        self._stop_event.clear()
        try:
            tag = self.config.to_scan_tag()
            self._runner = LiveTaskRunner(
                analyzer_group=self.config.analyzer_group,
                date_tag=tag,
                config_dir=self.config.config_dir,
                image_config_dir=self.config.image_config_dir,
                gdoc_enabled=self.config.gdoc_enabled,
                document_id=self.config.document_id,
            )
            self._runner.start()
            self.status_changed.emit("running")

            base_directory: Optional[Path] = None
            try:
                base_directory = ScanPaths.paths_config.base_path
            except Exception:
                pass

            while not self._stop_event.is_set():
                self._runner.process_new(
                    base_directory=base_directory,
                    max_items=self.config.max_items,
                    dry_run=self.config.dry_run,
                    rerun_completed=self.config.rerun_completed,
                    rerun_failed=self.config.rerun_failed,
                )
                # Sleep in small increments so we can respond to stop quickly
                for _ in range(10):
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.1)

        except Exception as exc:
            logger.exception("LiveWatchWorker encountered an error")
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("error")
        finally:
            if self._runner is not None:
                logger.info("Stopping LiveTaskRunner...")
                _stop_runner_with_timeout(self._runner, timeout=5.0)
                logger.info("LiveTaskRunner stop completed (or timed out)")
            self.status_changed.emit("stopped")

    # ------------------------------------------------------------------
    # Public API (called from the GUI thread)
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the worker loop to exit gracefully."""
        self._stop_event.set()
