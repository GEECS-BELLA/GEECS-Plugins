"""RunControl — interface between GEECS Scanner GUI and the scan execution backend."""

import logging
from pathlib import Path
from typing import Callable, Optional

from geecs_scanner.app.lib.action_control import ActionControl
from geecs_scanner.app.lib.gui_utilities import read_yaml_file_to_dict
from geecs_scanner.engine import DatabaseDictLookup
from geecs_scanner.engine.models.scan_execution_config import ScanExecutionConfig
from geecs_scanner.engine.scan_events import ScanEvent


class RunControl:
    """Interface class between the GEECS Scanner GUI and the scan execution backend.

    All scan execution is routed through the Bluesky RunEngine backend
    (``BlueskyScanner``) — the legacy ScanManager backend was deleted in G1
    of the greenfield cutover (``Planning/cutover_strategy/00_overview.md``).
    ``shot_control_configuration`` is optional and geecs-python-api hardware
    connections are not established at init time.
    """

    def __init__(
        self,
        experiment_name: str = "",
        shot_control_configuration: Optional[Path] = None,
        on_event: Optional[Callable[[ScanEvent], None]] = None,
    ):
        """Initialise with the Bluesky backend.

        Parameters
        ----------
        experiment_name : str
            Experiment name as in the GEECS Database and Scan Manager file structure.
        shot_control_configuration : Path, optional
            Path to the configuration file with shot control information.
        on_event : callable, optional
            Callback invoked for every :class:`~geecs_scanner.engine.scan_events.ScanEvent`.
            Pass ``pyqtSignal.emit`` to route events onto the Qt main thread.
        """
        from geecs_bluesky.scanner_bridge import BlueskyScanner

        from geecs_scanner.optimization.session_bridge import (
            load_session_optimization,
        )

        self.experiment_name = experiment_name
        self._database_lookup = DatabaseDictLookup()

        settings = None
        if shot_control_configuration is not None:
            settings = read_yaml_file_to_dict(shot_control_configuration)
        self.scan_manager = BlueskyScanner(
            experiment_dir=experiment_name,
            shot_control_information=settings,
            on_event=on_event,
            optimization_loader=load_session_optimization,
        )
        # Manual actions (ActionLibrary "perform action" / condition checks)
        # ride ActionManager via ActionControl — backend-independent (kept in
        # G1), constructed lazily in get_action_control for the current
        # experiment. Re-homing onto the schema action compiler is G-actions
        # (Planning/cutover_strategy/00_overview.md).
        self.action_control = None
        logging.info("RunControl: using Bluesky backend (BlueskyScanner)")

    def get_database_dict(self) -> dict:
        """Return the dictionary of the entire database stored in Scan Manager."""
        if self.scan_manager is None:
            return {}
        try:
            self._database_lookup.reload(experiment_name=self.experiment_name)
            return self._database_lookup.get_database()
        except Exception as exc:
            logging.warning("Error retrieving database dictionary: %s", exc)
            return {}

    def get_action_control(
        self, experiment_name_refresh: Optional[str] = None
    ) -> ActionControl:
        """Return the action control instance associated with the current experiment.

        Parameters
        ----------
        experiment_name_refresh : str, optional
            If provided, reinitialize the action control with this experiment name.

        Returns
        -------
        ActionControl
        """
        if experiment_name_refresh:
            self.action_control = ActionControl(experiment_name=experiment_name_refresh)
        elif self.action_control is None:
            # Lazy default: the legacy backend built this at init; post-G1 it
            # is created on first use so manual actions keep working
            # (ActionControl/ActionManager are backend-independent).
            self.action_control = ActionControl(experiment_name=self.experiment_name)
        return self.action_control

    def submit_run(self, exec_config: ScanExecutionConfig) -> bool:
        """Submit a scan request to Scan Manager after reinitializing it.

        Parameters
        ----------
        exec_config : ScanExecutionConfig
            Fully validated scan execution config produced by the GUI.

        Returns
        -------
        bool
            True if device reinitialization succeeded.
        """
        success = False
        if self.scan_manager is not None:
            success = self.scan_manager.reinitialize(exec_config=exec_config)
            self.scan_manager.start_scan_thread()
        return success

    def stop_scan(self):
        """Send a stop request to Scan Manager."""
        if self.scan_manager is not None:
            self.scan_manager.stop_scanning_thread()
