"""AppController — business logic extracted from GEECSScannerWindow.

Owns the RunControl lifecycle, experiment state flags, and database access.
Holds no Qt widgets; the window constructs one instance and delegates to it.
"""

from __future__ import annotations

import configparser
import importlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from geecs_scanner.engine import DatabaseDictLookup
from geecs_scanner.engine.scan_events import ScanEvent, ScanState
from geecs_scanner.utils import ApplicationPaths as AppPaths

if TYPE_CHECKING:
    from geecs_scanner.app.run_control import RunControl
    from geecs_scanner.engine.models.scan_execution_config import ScanExecutionConfig

logger = logging.getLogger(__name__)


class AppController:
    """Own the RunControl lifecycle and experiment-level app state.

    Parameters
    ----------
    on_scan_event : callable, optional
        Forwarded to RunControl as the ``on_event`` callback so scan events
        reach the GUI's ``_scan_event_received`` signal.
    unit_test_mode : bool
        When True, skip config-file writes and path initialisation.
    """

    def __init__(
        self,
        on_scan_event: Optional[Callable[[ScanEvent], None]] = None,
        unit_test_mode: bool = False,
    ) -> None:
        self._on_scan_event = on_scan_event
        self._unit_test_mode = unit_test_mode
        self.run_control: Optional[RunControl] = None
        self._database_lookup = DatabaseDictLookup()

        # UI-coordination flags
        self.is_starting: bool = False
        self.is_in_multiscan: bool = False
        self.is_in_action_library: bool = False

    # ------------------------------------------------------------------
    # RunControl lifecycle
    # ------------------------------------------------------------------

    def reinitialize_run_control(
        self,
        experiment_name: str,
        timing_configuration_name: str,
        app_paths: Optional[AppPaths],
    ) -> Optional[RunControl]:
        """Create a new RunControl for the given experiment.

        Parameters
        ----------
        experiment_name : str
        timing_configuration_name : str
        app_paths : AppPaths or None
            If None, RunControl cannot be created.

        Returns
        -------
        RunControl or None
            The new instance, or None if creation failed.  Errors are logged
            but never re-raised so callers can always check for None.
        """
        if not experiment_name or app_paths is None:
            self.run_control = None
            return None

        if not self._unit_test_mode:
            _write_config_if_changed(experiment_name, timing_configuration_name)

        shot_control_path: Optional[Path] = app_paths.shot_control() / (
            timing_configuration_name + ".yaml"
        )
        if not shot_control_path.exists():
            shot_control_path = None

        module_path = Path(__file__).parent / "run_control.py"
        sys.path.insert(0, str(module_path.parent))
        try:
            run_control_class = getattr(
                importlib.import_module("run_control"), "RunControl"
            )
            self.run_control = run_control_class(
                experiment_name=experiment_name,
                shot_control_configuration=shot_control_path,
                on_event=self._on_scan_event,
            )
        except AttributeError:
            logger.error(
                "AttributeError at RunControl: presumably the experiment is not in the GEECS database"
            )
            self.run_control = None
        except KeyError:
            logger.error(
                "KeyError at RunControl: presumably no GEECS Database is connected to locate devices"
            )
            self.run_control = None
        except ValueError:
            logger.error(
                "ValueError at RunControl: presumably no experiment name or shot control given"
            )
            self.run_control = None
        except (ConnectionError, ConnectionRefusedError) as e:
            logger.error("%s at RunControl: %s", type(e), e)
            self.run_control = None
        finally:
            sys.path.pop(0)

        return self.run_control

    # ------------------------------------------------------------------
    # Database access
    # ------------------------------------------------------------------

    def get_database_dict(self, experiment_name: str) -> dict:
        """Return device/variable database for the experiment.

        Falls back to a direct database lookup when RunControl is unavailable.
        """
        if self.run_control is not None:
            return self.run_control.get_database_dict()
        try:
            self._database_lookup.reload(experiment_name=experiment_name)
            return self._database_lookup.get_database()
        except Exception as exc:
            logger.warning("Error retrieving database dictionary: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Scan submission and control
    # ------------------------------------------------------------------

    def submit_scan(self, exec_config: ScanExecutionConfig) -> bool:
        """Reinitialize and start a scan; return True on success."""
        if self.run_control is None:
            return False
        return self.run_control.submit_run(exec_config=exec_config)

    def stop_scan(self) -> None:
        """Signal the scan engine to stop."""
        if self.run_control is not None:
            self.run_control.stop_scan()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> Optional[ScanState]:
        """Current ScanState from the engine, or None if no RunControl."""
        sm = getattr(self.run_control, "scan_manager", None)
        if sm is not None:
            return sm.current_state
        return None

    @property
    def is_scan_active(self) -> bool:
        """True if a scan thread is currently alive."""
        sm = getattr(self.run_control, "scan_manager", None)
        return bool(sm and sm.is_scanning_active())


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------


def _write_config_if_changed(
    experiment_name: str, timing_configuration_name: str
) -> None:
    """Persist experiment and timing config to the GEECS config file if changed."""
    config = configparser.ConfigParser()
    config.read(AppPaths.config_file())

    do_write = False
    if config["Experiment"]["expt"] != experiment_name:
        logger.info("Experiment name changed, rewriting config file")
        config.set("Experiment", "expt", experiment_name)
        do_write = True

    if (not config.has_option("Experiment", "timing_configuration")) or config[
        "Experiment"
    ]["timing_configuration"] != timing_configuration_name:
        logger.info("Timing configuration changed, rewriting config file")
        config.set("Experiment", "timing_configuration", timing_configuration_name)
        do_write = True

    if do_write:
        with open(AppPaths.config_file(), "w") as f:
            config.write(f)
