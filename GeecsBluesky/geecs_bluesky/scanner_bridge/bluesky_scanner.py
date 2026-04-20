"""BlueskyScanner — drop-in RunEngine backend for GEECS Scanner GUI.

Provides the same interface as ``ScanManager`` so that ``RunControl`` can
route scan requests to either the legacy ScanManager or this Bluesky backend,
controlled by a single flag.

Supported scan modes (MVP):
    STANDARD — step scan via :func:`~geecs_bluesky.plans.step_scan.geecs_step_scan`
    NOSCAN   — fixed-position data collection via ``bluesky.plans.count``

The RunEngine is created once on ``__init__`` and its internal event loop
persists for the lifetime of this object.  Devices are created from the GEECS
database and connected in the RE's loop at the start of each scan, then
disconnected when the scan finishes or is aborted.

Usage (standalone)::

    from geecs_bluesky.scanner_bridge import BlueskyScanner
    from geecs_data_utils import ScanConfig, ScanMode

    scanner = BlueskyScanner()
    scanner.reinitialize(config_dictionary={"shots_per_step": 5})
    scanner.start_scan_thread(ScanConfig(
        scan_mode=ScanMode.STANDARD,
        device_var="U_ESP_JetXYZ:Position.Axis 1",
        start=4.0, end=6.0, step=0.5,
    ))
    while scanner.is_scanning_active():
        time.sleep(0.5)
        print(f"{scanner.estimate_current_completion()*100:.0f}%")
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any

import numpy as np
from bluesky import RunEngine
import bluesky.plans as bp

from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.plans.step_scan import geecs_step_scan

# ScanConfig / ScanMode are only imported for type hints; duck-typing is used at
# runtime so this module can be imported without geecs_data_utils installed.
try:
    from geecs_data_utils import ScanConfig as _ScanConfig  # noqa: F401
except Exception:
    _ScanConfig = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 20.0
_DISCONNECT_TIMEOUT = 10.0


def _safe_ophyd_name(s: str) -> str:
    """Convert an arbitrary string to a valid ophyd-async device name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_").lower()


def _build_positions(scan_config: Any) -> list[float]:
    """Convert ScanConfig start/end/step to an explicit list of positions."""
    start = float(scan_config.start)
    end = float(scan_config.end)
    step = float(scan_config.step)
    if step == 0 or start == end:
        return [start]
    n = max(2, round(abs(end - start) / abs(step)) + 1)
    return list(np.linspace(start, end, n))


class BlueskyScanner:
    """RunEngine-backed scan executor compatible with the ``ScanManager`` API.

    Parameters
    ----------
    experiment_dir:
        Experiment name (currently unused; reserved for future path integration).
    shot_control_information:
        Shot-controller YAML config dict (currently unused; shots_per_step can
        be passed via ``config_dictionary`` in :meth:`reinitialize`).
    """

    def __init__(
        self,
        experiment_dir: str = "",
        shot_control_information: dict | None = None,
    ) -> None:
        self._experiment_dir = experiment_dir
        self._RE = RunEngine()
        self._RE.subscribe(self._on_document)

        self._scan_thread: threading.Thread | None = None
        self._config_dict: dict[str, Any] = {}
        self._shots_per_step: int = 10

        self._total_shots: int = 0
        self._completed_shots: int = 0

        # Devices held open between reinitialize() and scan completion
        self._motor: GeecsMotor | None = None
        self._detectors: list = []

        # Lock for serialising _motor create/destroy across threads
        self._device_lock = threading.Lock()

        logger.info("BlueskyScanner initialised (RunEngine ready)")

    # ------------------------------------------------------------------
    # ScanManager-compatible public API
    # ------------------------------------------------------------------

    def reinitialize(
        self,
        config_path: Any = None,
        config_dictionary: dict | None = None,
        **_kwargs: Any,
    ) -> bool:
        """Store the device configuration for the next scan.

        Parameters
        ----------
        config_path:
            Ignored (compatibility shim).
        config_dictionary:
            Dict of scan/device configuration.  Recognised keys:
                ``shots_per_step`` (int, default 10) — shots per motor position.

        Returns
        -------
        bool
            Always ``True`` (no hardware initialisation done here).
        """
        self._config_dict = dict(config_dictionary or {})
        self._shots_per_step = int(self._config_dict.get("shots_per_step", 10))
        self._completed_shots = 0
        self._total_shots = 0

        # Disconnect any leftover motor from a previous scan
        self._disconnect_devices_sync()

        logger.info(
            "BlueskyScanner reinitialised — shots_per_step=%d", self._shots_per_step
        )
        return True

    def start_scan_thread(self, scan_config: Any) -> None:
        """Launch the scan in a background thread.

        Parameters
        ----------
        scan_config:
            A :class:`~geecs_data_utils.ScanConfig`-compatible object with fields
            ``scan_mode``, ``device_var``, ``start``, ``end``, ``step``,
            ``wait_time``, ``additional_description``.
        """
        if self.is_scanning_active():
            logger.warning(
                "start_scan_thread called while scan already active; ignoring"
            )
            return

        self._completed_shots = 0
        self._scan_thread = threading.Thread(
            target=self._run_scan,
            args=(scan_config,),
            daemon=True,
            name="bluesky-scan",
        )
        self._scan_thread.start()
        logger.info("BlueskyScanner scan thread started")

    def stop_scanning_thread(self) -> None:
        """Abort the running scan and wait for the thread to finish."""
        logger.info("BlueskyScanner: abort requested")
        try:
            self._RE.abort(reason="stop_scanning_thread called")
        except Exception:
            logger.debug("RE.abort() raised (may not be running)", exc_info=True)
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=15)
            self._scan_thread = None

    def pause_scan(self) -> None:
        """Request the RunEngine to pause between plans steps."""
        logger.info("BlueskyScanner: pause requested")
        try:
            self._RE.request_pause()
        except Exception:
            logger.debug("RE.request_pause() raised", exc_info=True)

    def resume_scan(self) -> None:
        """Resume a paused RunEngine."""
        logger.info("BlueskyScanner: resume requested")
        try:
            self._RE.resume()
        except Exception:
            logger.debug("RE.resume() raised", exc_info=True)

    def is_scanning_active(self) -> bool:
        """Return ``True`` if a scan thread is currently running."""
        return bool(self._scan_thread and self._scan_thread.is_alive())

    def estimate_current_completion(self) -> float:
        """Return fraction complete (0.0–1.0) based on shots emitted."""
        if self._total_shots == 0:
            return 0.0
        return min(self._completed_shots / self._total_shots, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_document(self, name: str, doc: dict) -> None:
        if name == "event":
            self._completed_shots += 1

    def _connect_device(self, device) -> None:
        """Connect an ophyd-async device in the RE's persistent event loop."""
        asyncio.run_coroutine_threadsafe(device.connect(), self._RE._loop).result(
            timeout=_CONNECT_TIMEOUT
        )

    def _disconnect_device(self, device) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                device.disconnect(), self._RE._loop
            ).result(timeout=_DISCONNECT_TIMEOUT)
        except Exception:
            logger.debug("disconnect raised for %s", device, exc_info=True)

    def _disconnect_devices_sync(self) -> None:
        """Disconnect motor + detectors if they exist (called from non-scan threads)."""
        with self._device_lock:
            if self._motor is not None:
                self._disconnect_device(self._motor)
                self._motor = None
            for det in self._detectors:
                self._disconnect_device(det)
            self._detectors = []

    def _run_scan(self, scan_config: Any) -> None:
        """Scan thread body: create devices, run plan, clean up."""
        try:
            mode = scan_config.scan_mode
            # Support both ScanMode enum and plain strings
            mode_val = mode.value if hasattr(mode, "value") else str(mode).lower()
            logger.info("BlueskyScanner: starting %s scan", mode_val)

            if mode_val == "standard":
                self._run_standard_scan(scan_config)
            elif mode_val == "noscan":
                self._run_noscan(scan_config)
            else:
                logger.warning(
                    "BlueskyScanner: scan mode %r not yet supported; skipping", mode_val
                )
        except Exception:
            logger.exception("BlueskyScanner scan thread raised an exception")
        finally:
            self._disconnect_devices_sync()
            logger.info("BlueskyScanner: scan thread finished")

    def _run_standard_scan(self, scan_config: Any) -> None:
        """Step scan: move motor through positions, collect shots at each step."""
        if not scan_config.device_var:
            logger.error("STANDARD scan requires device_var; got None")
            return

        if ":" not in scan_config.device_var:
            logger.error(
                "device_var must be 'DeviceName:VariableName', got: %r",
                scan_config.device_var,
            )
            return

        device_name, variable = scan_config.device_var.split(":", 1)
        ophyd_name = _safe_ophyd_name(f"{device_name}_{variable}")

        logger.info(
            "Creating motor: %s / %s → name=%r", device_name, variable, ophyd_name
        )
        motor = GeecsMotor.from_db_axis(device_name, variable, name=ophyd_name)

        logger.info("Connecting motor …")
        self._connect_device(motor)
        with self._device_lock:
            self._motor = motor

        positions = _build_positions(scan_config)
        n_steps = len(positions)
        n_shots = n_steps * self._shots_per_step
        self._total_shots = n_shots

        logger.info(
            "Step scan: %d steps × %d shots/step = %d total events",
            n_steps,
            self._shots_per_step,
            n_shots,
        )
        logger.info("Positions: %s", positions)

        md = {
            "operator": "",
            "scan_mode": getattr(
                scan_config.scan_mode, "value", str(scan_config.scan_mode)
            ),
            "device_var": scan_config.device_var,
            "wait_time": scan_config.wait_time,
            "bluesky_backend": True,
        }
        if scan_config.additional_description:
            md["description"] = scan_config.additional_description

        self._RE(
            geecs_step_scan(
                motor=motor,
                positions=positions,
                detectors=list(self._detectors),
                shots_per_step=self._shots_per_step,
                md=md,
            )
        )

    def _run_noscan(self, scan_config: Any) -> None:
        """Fixed-position data collection — count plan if detectors present."""
        n_shots = self._shots_per_step
        self._total_shots = n_shots

        if not self._detectors:
            logger.info(
                "NOSCAN with no detectors: nothing to collect (%d shots requested). "
                "Add detector devices via the device configuration to enable data collection.",
                n_shots,
            )
            return

        logger.info(
            "NOSCAN: collecting %d shots from %d detectors",
            n_shots,
            len(self._detectors),
        )

        md = {
            "scan_mode": getattr(
                scan_config.scan_mode, "value", str(scan_config.scan_mode)
            ),
            "bluesky_backend": True,
        }
        self._RE(bp.count(self._detectors, num=n_shots, md=md))
