"""Camera-variable threshold interlock.

Wraps the camera-threshold pattern that was originally implemented as a
factory function (``camera_thresh_check``) in the example notebook.  The
class form makes the parameters configurable from YAML and the device
wiring testable in isolation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from geecs_python_api.controls.devices.geecs_device import GeecsDevice

from .base_interlock import BaseInterlock

logger = logging.getLogger(__name__)


class CameraThresholdInterlock(BaseInterlock):
    """Trip when a camera variable drops below ``threshold`` or stops updating.

    Parameters
    ----------
    device_name : str
        GEECS device name (e.g. ``"CAM-PL1-LC_Film"``).
    variable : str
        Variable on the device to monitor (e.g. ``"MeanCounts"``).
    threshold : float
        Condition is unsafe when the variable's value is strictly below
        this threshold.
    name : str, optional
        Identifier broadcast to clients.  Defaults to
        ``"{device_name} {variable}"``.
    poll_interval : float, optional
        Seconds between checks (default 0.5 s).
    data_timeout : float, optional
        If the device's own ``timestamp`` field hasn't advanced within
        this many seconds, treat the interlock as unsafe.  Default 5 s.
    """

    def __init__(
        self,
        device_name: str,
        variable: str,
        threshold: float,
        name: Optional[str] = None,
        poll_interval: float = 0.5,
        data_timeout: float = 5.0,
    ):
        super().__init__(
            name=name if name is not None else f"{device_name} {variable}",
            poll_interval=poll_interval,
        )
        self.device_name = device_name
        self.variable = variable
        self.threshold = threshold
        self.data_timeout = data_timeout

        self.device: Optional[GeecsDevice] = None
        self._last_device_timestamp = None
        self._last_device_update = time.time()

    def connect(self) -> None:
        """Subscribe to the camera variable.

        Called once by the server before the monitor loop starts.  Kept
        out of ``__init__`` so YAML loading does not trigger a database
        lookup and a TCP subscription as a side effect of constructing
        the object.
        """
        self.device = GeecsDevice(self.device_name)
        self.device.subscribe_var_values([self.variable, "timestamp"])

    def check(self) -> bool:
        """Return True (unsafe) when value < threshold or data is stale."""
        if self.device is None:
            raise RuntimeError(f"{self.name}: connect() must be called before check()")

        state = self.device.state
        value = state.get(self.variable)
        device_timestamp = state.get("timestamp")
        now = time.time()

        if device_timestamp is not None:
            if device_timestamp == self._last_device_timestamp:
                time_frozen = now - self._last_device_update
                if time_frozen > self.data_timeout:
                    logger.warning(
                        "%s: no new %s data for %.1fs; treating as unsafe",
                        self.name,
                        self.variable,
                        time_frozen,
                    )
                    return True
            else:
                self._last_device_timestamp = device_timestamp
                self._last_device_update = now

        if value is None:
            logger.warning(
                "%s: no %s value yet; treating as unsafe",
                self.name,
                self.variable,
            )
            return True

        unsafe = value < self.threshold
        if unsafe:
            logger.info(
                "%s: %s=%s below threshold %s",
                self.name,
                self.variable,
                value,
                self.threshold,
            )
        return unsafe
