"""GUI launcher for the windmill PM interlock configuration."""

import logging
import os

from dotenv import load_dotenv

from geecs_python_api.controls.interlocks import (
    InterlockConstructor,
    MultiCheck,
    ThresholdCheck,
    launch_interlock_gui,
)
from geecs_python_api.controls.interlocks.monitor_conditions import (
    ConfigurationChangeCheck,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()


def build_windmill_interlock() -> tuple[InterlockConstructor, list]:
    """Build the same windmill checks used by ``windmill_interlock.py``."""
    interlock = InterlockConstructor(
        "BELLA",
        host=os.getenv("SERVER_IP"),
        port=int(os.environ["SERVER_PORT"]),
        staleness_timeout_ms=5000,
    )

    tape_pointing = interlock.add_device(
        "tapePointing",
        "CAM-PL1-TapeDrivePointing",
        ["MeanCounts", "acq_timestamp", "exposure", "gain"],
    )
    aegaeon_cam = interlock.add_device(
        "AegaeonCam",
        "CAM-2BL-Aegeaon",
        ["MeanCounts", "acq_timestamp", "exposure", "gain"],
    )
    tape_pointing.use_alias_in_TCP_subscription = False
    aegaeon_cam.use_alias_in_TCP_subscription = False

    tape_check = ThresholdCheck(
        interlock.device_group, "tapePointing", "MeanCounts", 4, operator="<"
    )
    wiper_check = ThresholdCheck(
        interlock.device_group, "AegaeonCam", "MeanCounts", 5, operator=">"
    )
    tape_config = ConfigurationChangeCheck(
        interlock.device_group, "tapePointing", ["exposure", "gain"]
    )
    aegaeon_config = ConfigurationChangeCheck(
        interlock.device_group, "AegaeonCam", ["exposure", "gain"]
    )
    interlock.add_monitor(
        "Camera Multi Check",
        MultiCheck([tape_check, wiper_check, tape_config, aegaeon_config]),
        interval=0.1,
    )
    return interlock, [tape_config, aegaeon_config]


if __name__ == "__main__":
    configured_interlock, armable_checks = build_windmill_interlock()
    launch_interlock_gui(
        configured_interlock,
        armable_checks,
        title="Windmill PM Interlock",
    )
