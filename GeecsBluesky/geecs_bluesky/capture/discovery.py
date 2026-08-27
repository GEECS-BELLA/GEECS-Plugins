"""Devicetype-keyed discovery of capture-eligible cameras.

Eligibility v0: the device's GEECS devicetype appears in
``CAPTURE_DEVICE_TYPES`` (today: Point Grey cameras — ~90% of scan data by
the owner's estimate; proprietary-format devices keep the legacy per-shot
file save). The registry maps devicetype → the image variable the PVA
gateway serves for it, so PV names compose from ``geecs_core.pv_naming``
without importing any gateway code (dependency-graph rule: gateways are
consumed as services, never imported).

Uses the batch ``GeecsDb.get_experiment_device_types`` /
``get_experiment_devices`` queries — two connections total, never one per
device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from geecs_core.db.geecs_db import GeecsDb
from geecs_core.pv_naming import pv_name

from geecs_bluesky.assets.specs import POINTGREY_CAMERA_DEVICE_TYPE

logger = logging.getLogger(__name__)

# devicetype -> the GEECS image variable name the PVA gateway serves for it.
CAPTURE_DEVICE_TYPES: dict[str, str] = {
    POINTGREY_CAMERA_DEVICE_TYPE: "image",
}


@dataclass(frozen=True)
class CameraTarget:
    """One capture-eligible camera: identity plus its PVA coordinates."""

    device: str
    device_type: str
    pv: str
    server_ip: str


def discover_capture_cameras(experiment: str) -> list[CameraTarget]:
    """Return every capture-eligible camera in *experiment*, sorted by device.

    Two batched DB queries; a device whose endpoint row is missing is
    skipped with a warning (it cannot be reached over PVA anyway).
    """
    device_types = GeecsDb.get_experiment_device_types(experiment)
    endpoints = GeecsDb.get_experiment_devices(experiment)
    device_variables = GeecsDb.get_experiment_device_variables(experiment)
    targets: list[CameraTarget] = []
    for device, dtype in sorted(device_types.items()):
        image_var = CAPTURE_DEVICE_TYPES.get(dtype)
        if image_var is None:
            continue
        endpoint = endpoints.get(device)
        if endpoint is None:
            logger.warning(
                "capture-eligible device %s (%s) has no endpoint row — skipped",
                device,
                dtype,
            )
            continue
        # Loud-warning cross-check: the registry hardcodes the image variable
        # per devicetype; a device whose DB variables don't include it shows
        # the silent dead-PV signature (0 frames) at capture time.
        var_names = {m.get("name") for m in device_variables.get(device, [])}
        if var_names and image_var not in var_names:
            logger.warning(
                "capture %s: registry image variable %r not among its DB "
                "variables — expect a dead PV (0 frames) unless the registry "
                "mapping is corrected",
                device,
                image_var,
            )
        targets.append(
            CameraTarget(
                device=device,
                device_type=dtype,
                pv=pv_name(experiment, device, image_var),
                server_ip=endpoint[0],
            )
        )
    logger.info(
        "capture discovery: %d eligible cameras of %d devices in %s",
        len(targets),
        len(device_types),
        experiment,
    )
    return targets
