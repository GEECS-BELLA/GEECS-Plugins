"""NativeDevices — the experiment's devices served by native EPICS IOCs.

A device can be served by its own EPICS IOC (geecs-epics) instead of a
LabVIEW device app.  This registry names those devices, per experiment, in
the configs repository::

    scanner_configs/experiments/<Experiment>/native_devices.yaml

One owner per device: a name listed here must not also be an enabled device
of the experiment in the GEECS DB (the scan worker refuses to open
otherwise).  The registry carries names, kind and trigger source only — the
PV prefix is derived from the experiment and device name
(``<experiment>:<device>:``, GEECS normalization), and hosts and paths are
facility values with a home of their own.  An absent file means the
experiment has no native devices.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, field_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

#: The registry's file name in the experiment's configs folder.
NATIVE_DEVICES_FILENAME = "native_devices.yaml"


class AravisCamera(SchemaModel):
    """A GigE Vision camera behind ADAravis, writing HDF5 through its IOC's own file plugin."""

    kind: Literal["aravis_camera"] = Field(
        description="Device kind: 'aravis_camera' (an areaDetector ADAravis camera)."
    )
    trigger_source: Literal["line0", "line1", "software"] = Field(
        description=(
            "Where each frame comes from. 'line0' / 'line1': the camera's "
            "hardware input on that line (a Basler's opto-isolated input is "
            "Line1, a FLIR Blackfly's Line0), wired to a timing-system output "
            "gated by the trigger box; the line must be one of the camera's "
            "own TriggerSource choices. 'software': the scan worker presses "
            "the camera's software trigger at each shot (exposed when the "
            "worker says go, not when the laser fires)."
        )
    )


#: Every native device kind, discriminated by ``kind`` (a new kind is one more member).
NativeDevice = Annotated[Union[AravisCamera], Field(discriminator="kind")]


class NativeDevices(VersionedSchemaModel):
    """The experiment's devices served by native EPICS IOCs, by GEECS device name."""

    schema_version: Literal[1] = Field(
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
    devices: dict[str, NativeDevice] = Field(
        default_factory=dict,
        description=(
            "GEECS device name (e.g. 'UC_ExampleCam') → its kind and "
            "settings. The IOC serves it under the derived prefix "
            "'<experiment>:<device>:'; the device must not also be enabled "
            "in the GEECS DB."
        ),
    )

    @field_validator("devices")
    @classmethod
    def _names_are_names(
        cls, devices: dict[str, NativeDevice]
    ) -> dict[str, NativeDevice]:
        blank = [name for name in devices if not name.strip()]
        if blank:
            raise ValueError("a native device name must not be blank")
        return devices
