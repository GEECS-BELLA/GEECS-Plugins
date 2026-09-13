"""ShotOffsets — each device's edge-to-stamp latency, measured once.

A GEECS device's ``acq_timestamp`` is the shot's identity in everything but
name, but two devices stamp the *same* shot at different times: the stamp is
the trigger's arrival plus however long that device took to drain the frame,
a per-device constant (``03_clean_room_rebuild.md`` §11.3/§11.4).  The
constant differs by tens of milliseconds across a camera set — 36 ms between
two amplifier cameras, near 100 ms across a set mixing a 4 MB camera with a
0.5 MB one.

Joining frames to shot rows therefore corrects each side by its own offset
before matching (``geecs_data_utils.shot_join``).  Until this document
exists every offset reads ``0.0``, which is harmless only while the join
windows are wide: at 1 Hz they are ±0.5 s and swallow the spread, but they
narrow with the rep rate, and at 5 Hz they are ±0.1 s — the same order as
the spread itself, where an uncalibrated offset costs rows.  Measuring these
numbers is what makes faster running safe.

This document is what the ``measure_shot_offsets`` plan writes and what the
worker reads at startup to populate each detector's ``drain_offset`` config
signal.  It is per experiment, and it is written by a *measurement*, not by
hand — edit it only to correct a mistake.

Only differences matter
-----------------------
The join subtracts each device's offset from its stamp before matching, so
adding the same constant to every offset changes nothing.  The offsets are
therefore anchored: :attr:`ShotOffsets.reference` names the device measured
as stamping first, whose offset is ``0.0`` by construction, and every other
device's :attr:`DeviceOffset.offset_s` says how long *after* that device it
stamps the same shot.  A later re-measurement may pick a different
reference without any of the numbers meaning something different.

Why the scatter is recorded
---------------------------
A device's offset is not perfectly steady: each machine's clock dithers
around its average by up to ~10 ms (a consequence of the host's own
timekeeping — higher-end boxes hold ~1 ms), while the domain keeps the
averages on a common target.  One shot therefore estimates the offset only
to that precision, so the plan averages several shots and records the
observed scatter alongside the mean.  The scatter is evidence in its own
right: a device whose scatter is much larger than its peers' has a
timekeeping problem that an average would hide.

Developer notes
---------------
There is no legacy YAML dialect behind this model — the legacy scanner had
no stored calibration at all (the sync ritual was run by hand and the
numbers lived in the operator's head), so there is no converter for it.
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class DeviceOffset(SchemaModel):
    """One device's measured edge-to-stamp latency, relative to the reference."""

    offset_s: float = Field(
        description=(
            "Seconds after the reference device that this device stamps the "
            "same shot — the mean over the shots that contributed. The "
            "reference device's own value is 0.0. Subtracted from this "
            "device's acq_timestamp before frames are matched to shot rows."
        )
    )
    scatter_s: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Peak-to-peak spread of this device's per-shot offset across the "
            "measurement, seconds. Expect up to ~10 ms from ordinary host "
            "clock dither; markedly more than the other devices in the set "
            "means this machine's timekeeping is worth looking at."
        ),
    )
    shots: int = Field(
        1,
        ge=1,
        description=(
            "How many shots contributed to the mean. Fewer than the "
            "measurement requested means this device missed shots."
        ),
    )
    geecs_device: str = Field(
        "",
        description=(
            "GEECS device name this offset was measured for, e.g. "
            "'UC_Amp3_IR_input'. Informational: the mapping key is the ophyd "
            "object name the runtime uses."
        ),
    )

    @model_validator(mode="after")
    def _finite(self) -> "DeviceOffset":
        if not math.isfinite(self.offset_s):
            raise ValueError(f"offset_s must be a finite number, got {self.offset_s}")
        if not math.isfinite(self.scatter_s):
            raise ValueError(f"scatter_s must be a finite number, got {self.scatter_s}")
        return self


class ShotOffsets(VersionedSchemaModel):
    """The experiment's measured per-device drain offsets.

    Written by the ``measure_shot_offsets`` calibration plan and read at
    worker startup to populate every detector's ``drain_offset`` config
    signal, which rides in each run's descriptors and is what the s-file
    join corrects by.
    """

    reference: str = Field(
        description=(
            "The ophyd object name of the device the offsets are measured "
            "against — the one that stamped first. Its own offset_s is 0.0. "
            "Only differences matter, so which device this is carries no "
            "meaning beyond anchoring the numbers."
        )
    )
    devices: dict[str, DeviceOffset] = Field(
        default_factory=dict,
        description=(
            "Ophyd object name (e.g. 'uc_amp3_ir_input') → that device's "
            "measured offset. A device absent from this mapping keeps the "
            "0.0 default, which is correct only if it really stamps with the "
            "reference."
        ),
    )
    measured_at: Optional[str] = Field(
        None,
        description=(
            "ISO-8601 timestamp of the measurement, with offset. "
            "Informational, but the thing to look at when a join goes wrong: "
            "a calibration older than the last camera or server change is "
            "suspect."
        ),
    )
    trigger_profile: Optional[str] = Field(
        None,
        description=(
            "Trigger profile the measurement fired through. Recorded because "
            "a profile that drives a different trigger box would measure "
            "different latencies."
        ),
    )
    description: str = Field(
        "",
        description="Optional note about this measurement.",
    )

    @model_validator(mode="after")
    def _reference_is_consistent(self) -> "ShotOffsets":
        """The reference must be listed, and must be the zero.

        A document whose reference names a device it does not list, or whose
        reference carries a non-zero offset, is not self-consistent: readers
        would anchor the set differently from the measurement that wrote it.
        """
        entry = self.devices.get(self.reference)
        if entry is None:
            listed = ", ".join(sorted(self.devices)) or "none"
            raise ValueError(
                f"reference {self.reference!r} is not among the measured "
                f"devices ({listed})"
            )
        if entry.offset_s != 0.0:
            raise ValueError(
                f"reference {self.reference!r} must have offset_s 0.0 "
                f"(it is the device the others are measured against), "
                f"got {entry.offset_s}"
            )
        return self

    def offset_for(self, object_name: str) -> float:
        """The device's offset in seconds; ``0.0`` when it was not measured."""
        entry = self.devices.get(object_name)
        return entry.offset_s if entry is not None else 0.0
