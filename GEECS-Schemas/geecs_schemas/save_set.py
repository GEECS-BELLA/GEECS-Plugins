"""SaveSet — which devices (and which of their readings) get saved in a scan.

A save set is the shopping list of what to record: for each device, which
scalar readings to log and whether to save its images/files.  You would edit
one when a new diagnostic should be included in scans, or when you want an
extra readback column in the scan data.  A scan picks a save set by name in
its :class:`~geecs_schemas.scan_request.ScanRequest`.

Developer notes — what is *derived* and no longer written by hand
-----------------------------------------------------------------
The legacy save-element YAML asked the operator to declare mechanics that the
engine can derive.  In this schema they are gone, with these derivation rules
(implemented engine-side, documented here as the contract):

- ``acq_timestamp`` is **always implicit** — the device layer records it for
  every entry; listing it is unnecessary and the converter drops it.
- ``synchronous`` is derived from the entry's **role**: ``reference``,
  ``contributor``, and strict-mode triggered devices are synchronous;
  ``snapshot`` devices are asynchronous.
- **Roles themselves are derived from the acquisition mode**: in free-run
  the first non-snapshot entry becomes the ``reference`` (pacemaker) and
  later ones ``contributor``; in strict mode every non-snapshot entry is a
  triggered detector.  The optional ``role`` field on an entry is an
  *override* for the exceptional case (e.g. pinning a specific camera as the
  free-run reference, or forcing a slow device to ``snapshot``).
- ``save_nonscalar_data`` became the plain ``images`` flag.
- The legacy per-device ``scan_setup`` pre/post value pairs are expressed as
  setup/closeout :class:`~geecs_schemas.action_plan.ActionPlan` steps (the
  converter extracts them); a save set holds only *what to record*.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class SaveRole(str, Enum):
    """How a device participates in shot-by-shot bookkeeping.

    You almost never set this yourself — the scanner works it out from the
    acquisition mode.  Set it only to override the automatic choice.

    Attributes
    ----------
    REFERENCE : str
        The pacemaker in free-run mode: this device's shots define the event
        rows every other device fills in.
    CONTRIBUTOR : str
        A synchronous device that fills columns in the reference's rows
        (free-run mode).
    SNAPSHOT : str
        An asynchronous device sampled once per row — for slow readbacks
        (pressures, temperatures) that don't produce one value per shot.
    """

    REFERENCE = "reference"
    CONTRIBUTOR = "contributor"
    SNAPSHOT = "snapshot"


class SaveSetEntry(SchemaModel):
    """What to record from one device during a scan.

    Name the device, list the scalar readings you want as columns in the scan
    data, and say whether its images/files should be saved too.

    Notes
    -----
    ``acq_timestamp`` is recorded automatically for every entry — do not list
    it in ``scalars``.  ``role`` is normally left unset (derived from the
    acquisition mode; see the module docstring for the rules).
    """

    device: str = Field(
        description=(
            "GEECS device name exactly as it appears in the device database, "
            "e.g. 'UC_ALineEbeam1'. Spelling (including case) is checked "
            "against the database when the config is loaded."
        )
    )
    scalars: list[str] = Field(
        default_factory=list,
        description=(
            "Scalar readings to record as columns in the scan data, e.g. "
            "['MaxCounts', 'centroidx']. Leave empty to record only the "
            "automatic timestamp (and images, if enabled)."
        ),
    )
    all_scalars: bool = Field(
        False,
        description=(
            "Record every scalar variable the device publishes instead of "
            "naming them one by one. If 'scalars' is also given, the explicit "
            "list wins."
        ),
    )
    images: bool = Field(
        False,
        description=(
            "Save the device's images / non-scalar files (camera frames, "
            "traces) alongside the scalar data."
        ),
    )
    role: Optional[SaveRole] = Field(
        None,
        description=(
            "Override for how this device is synchronized with shots. Leave "
            "unset to let the scanner decide; set 'snapshot' for slow "
            "readbacks that don't produce one value per shot."
        ),
    )


class SaveSet(VersionedSchemaModel):
    """A named list of devices to save during a scan.

    This is what you pick in the scanner when deciding what data a scan
    records.  Edit it to add or remove diagnostics, or to change which
    readings of a device end up as columns in the scan file.

    Notes
    -----
    Successor of the legacy "save element" YAML (``Devices:`` mapping with
    ``synchronous`` / ``save_nonscalar_data`` / ``variable_list``).  See the
    module docstring for the intent→mechanics derivation rules.
    """

    name: str = Field(description="The name scans use to refer to this save set.")
    entries: list[SaveSetEntry] = Field(
        min_length=1,
        description="The devices to record, one entry per device.",
    )
    description: str = Field(
        "",
        description="Optional note about what this save set is for.",
    )

    @model_validator(mode="after")
    def _no_duplicate_devices(self) -> "SaveSet":
        """Reject two entries for the same device.

        Returns
        -------
        SaveSet
            The validated model.

        Raises
        ------
        ValueError
            If the same device name appears in more than one entry.
        """
        seen: set[str] = set()
        for entry in self.entries:
            if entry.device in seen:
                raise ValueError(
                    f"Device {entry.device!r} appears more than once in save "
                    f"set {self.name!r} — merge the entries into one."
                )
            seen.add(entry.device)
        return self
