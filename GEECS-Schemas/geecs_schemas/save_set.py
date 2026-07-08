"""SaveSet — the devices a scan *requires*, and what it guarantees for them.

A save set names the devices a scan cannot do without: the ones whose data
must be complete, whose images get saved, whose absence is worth a dialog or
an abort, and whose setup/closeout rituals run around the scan.  You would
edit one when a diagnostic becomes essential to a measurement — not merely
"nice to have in the log".  A scan picks a save set by name in its
:class:`~geecs_schemas.scan_request.ScanRequest`.

Required vs recorded — the two-tier model
-----------------------------------------
"Required" and "recorded" are orthogonal.  The save set is **tier 1**: the
required devices, with guarantees (strict-mode completeness, dialogs on
death, images, roles, rituals, scan-start/end overrides).  **Tier 2** is
background telemetry: every enabled experiment device with scan-logged
(get='yes') variables that is *not* in the save set is still recorded as
best-effort snapshot columns from the gateway's monitor cache — read-only,
never waited on, dropped with a log line if dead at scan start, never a
dialog or an abort.  So leaving a device out of the save set no longer means
losing its data; it means giving up the *guarantees*.  Needing a device
synchronous to shots is the definition of required — softness and
synchronicity are mutually exclusive by design.  Images are only ever
required-tier.  The per-scan/per-experiment switch lives on
:class:`~geecs_schemas.scan_request.ScanRequest` and
:class:`~geecs_schemas.experiment_defaults.ExperimentDefaults`
(``background_telemetry``).

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
- Legacy element-level ``setup_action`` / ``closeout_action`` blocks and
  per-device ``scan_setup`` pre/post pairs are extracted into **named**
  :class:`~geecs_schemas.action_plan.ActionPlan` objects; entries carry
  only name *references* to them (``setup`` / ``closeout`` fields), so
  selecting a device still brings its ritual along without inlining plans
  into the save set.
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
    """One *required* device of a scan and the guarantees it gets.

    Name the device, say whether its images/files are saved, and — beyond
    the database's standard telemetry list, which is recorded by default —
    list any extra scalar readings you want as columns.

    Notes
    -----
    ``acq_timestamp`` is recorded automatically for every entry — do not list
    it in ``scalars``.  ``role`` is normally left unset (derived from the
    acquisition mode; see the module docstring for the rules).

    ``setup`` / ``closeout`` hold action-plan *names* (references into the
    experiment's action library), never inline plans — the ritual travels
    with the device when entries are composed into bigger save sets, while
    the plan itself stays a single named, editable object.

    **Runtime contract for the DB scan defaults** (``at_scan_start`` /
    ``at_scan_end`` / ``db_scalars``): the GEECS device database's
    ``expt_device_variable`` table marks variables with ``set='yes'`` and
    carries per-variable ``startvalue`` / ``endvalue`` that MC applies at
    scan start/end, and marks scan-logged telemetry with ``get='yes'``.
    The engine reads those ``set='yes'`` start/end rows **only for devices
    participating in the scan**, always skips ``save`` /
    ``localsavingpath`` (native saving is owned by the run discipline, not
    per-device DB rows), applies this entry's ``at_scan_start`` /
    ``at_scan_end`` overrides on top, and records every write it actually
    applied into the run's metadata for provenance.  The DB rows
    themselves get no schema — device facts live below the configs.

    The soft tier (background telemetry, see the module docstring) gets
    **no** scan-start/end writes — writing to a possibly-dead device would
    block, and softness means never waiting.  Soft-tier columns carry their
    own ``acq_timestamp`` plus a validity marker for downstream alignment;
    strict-mode completeness applies to required devices only.
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
            "EXTRA scalar readings to record beyond the device's standard "
            "telemetry (which 'db_scalars' records by default), e.g. "
            "['MaxCounts', 'centroidx']. Usually empty — list variables "
            "here only when you need something the database doesn't mark "
            "for scan logging."
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
    setup: list[str] = Field(
        default_factory=list,
        description=(
            "Names of action plans that must run before any scan that "
            "records this device — its setup ritual (turn analysis on, "
            "insert a stage, ...). The plans named by all entries of a save "
            "set are collected together, de-duplicated by name, and each "
            "runs once before the scan."
        ),
    )
    closeout: list[str] = Field(
        default_factory=list,
        description=(
            "Names of action plans that run after any scan that records "
            "this device — its cleanup ritual. Collected and de-duplicated "
            "the same way as 'setup', and run once after the scan (even on "
            "abort)."
        ),
    )
    db_scalars: bool = Field(
        True,
        description=(
            "Record every variable the device database marks for scan "
            "logging (get='yes') for this device — the MC-style 'standard "
            "telemetry', and the default scalar source for a required "
            "device. The 'scalars' list adds extras on top. Turn off to "
            "record only what 'scalars' lists explicitly (converted legacy "
            "elements do this, preserving their exact old behavior)."
        ),
    )
    at_scan_start: dict[str, Optional[str]] = Field(
        default_factory=dict,
        description=(
            "Per-variable tweaks to what the device database writes to this "
            "device when a scan starts. Three cases: a variable you don't "
            "mention keeps its database behavior; 'Variable: \"value\"' "
            "sends your value instead of the database's; 'Variable: null' "
            "suppresses the database write entirely (nothing is sent)."
        ),
    )
    at_scan_end: dict[str, Optional[str]] = Field(
        default_factory=dict,
        description=(
            "Per-variable tweaks to what the device database writes to this "
            "device when a scan ends — same three cases as 'at_scan_start': "
            "unmentioned = database behavior, a value = replace the "
            "database's value, null = suppress the write entirely."
        ),
    )


class SaveSet(VersionedSchemaModel):
    """The devices a scan *requires* — its participation list, not a logging list.

    Putting a device here is a statement of need: the scan guarantees its
    data (strict-mode completeness, dialogs when it dies), saves its images
    if asked, gives it a shot-synchronization role, runs its setup/closeout
    rituals, and applies its scan-start/end overrides.  Devices left out
    are still logged as background telemetry when alive (see the module
    docstring) — so edit this file when a device becomes *essential* to a
    measurement, not merely worth keeping in the log.

    Notes
    -----
    Successor of the legacy "save element" YAML (``Devices:`` mapping with
    ``synchronous`` / ``save_nonscalar_data`` / ``variable_list``).  See the
    module docstring for the intent→mechanics derivation rules and the
    required-vs-recorded two-tier model.
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
