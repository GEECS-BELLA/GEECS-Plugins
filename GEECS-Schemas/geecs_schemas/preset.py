"""Preset — a saved scan: the device group plus the plan call.

The same scan is run often, so it is saved once: which devices record
(and whether each saves its images), which stock plan runs with which
arguments, under which trigger profile.  A preset **is** a queue item in
waiting — the client expands it into ``plan(detectors, *args, **kwargs)``
against the worker's device namespace at submission and the plan's
arguments are the scan's one description (GEECS-Plugins#807, plan of record
§4.D, §10.5).  You would edit one when a routine measurement changes
shape: a new diagnostic joins the group, the sweep range moves, a camera's
frames stop being worth the disk.

Successor of the save set (the device grouping) and the scan request (the
plan parameters): both concepts survive, folded into one document.  What
did **not** survive, deliberately: per-scalar selection (every subscribed
scalar of every device in the run is recorded, and the s-file carries all
of them), and setup/closeout rituals and the ``SaveRole`` enum (an
explicit action plan is its own queue item).  What phase 2 added
(GEECS-Plugins#807, ``08_gated_batch.md`` §4.6): ``essential`` on each
device — an essential device is waited on every shot, a non-essential
one streams its frames for the run and never holds a shot — and the
acquisition mode as a plan keyword (``acquisition: gated`` in
``plan.kwargs``, beside ``shots_per_step``), not a preset field.

Device references
-----------------
``devices[].device`` is the GEECS device name; the client turns it into the
namespace binding (``UC_Amp4_IR_input``, or ``UC_Amp4_IR_input.scalars``
when ``save_images`` is off — the detector's scalars-only view).  Inside
``plan.args`` / ``plan.kwargs`` a scan variable is a **string**: a
``Device:Variable`` pair, or a name from the experiment's scan-variable
catalog; the client resolves it to the namespace's Movable child
(``U_S1H.current``) at submission.  Everything else is a plain JSON value.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, JsonValue, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class PresetDevice(SchemaModel):
    """One device of the group: it records every shot of the scan.

    ``essential`` (default on) means the scan waits for this device on
    every shot — a shot is not complete without its reading.  Off means
    the device streams what it produces for the run's duration (its
    frames, through the camera server's file plugin) and never holds a
    shot or aborts a run: the choice for a slow or unreliable camera whose
    frames are welcome but not required (phase 2, GEECS-Plugins#807).
    """

    device: str = Field(
        min_length=1,
        description=(
            "GEECS device name exactly as it appears in the GEECS experiment "
            "database (MySQL), e.g. 'UC_ALineEbeam1'."
        ),
    )
    save_images: bool = Field(
        True,
        description=(
            "Save the device's images / non-scalar files (camera frames, "
            "traces) beside the scalar data. Off records the device's "
            "scalars only — its per-shot readings still land in every row; "
            "the frames stay off the disk. Meaningless for a scalar-only "
            "device (nothing to save either way)."
        ),
    )
    essential: bool = Field(
        True,
        description=(
            "Wait for this device on every shot (on, the default) — a shot is "
            "not complete without its reading. Off streams the device's "
            "frames for the run's duration instead: it never holds a shot or "
            "aborts the scan, so use it for a slow or unreliable camera whose "
            "frames are welcome but not required. Off needs the images saved "
            "(a scalars-only device cannot stream)."
        ),
    )


class PlanCall(SchemaModel):
    """The stock plan the preset runs, with its arguments.

    ``detectors`` (the plan's first argument) is **not** listed here: the
    client builds it from ``devices``.  ``args`` and ``kwargs`` are the
    rest of the stock signature — ``scan``'s ``[motor, start, stop, num]``,
    ``count``'s ``{num: 100}`` — plus the GEECS keyword arguments the
    worker adds to every scan verb: ``shots_per_step`` (rows per position)
    and ``trigger_profile`` (normally the preset's own field).
    """

    name: str = Field(
        min_length=1,
        description=(
            "The stock bluesky plan to run, e.g. 'count', 'scan', 'list_scan', "
            "'grid_scan' — one of the names the worker registers."
        ),
    )
    args: list[JsonValue] = Field(
        default_factory=list,
        description=(
            "Positional arguments after the detectors, in the plan's own "
            "order — e.g. ['EMQ1 Current', 1.2, 1.7, 6] for scan (motor, "
            "start, stop, number of points). A scan variable is a string: "
            "'Device:Variable' or a scan-variable catalog name."
        ),
    )
    kwargs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments — e.g. {num: 100} for count, {shots_per_step: "
            "10} for the scan verbs (rows recorded at every position)."
        ),
    )


class Preset(VersionedSchemaModel):
    """A saved scan: the device group plus the plan call.

    Submitting a preset runs ``plan.name`` with the group as its detectors
    and ``plan.args`` / ``plan.kwargs`` as the rest — a single queue item.
    A preset without a ``plan`` is a device group waiting for one: it is
    listed and editable but cannot be submitted as is.
    """

    name: str = Field(description="The name clients use to pick this preset.")
    description: str = Field(
        "",
        description=(
            "What this scan is for — becomes the run's description "
            "(ScanStartInfo in the legacy ScanInfo file)."
        ),
    )
    trigger_profile: Optional[str] = Field(
        None,
        description=(
            "Name of the trigger profile driving the shots. Leave unset to "
            "use the experiment default (experiment_defaults.yaml)."
        ),
    )
    background: bool = Field(
        False,
        description=(
            "Flag this scan as a background measurement (metadata only: "
            "ScanMode 'background' in ScanInfo, 'background' in the run)."
        ),
    )
    devices: list[PresetDevice] = Field(
        default_factory=list,
        description=(
            "The devices recording this scan, one entry per device. Every "
            "subscribed scalar of each is a column of every row; each entry "
            "chooses whether its images are saved. An empty list scans with "
            "the motors' readbacks only."
        ),
    )
    plan: Optional[PlanCall] = Field(
        None,
        description=(
            "The stock plan call this preset submits. Unset means a device "
            "group with no scan attached yet."
        ),
    )

    @model_validator(mode="after")
    def _no_duplicate_devices(self) -> "Preset":
        """Reject two entries for the same device.

        Returns
        -------
        Preset
            The validated model.

        Raises
        ------
        ValueError
            If the same device name appears in more than one entry.
        """
        seen: set[str] = set()
        for entry in self.devices:
            if entry.device in seen:
                raise ValueError(
                    f"Device {entry.device!r} appears more than once in preset "
                    f"{self.name!r} — keep one entry."
                )
            seen.add(entry.device)
        return self
