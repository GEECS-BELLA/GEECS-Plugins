"""ExperimentDefaults — what the experiment assumes when a scan doesn't say.

Most scans in an experiment use the same trigger profile.  This file
declares it once, per experiment, with the experiment-wide switches (the
LabVIEW-files dual-write, the background telemetry); a preset only needs to
mention them when it wants something *different*.  A default applies only
where the scan is silent.

Developer notes
---------------
There is no legacy YAML dialect behind this model — the legacy scanner kept
these choices in GUI state — so there is no converter for it.  A named
device state to apply before or after scans is a setup
(:mod:`geecs_schemas.setup`), applied as a queue item of its own; this file
names none.

DB scan defaults — what is honored
----------------------------------
The get-side DB defaults (``background_telemetry``, and per-device
``db_scalars`` on the save set) ARE honored.  The set-side
(``apply_db_scan_defaults`` — the database's ``set='yes'`` scan start/end
writes) is **reserved and not honored**: the engine never writes those
rows itself.  The *device* applies them, on the ``ScanStart`` / ``ScanEnd``
commands of GEECS scan mode — so under a trigger profile with
``use_scan_mode`` they take effect exactly as under Master Control, and
without one they do not; this flag changes neither.  Kept for a possible
future engine-side application.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import Field

from geecs_schemas._base import Tier, VersionedSchemaModel

#: An experiment's defaults file, at the top of its configs folder.
EXPERIMENT_DEFAULTS_FILENAME = "experiment_defaults.yaml"


class ExperimentDefaults(VersionedSchemaModel):
    """Per-experiment fallbacks applied where a scan request is silent.

    Declares the trigger profile most scans share, so individual presets
    stay short.  Defaults never override what a scan says explicitly: a
    default trigger profile applies only when the scan names none.
    """

    tier: ClassVar[Tier] = "operator"

    trigger_profile: Optional[str] = Field(
        None,
        description=(
            "Name of the trigger profile to use when a scan doesn't name "
            "one. Leave unset if scans must always choose explicitly."
        ),
    )
    native_image_save: bool = Field(
        True,
        description=(
            "Whether cameras whose frames the PVA gateway's file plugin "
            "captures also write their LabVIEW per-shot files (PNGs) when a "
            "scan does not say (the preset's 'native_image_save' unset). On "
            "by default: the dual-write is the rollout's parity evidence "
            "until PNG retirement (GEECS-Plugins#738). Set false to run the "
            "experiment on the plugin's HDF5 stacks alone; a device without "
            "a file plugin always keeps its native files. The worker reads "
            "this at every scan, so an edit takes effect at the next scan "
            "without reopening its environment."
        ),
    )
    apply_db_scan_defaults: bool = Field(
        True,
        description=(
            "RESERVED AND NOT HONORED. The DB's scan start/end writes (MySQL "
            "table expt_device_variable: rows with set='yes', their "
            "startvalue/endvalue) are applied by the DEVICE itself on the "
            "ScanStart/ScanEnd commands of GEECS scan mode, never by the "
            "engine: under a trigger profile with use_scan_mode they take "
            "effect exactly as under Master Control, and without one they do "
            "not. This flag changes neither. Kept for a possible future "
            "engine-side application. This is the set-side only: the "
            "get-side 'db_scalars' (standard telemetry) and "
            "'background_telemetry' are separate and remain active."
        ),
    )
    background_telemetry: bool = Field(
        True,
        description=(
            "Read every logged scalar of the experiment that is not in a "
            "scan's device group — the variables the GEECS experiment "
            "database marks for scan logging (MySQL table "
            "expt_device_variable, get='yes') — into every row of the scan "
            "as well, the way Master Control did: softly, from the "
            "gateway's monitor cache, never waited on. A device that does "
            "not answer when the scan starts is left out of that scan with "
            "a log line (and named in the run's start document), one the "
            "gateway marks INVALID reads NaN; nothing here can stall or fail "
            "a scan. On by default so no data is silently lost; a preset can "
            "override it with its own 'background_telemetry'. The worker "
            "reads this at every scan, so an edit takes effect at the next "
            "scan without reopening its environment."
        ),
    )
    description: str = Field(
        "",
        description="Optional note about what these defaults are for.",
    )
