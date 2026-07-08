"""ExperimentDefaults — what the experiment assumes when a scan doesn't say.

Most scans in an experiment use the same trigger profile and run the same
routine setup/closeout plans.  This file declares those once, per
experiment; a :class:`~geecs_schemas.scan_request.ScanRequest` only needs to
mention them when it wants something *different*.  You would edit it when
the experiment's routine changes — a new standard trigger profile, a new
"always run this first" checklist.

The merge rule, in plain terms: **defaults run first on the way in and last
on the way out.**  A default trigger profile is used only when the scan
names none; default setup plans run before the scan's own setup plans, and
default closeout plans run *after* the scan's own closeout plans — teardown
mirrors setup, so the experiment-wide baseline is the outermost bracket
around every scan.

Developer notes
---------------
There is no legacy YAML dialect behind this model — the legacy scanner kept
these choices in GUI state — so there is no converter for it.

The mirrored closeout ordering is deliberate (ratified with the action
execution milestone): the four setup layers (vision doc §4.4b) nest like
context managers.  On the way in the order is defaults → save-set entry
rituals → the scan's own setup; on the way out it is the exact reverse —
the scan's own closeout → entry rituals → defaults.  A defaults closeout
like "return the machine to standby" therefore always runs last, after
every scan-specific cleanup has finished.

Resolvers MUST record the defaults they applied into the resolved request
(provenance): a run's metadata has to show the trigger profile and action
plans it *actually* used, not require the reader to reconstruct which
defaults file was in force at the time.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class DefaultActions(SchemaModel):
    """The action plans every scan of the experiment runs by default.

    These bracket whatever the scan itself asks for: default setup runs
    first, then the scan's own plans; default closeout runs last, after the
    scan's own — teardown mirrors setup.
    """

    setup: list[str] = Field(
        default_factory=list,
        description=(
            "Names of action plans to run before every scan, ahead of any "
            "setup plans the scan itself lists."
        ),
    )
    closeout: list[str] = Field(
        default_factory=list,
        description=(
            "Names of action plans to run after every scan, after any "
            "closeout plans the scan itself lists (teardown mirrors setup: "
            "these are the outermost bracket)."
        ),
    )


class ExperimentDefaults(VersionedSchemaModel):
    """Per-experiment fallbacks applied where a scan request is silent.

    Declares the trigger profile and routine action plans most scans share,
    so individual scan requests stay short.  Defaults never override what a
    scan says explicitly: a default trigger profile applies only when the
    scan names none, and default plans bracket the scan's own — defaults
    run first on setup and last on closeout.
    """

    trigger_profile: Optional[str] = Field(
        None,
        description=(
            "Name of the trigger profile to use when a scan doesn't name "
            "one. Leave unset if scans must always choose explicitly."
        ),
    )
    actions: DefaultActions = Field(
        default_factory=DefaultActions,
        description=(
            "Action plans every scan runs by default — setup plans run first "
            "(before the scan's own), closeout plans run last (after the "
            "scan's own)."
        ),
    )
    apply_db_scan_defaults: bool = Field(
        True,
        description=(
            "Honor the GEECS experiment database's scan-start/end writes "
            "(MySQL table expt_device_variable: rows with set='yes', "
            "writing their startvalue/endvalue) for devices taking part in "
            "a scan. On by default, matching how MC behaves; turn off to "
            "run the experiment purely from config files, ignoring the "
            "database's start/end writes everywhere."
        ),
    )
    background_telemetry: bool = Field(
        True,
        description=(
            "Log every live experiment device that is not in a scan's save "
            "set as best-effort snapshot columns — the variables the GEECS "
            "experiment database marks for scan logging (MySQL table "
            "expt_device_variable, get='yes') — read from the gateway's "
            "always-on monitor cache. Safe by construction: read-only and "
            "never waited on, so it cannot slow or stall a scan — a dead "
            "device is just dropped with a log line. On by default so no "
            "data is silently lost; individual scans can override with "
            "their own 'background_telemetry' setting."
        ),
    )
    description: str = Field(
        "",
        description="Optional note about what these defaults are for.",
    )
