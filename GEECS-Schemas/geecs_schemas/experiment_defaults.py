"""ExperimentDefaults — what the experiment assumes when a scan doesn't say.

Most scans in an experiment use the same trigger profile and run the same
routine setup/closeout plans.  This file declares those once, per
experiment; a :class:`~geecs_schemas.scan_request.ScanRequest` only needs to
mention them when it wants something *different*.  You would edit it when
the experiment's routine changes — a new standard trigger profile, a new
"always run this first" checklist.

The merge rule, in plain terms: **defaults run first, then the scan's own.**
A default trigger profile is used only when the scan names none; default
setup plans run before the scan's own setup plans, and default closeout
plans run before the scan's own closeout plans.

Developer notes
---------------
There is no legacy YAML dialect behind this model — the legacy scanner kept
these choices in GUI state — so there is no converter for it.

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

    These prepend to whatever the scan itself asks for: defaults run first,
    then the scan's own plans.
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
            "Names of action plans to run after every scan, ahead of any "
            "closeout plans the scan itself lists."
        ),
    )


class ExperimentDefaults(VersionedSchemaModel):
    """Per-experiment fallbacks applied where a scan request is silent.

    Declares the trigger profile and routine action plans most scans share,
    so individual scan requests stay short.  Defaults never override what a
    scan says explicitly: a default trigger profile applies only when the
    scan names none, and default plans run first, followed by the scan's
    own.
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
            "Action plans every scan runs by default — these run first, "
            "then the scan's own plans."
        ),
    )
    apply_db_scan_defaults: bool = Field(
        True,
        description=(
            "Honor the device database's scan-start/end writes (its "
            "set='yes' start/end values) for devices taking part in a scan. "
            "On by default, matching how MC behaves; turn off to run the "
            "experiment purely from config files, ignoring the database's "
            "start/end writes everywhere."
        ),
    )
    background_telemetry: bool = Field(
        True,
        description=(
            "Log every live experiment device that is not in a scan's save "
            "set as best-effort snapshot columns, read from the gateway's "
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
