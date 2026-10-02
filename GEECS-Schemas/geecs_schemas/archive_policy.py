"""ArchivePolicy — the per-experiment curation overlay for the Archiver Appliance.

The archive set itself is *derived*: ``geecs-archiver onboard`` enumerates the
experiment's monitored (``get='yes'``) readbacks, the settable variables'
``:SP`` setpoints, every device's ``connected`` status PV and the gateway's
derived channels from the GEECS database, exactly as the CA gateway serves
them.  This document is the small, optional, committed set of *exceptions*:
PV globs to leave out, PVs to add, sampling for the few PVs that need a rate
other than the default, and the switches for the three optional classes.

It lives in the configs repository beside the gateway's own overlay::

    scanner_configs/experiments/<Experiment>/archiver/archive_policy.yaml

An absent file means "the defaults": everything the rule derives, sampled
every second on change.  This file — not the appliance's own store — is the
configuration of record together with the rule: ``onboard`` re-applies it
idempotently, so a PV someone archives by hand from the appliance's UI is
paused on the next run unless it is listed in ``include``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

SamplingMethod = Literal["MONITOR", "SCAN"]


class SamplingOverride(SchemaModel):
    """A sampling rule for the PVs matching one glob.

    Globs are ``fnmatch`` patterns over the full, lowercase PV name
    (``undulator:u_vacuumgauge:*``).  The last matching override wins per
    field.  At least one of ``sampling_period`` or ``sampling_method`` must be
    given.  The values are sent to the appliance with the archive request
    (its "user-specified sampling"), so what this file says is what the
    appliance does — there is no second table of named policies to agree with.
    """

    match: str = Field(
        ...,
        min_length=1,
        description="fnmatch glob over the full lowercase PV name.",
    )
    sampling_period: float | None = Field(
        default=None,
        gt=0,
        description="Sampling period in seconds (the appliance's samplingperiod).",
    )
    sampling_method: SamplingMethod | None = Field(
        default=None,
        description="MONITOR (store on change, throttled to the period) or SCAN (poll).",
    )

    @model_validator(mode="after")
    def _at_least_one_setting(self) -> "SamplingOverride":
        if self.sampling_period is None and self.sampling_method is None:
            raise ValueError(
                f"sampling override {self.match!r} sets nothing: give sampling_period "
                "or sampling_method"
            )
        return self


class ArchivePolicy(VersionedSchemaModel):
    """The experiment's archive-set exceptions and sampling defaults."""

    exclude: list[str] = Field(
        default_factory=list,
        description=(
            "fnmatch globs over full lowercase PV names that the derived rule "
            "would include but this experiment does not archive."
        ),
    )
    include: list[str] = Field(
        default_factory=list,
        description=(
            "Full PV names to archive beyond what the rule derives (e.g. a "
            "gateway diagnostic such as 'undulator:cagateway:devices_connected'). "
            "Listed PVs are never paused by onboarding; 'exclude' does not apply "
            "to them."
        ),
    )
    include_setpoints: bool = Field(
        default=True,
        description=(
            "Archive the ':SP' setpoint PV of every settable variable. They "
            "change only on puts and carry the operator's intent and the "
            "refused-write alarm history."
        ),
    )
    include_status: bool = Field(
        default=True,
        description="Archive every device's 'connected' status PV (state changes only).",
    )
    include_derived: bool = Field(
        default=True,
        description=(
            "Archive the gateway's derived channels declared in the experiment's "
            "gateway/derived_channels.yaml."
        ),
    )
    default_sampling_period: float = Field(
        default=1.0,
        gt=0,
        description="Sampling period (seconds) for PVs no override matches.",
    )
    default_sampling_method: SamplingMethod = Field(
        default="MONITOR",
        description="Sampling method for PVs no override matches.",
    )
    sampling_overrides: list[SamplingOverride] = Field(
        default_factory=list,
        description="Per-glob sampling rules; the last matching entry wins per field.",
    )
