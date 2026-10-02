"""The rule: which of an experiment's PVs the appliance archives, and how.

The set is *derived* from the GEECS database with the same three batched
queries the CA gateway builds its served set from — the two can only drift
if one of them changes its rule, and onboarding's stuck-request check
(:func:`geecs_archiver.onboard.verify`, the appliance's never-connected
list) makes that loud.  Per enabled device:

* the **readback** PV of every monitored (``get='yes'``) scalar variable;
* the **``:SP`` setpoint** of every settable scalar variable (they change
  only on puts and carry the operator's intent) — ``include_setpoints``;
* the device's **``connected`` status** PV — ``include_status``;

plus the gateway's **derived channels** — ``include_derived`` — and the
policy's explicit ``include`` PVs.  Never: the timestamp variables (the
intrinsic pair and any ``…timestamp``: pure disk burn), image / array
variables (not scalar CA data) and ``path``-typed long strings (the
appliance cannot type the gateway's char-array channels; pilot 2026-10-02).
The experiment's :class:`~geecs_schemas.ArchivePolicy` then removes its
``exclude`` globs and sets per-glob sampling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

from geecs_core.db.variable_types import (
    TIMESTAMP_LADDER,
    VARTYPE_TO_DTYPE,
    effective_vartype,
    is_scalar_vartype,
)
from geecs_core.pv_naming import (
    DEVICE_STATUS_VARIABLE,
    device_status_pv,
    pv_name,
    setpoint_pv,
)
from geecs_schemas import ArchivePolicy, DerivedChannels

Kind = Literal["readback", "setpoint", "status", "derived", "include"]

#: The intrinsic per-device timestamps the gateways subscribe (GEECS-Core's ladder).
TIMESTAMP_VARIABLES: frozenset[str] = frozenset(TIMESTAMP_LADDER)
#: Served dtypes the appliance cannot archive from this gateway.
UNARCHIVABLE_DTYPES: frozenset[str] = frozenset({"path"})


@dataclass(frozen=True, order=True)
class ArchiveCandidate:
    """One PV the rule wants archived, with where it came from."""

    pv: str
    device: str
    variable: str
    kind: Kind
    dtype: str


@dataclass(frozen=True)
class Sampling:
    """How one PV is sampled: the appliance's ``samplingperiod`` / ``samplingmethod``.

    Sent with every archive request as the appliance's *user-specified*
    sampling, which takes precedence over its policy file — so the overlay is
    the one table, and what it says is what the appliance does.
    """

    period: float
    method: str

    def request(self, pv: str) -> dict[str, str]:
        """The ``archivePV`` request body entry for *pv*."""
        return {
            "pv": pv,
            "samplingperiod": f"{self.period:g}",
            "samplingmethod": self.method,
        }


def is_timestamp_variable(name: str) -> bool:
    """Whether *name* is a timestamp variable: the intrinsic two, or any device's own ``…timestamp``.

    They advance every frame by design, so archiving them is pure disk burn
    (HTU's monitoring set carries three devices' plain ``timestamp`` beside the
    intrinsic pair).
    """
    lowered = name.lower()
    return lowered in TIMESTAMP_VARIABLES or lowered.endswith("timestamp")


def classify(row: Mapping[str, object]) -> str | None:
    """The served dtype of a DB variable row, or ``None`` when it is not scalar CA data."""
    effective = effective_vartype(
        row.get("variabletype"),  # type: ignore[arg-type]
        row.get("choices"),  # type: ignore[arg-type]
    )
    if not is_scalar_vartype(effective):
        return None
    return VARTYPE_TO_DTYPE.get(effective, "float")


def is_excluded(pv: str, policy: ArchivePolicy) -> bool:
    """Whether *pv* matches one of the policy's ``exclude`` globs (case-folded)."""
    lowered = pv.lower()
    return any(fnmatchcase(lowered, glob.lower()) for glob in policy.exclude)


def sampling_for(pv: str, policy: ArchivePolicy) -> Sampling:
    """The sampling for *pv*: the defaults, then every matching override in order (last wins per field)."""
    period = policy.default_sampling_period
    method = policy.default_sampling_method
    lowered = pv.lower()
    for override in policy.sampling_overrides:
        if not fnmatchcase(lowered, override.match.lower()):
            continue
        if override.sampling_period is not None:
            period = override.sampling_period
        if override.sampling_method is not None:
            method = override.sampling_method
    return Sampling(period=period, method=method)


def derive_candidates(
    experiment: str,
    endpoints: Mapping[str, object],
    var_map: Mapping[str, Sequence[Mapping[str, object]]],
    sub_map: Mapping[str, Iterable[str]],
    *,
    policy: ArchivePolicy,
    derived: DerivedChannels | None = None,
) -> list[ArchiveCandidate]:
    """Apply the rule to already-fetched DB data (the pure, network-free core).

    Parameters
    ----------
    experiment : str
        GEECS experiment name; the PV namespace prefix.
    endpoints : mapping
        ``{device: (host, port)}`` — the enabled devices
        (``GeecsDb.get_experiment_devices``).  Only the keys matter here.
    var_map : mapping
        ``{device: [row, ...]}`` with ``name``, ``settable``, ``variabletype``,
        ``choices`` per row (``GeecsDb.get_experiment_device_variables``).
    sub_map : mapping
        ``{device: [variable, ...]}`` of ``get='yes'`` variables
        (``GeecsDb.get_subscribed_variables``).
    policy : ArchivePolicy
        The experiment's overlay (defaults when it has none).
    derived : DerivedChannels, optional
        The gateway's derived-channel overlay.

    Returns
    -------
    list of ArchiveCandidate
        Sorted by PV name, after the policy's ``exclude`` globs; the policy's
        explicit ``include`` PVs are appended regardless of the globs.
    """
    out: list[ArchiveCandidate] = []
    for device in sorted(endpoints):
        monitored = set(sub_map.get(device, ()))
        seen: set[str] = set()
        serves_anything = False
        for row in var_map.get(device, ()):
            name = str(row["name"])
            if name in seen:  # the DB can list a variable more than once
                continue
            seen.add(name)
            settable = bool(row.get("settable", False))
            is_monitored = name in monitored
            if not (is_monitored or settable):
                continue  # the gateway serves the get-list plus the control surface
            dtype = classify(row)
            if dtype is None:
                continue  # image / array: not scalar CA data
            serves_anything = True
            if is_timestamp_variable(name) or dtype in UNARCHIVABLE_DTYPES:
                continue
            readback = pv_name(experiment, device, name)
            if is_monitored:
                out.append(ArchiveCandidate(readback, device, name, "readback", dtype))
            if settable and policy.include_setpoints:
                out.append(
                    ArchiveCandidate(
                        setpoint_pv(readback), device, name, "setpoint", dtype
                    )
                )
        # The gateway skips a device that exposes nothing, status PV included.
        if serves_anything and policy.include_status:
            out.append(
                ArchiveCandidate(
                    device_status_pv(experiment, device),
                    device,
                    DEVICE_STATUS_VARIABLE,
                    "status",
                    "enum",
                )
            )
    if derived is not None and policy.include_derived:
        for channel in derived.derived_channels:
            out.append(
                ArchiveCandidate(
                    pv_name(*channel.pv_parts(experiment)),
                    channel.device,
                    channel.pv or channel.variable,
                    "derived",
                    "float",
                )
            )
    kept = {c.pv: c for c in out if not is_excluded(c.pv, policy)}
    for pv in policy.include:
        kept.setdefault(pv, ArchiveCandidate(pv, "", "", "include", "unknown"))
    return sorted(kept.values())


def build_archive_set(
    experiment: str,
    *,
    policy: ArchivePolicy,
    derived: DerivedChannels | None = None,
    enabled_only: bool = True,
) -> list[ArchiveCandidate]:
    """Derive the experiment's archive set live from the GEECS database."""
    from geecs_core.db.geecs_db import GeecsDb  # lazy: the DB client needs lab access

    endpoints = GeecsDb.get_experiment_devices(experiment, enabled_only=enabled_only)
    var_map = GeecsDb.get_experiment_device_variables(
        experiment, enabled_only=enabled_only
    )
    sub_map = GeecsDb.get_subscribed_variables(experiment, enabled_only=enabled_only)
    return derive_candidates(
        experiment, endpoints, var_map, sub_map, policy=policy, derived=derived
    )
