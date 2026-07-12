"""Declarative configuration for the GEECS → EPICS CA gateway.

The gateway is driven entirely by a :class:`GatewayConfig`.  Production
configs are generated from the GEECS experiment database
(:meth:`GatewayConfig.from_geecs_experiment`, pulling per-variable
units/limits/dtype); hand-built configs remain possible (see ``demo.py``).
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .alarms import AlarmLimits
from .derived import DerivedChannelSpec
from .naming import normalize_pv_component

logger = logging.getLogger(__name__)

DType = Literal["float", "int", "string", "path", "enum"]

# EPICS DBR_STRING (the .DESC field's type) caps at 40 characters. A longer
# description is truncated by CA clients, so we clip it at the source with a
# warning rather than silently shipping something that displays wrong.
_MAX_DESC_LENGTH = 40

# GEECS `variabletype` → gateway dtype. Types absent here (image, 1darray) are
# not scalar CA data and are skipped when building specs from the DB.
# `path` is distinct from `string`: EPICS DBR_STRING caps at 40 characters, so
# path variables (file/save paths routinely exceed that) are served as
# char-array PVs — the standard EPICS long-string convention (areaDetector
# FilePath does the same). A plain `string` stays a native 40-char string PV.
_VARTYPE_TO_DTYPE: dict[str, DType] = {
    "numeric": "float",
    "string": "string",
    "path": "path",
    "choice": "enum",
}
_SKIP_VARTYPES = {"image", "1darray"}

# EPICS DBR_ENUM limits: at most 16 states, each label at most 26 chars. A GEECS
# `choice` that exceeds either can't be a CA enum, so it degrades to a string PV
# (the option value still round-trips as text; only the dropdown is lost).
_MAX_ENUM_STATES = 16
_MAX_ENUM_STRING_LEN = 26

# `choices` values that are bare type descriptors rather than option lists (the
# `choice` table's low IDs double as type descriptors in the GEECS DB).
_CHOICE_TYPE_DESCRIPTORS = _SKIP_VARTYPES | {"numeric", "string", "path"}


def effective_vartype(variabletype: str | None, choices: str | None) -> str:
    """Resolve the effective GEECS variable type from DB metadata.

    The single source of truth for how the gateway interprets a DB variable
    row's type — used by :meth:`DeviceSpec.from_db_metadata` when building
    served specs *and* by the DB-hygiene audit
    (:func:`geecs_ca_gateway.audit.audit_subscribed_variables`), so the audit
    flags exactly what the gateway skips (#512).

    The ``choice`` table's low IDs double as type descriptors, so when
    ``choices`` is a bare descriptor word (``image``, ``1darray``, ``numeric``,
    ``string``, ``path``) it is the AUTHORITATIVE type — even when
    ``variabletype`` says otherwise (e.g. ``variabletype='choice'`` with
    ``choices='image'`` is an image variable streaming raw bytes, not a
    one-option enum).  Otherwise trust ``variabletype``; if it is blank, a real
    option list is a ``choice``, else fall back to ``numeric``.

    Parameters
    ----------
    variabletype : str or None
        The DB ``variabletype`` column (may be blank/None).
    choices : str or None
        The DB ``choices`` column (an option list, a bare type descriptor,
        or blank/None).

    Returns
    -------
    str
        The effective type, lower-cased: one of the descriptor words above,
        a ``variabletype`` value, or ``"choice"`` / ``"numeric"`` fallbacks.
    """
    vartype = (variabletype or "").strip().lower()
    raw_choices = (choices or "").strip()
    descriptor = raw_choices.lower()
    if descriptor in _CHOICE_TYPE_DESCRIPTORS:
        return descriptor
    if vartype:
        return vartype
    if "," in raw_choices:
        return "choice"
    return "numeric"


class VariableSpec(BaseModel):
    """One GEECS variable exposed as EPICS PV(s).

    A readback PV (``prefix:suffix``) is always created.  When ``settable`` is
    true, a companion setpoint PV (``prefix:suffix:SP``) is created whose CA puts
    are forwarded to the device over UDP.
    """

    geecs_var: str
    pv: str | None = None
    dtype: DType = "float"
    settable: bool = False
    egu: str = ""
    precision: int = 3
    lo: float | None = None
    hi: float | None = None
    choices: list[str] = Field(default_factory=list)  # ordered options for enum
    deadband: float = 0.0  # float monitor deadband; only post when |Δ| > deadband
    alarm_limits: AlarmLimits | None = None
    # One-line human description for the PV's .DESC field (Phoebus/archiver
    # label). Stable *identity* only — not a change log; time-varying
    # provenance belongs in git-tracked config or the elog, never here (a
    # .DESC edit leaves no history). Clipped to 40 chars (EPICS DBR_STRING).
    description: str = ""

    @field_validator("description")
    @classmethod
    def _clip_description(cls, value: str) -> str:
        """Clip an over-long description to the EPICS DBR_STRING limit."""
        if len(value) > _MAX_DESC_LENGTH:
            logger.warning(
                "description %r exceeds %d chars (EPICS .DESC limit) — clipping",
                value,
                _MAX_DESC_LENGTH,
            )
            return value[:_MAX_DESC_LENGTH]
        return value

    @property
    def pv_suffix(self) -> str:
        """PV component for this variable (explicit ``pv`` or GEECS var), normalized."""
        raw = self.pv if self.pv is not None else self.geecs_var
        return normalize_pv_component(raw)


class DeviceSpec(BaseModel):
    """One GEECS device: connection endpoint plus the variables to expose."""

    name: str
    host: str
    port: int
    prefix: str | None = None
    experiment: str | None = None
    variables: list[VariableSpec] = Field(default_factory=list)
    # Ordered ladder of GEECS variables to use as the PV wall-clock timestamp.
    # Both are subscribed on every device: `acq_timestamp` (true shot time, only
    # on triggered devices) is preferred, falling back to `systimestamp`
    # (LabVIEW epoch, present on every device). Non-triggered devices simply lack
    # acq_timestamp and use systimestamp.
    timestamp_vars: list[str] = Field(
        default_factory=lambda: ["acq_timestamp", "systimestamp"]
    )

    @property
    def pv_prefix(self) -> str:
        """EPICS PV prefix for this device (explicit ``prefix`` or device name)."""
        return self.prefix if self.prefix is not None else self.name

    def pv_name_for(self, var: VariableSpec) -> str:
        """Full PV name for ``var``: ``[Experiment:]Device:Variable``.

        Each namespace component is normalized to the CA-safe character set; the
        variable suffix is already normalized by :attr:`VariableSpec.pv_suffix`.
        """
        parts: list[str] = []
        if self.experiment:
            parts.append(normalize_pv_component(self.experiment))
        parts.append(normalize_pv_component(self.pv_prefix))
        parts.append(var.pv_suffix)
        return ":".join(parts)

    @classmethod
    def from_db_metadata(
        cls,
        name: str,
        host: str,
        port: int,
        variables_metadata: list[dict],
        *,
        include: list[str] | None = None,
        include_settable: bool = False,
        dtypes: dict[str, DType] | None = None,
        prefix: str | None = None,
        experiment: str | None = None,
    ) -> "DeviceSpec":
        """Build a spec from already-fetched GEECS DB variable metadata.

        This is the pure, network-free core of :meth:`from_geecs_db` — it takes
        the rows returned by ``GeecsDb.get_device_variables`` and maps them onto
        :class:`VariableSpec` fields (units → ``egu``, min/max → ``lo``/``hi``,
        ``set`` → ``settable``).

        Parameters
        ----------
        name : str
            GEECS device name (also the default PV prefix).
        host, port : str, int
            Device endpoint, as returned by ``GeecsDb.find_device``.
        variables_metadata : list of dict
            Rows with keys ``name``, ``units``, ``min``, ``max``, ``settable``.
        include : list of str, optional
            If given, only these variable names are exposed.  Otherwise all are.
        include_settable : bool
            Expose settable variables even when ``include`` would filter them
            out.  The include list is the *monitoring* subset (``get='yes'``);
            settable variables are the device's *control surface* (e.g. a
            camera's ``save`` / ``localsavingpath``), which CA clients need for
            writes regardless of what is monitored per shot.
        dtypes : dict, optional
            Per-variable dtype overrides.  Variables default to ``"float"`` —
            the GEECS DB carries no scalar type, so non-float variables (enums,
            addresses) must be named here or excluded via ``include``.
        prefix : str, optional
            Override the PV prefix (defaults to ``name``).

        Returns
        -------
        DeviceSpec
        """
        dtypes = dtypes or {}
        specs: list[VariableSpec] = []
        seen: set[str] = set()
        for meta in variables_metadata:
            var_name = meta["name"]
            if include is not None and var_name not in include:
                if not (include_settable and bool(meta.get("settable", False))):
                    continue
            if var_name in seen:  # DB can list a variable more than once
                continue
            seen.add(var_name)

            override = dtypes.get(var_name)
            raw_choices = (meta.get("choices") or "").strip()

            # Effective-type resolution (choice-descriptor quirks included)
            # lives in the shared pure helper so the DB-hygiene audit applies
            # the identical rules (see `effective_vartype`, #512).
            effective = effective_vartype(meta.get("variabletype"), raw_choices)

            if override is None and effective in _SKIP_VARTYPES:
                logger.debug(
                    "%s: skipping %r (type=%s — not scalar CA data)",
                    name,
                    var_name,
                    effective,
                )
                continue
            dtype: DType = override or _VARTYPE_TO_DTYPE.get(effective, "float")

            choices: list[str] = []
            if dtype == "enum":
                choices = [c.strip() for c in raw_choices.split(",") if c.strip()]
                too_many = len(choices) > _MAX_ENUM_STATES
                too_long = any(len(c) > _MAX_ENUM_STRING_LEN for c in choices)
                if not choices or too_many or too_long:
                    # unusable or not CA-representable → fall back to a string PV
                    if choices:
                        logger.debug(
                            "%s: %r enum not CA-representable "
                            "(%d options, longest %d chars) → string",
                            name,
                            var_name,
                            len(choices),
                            max(len(c) for c in choices),
                        )
                    dtype = "string"
                    choices = []

            specs.append(
                VariableSpec(
                    geecs_var=var_name,
                    dtype=dtype,
                    settable=bool(meta.get("settable", False)),
                    egu=meta.get("units") or "",
                    lo=meta.get("min"),
                    hi=meta.get("max"),
                    choices=choices,
                    # Present only once the DB SELECT exposes a description
                    # column (see CHANGELOG / design note — a deliberate
                    # on-network follow-up); absent today → "" and inert.
                    description=meta.get("description") or "",
                    # Deadband stays 0.0 — the DB "tolerance" is a *set
                    # convergence* criterion, NOT a monitor deadband; wiring
                    # it here froze sub-tolerance motion out of readbacks
                    # (shipped bug; see CLAUDE.md quirks).
                    deadband=0.0,
                )
            )
        return cls(
            name=name,
            host=host,
            port=port,
            prefix=prefix,
            experiment=experiment,
            variables=specs,
        )

    @classmethod
    def from_geecs_db(
        cls,
        name: str,
        *,
        include: list[str] | None = None,
        include_settable: bool = False,
        dtypes: dict[str, DType] | None = None,
        prefix: str | None = None,
        experiment: str | None = None,
    ) -> "DeviceSpec":
        """Build a spec by querying the GEECS database (needs lab network).

        Thin wrapper over :meth:`from_db_metadata`: resolves the device endpoint
        with ``GeecsDb.find_device`` and its variables with
        ``GeecsDb.get_device_variables``.  Import of ``GeecsDb`` is deferred so
        the rest of the package stays importable without a DB/MySQL dependency.

        Parameters
        ----------
        name : str
            GEECS device name exactly as it appears in the database.
        include, include_settable, dtypes, prefix
            Forwarded to :meth:`from_db_metadata`.

        Returns
        -------
        DeviceSpec
        """
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(name)
        metadata = GeecsDb.get_device_variables(name)
        return cls.from_db_metadata(
            name,
            host,
            port,
            metadata,
            include=include,
            include_settable=include_settable,
            dtypes=dtypes,
            prefix=prefix,
            experiment=experiment,
        )


class GatewayConfig(BaseModel):
    """Top-level gateway configuration: the set of devices to serve."""

    devices: list[DeviceSpec] = Field(default_factory=list)
    derived_channels: list[DerivedChannelSpec] = Field(default_factory=list)

    @classmethod
    def from_geecs_experiment(
        cls,
        experiment: str,
        *,
        subscribed_only: bool = True,
        enabled_only: bool = True,
        include_settable: bool = True,
    ) -> "GatewayConfig":
        """Build a config for a whole experiment from the GEECS database.

        Enumerates the experiment's devices (``enabled_only`` skips devices whose
        ``expt_device.enabled`` is not ``"yes"``), and builds a
        :class:`DeviceSpec` for each via :meth:`DeviceSpec.from_geecs_db`, tagged
        with ``experiment`` as the PV namespace prefix.  Devices that fail to
        resolve are logged and skipped rather than aborting the whole config.

        With ``subscribed_only`` (default), only the per-shot monitoring subset is
        exposed: each device is limited to its ``get='yes'`` variables from
        ``expt_device_variable`` — a far smaller, sensible set than every
        device-type variable.  Set it false to expose every variable.

        ``include_settable`` (default true) additionally exposes each device's
        settable variables even in subscribed mode: the get-list is the
        *monitoring* subset, but settable variables are the device's *control
        surface* (camera ``save`` / ``localsavingpath``, magnet setpoints, …)
        and CA clients need their ``:SP`` PVs for writes regardless of what is
        monitored per shot.  A device with *zero* ``get='yes'`` variables still
        gets its control surface (an empty monitoring list is not "no PVs").

        Everything comes from three batched queries (endpoints, variable
        metadata, get-list) rather than per-device lookups — whole-experiment
        startup cost is constant in the number of devices.

        This is the live-from-DB path (the DB is the source of truth); further
        curation belongs in a separate overlay applied on top, not here.

        Parameters
        ----------
        experiment : str
            GEECS experiment name (e.g. ``"Undulator"``); also the PV prefix.
        subscribed_only : bool
            Limit each device to its ``get='yes'`` variables (default true).
        enabled_only : bool
            Skip devices not enabled in the experiment (default true).
        include_settable : bool
            Also expose settable (control) variables in subscribed mode
            (default true).

        Returns
        -------
        GatewayConfig
        """
        from geecs_ca_gateway.db.geecs_db import GeecsDb

        endpoints = GeecsDb.get_experiment_devices(
            experiment, enabled_only=enabled_only
        )
        var_map = GeecsDb.get_experiment_device_variables(
            experiment, enabled_only=enabled_only
        )
        alarm_map = GeecsDb.get_ca_alarm_limits(experiment)
        sub_map: dict[str, list[str]] = {}
        if subscribed_only:
            sub_map = GeecsDb.get_subscribed_variables(
                experiment, enabled_only=enabled_only
            )
            for name in sub_map.keys() - endpoints.keys():
                logger.warning(
                    "from_geecs_experiment: %s has get='yes' variables but no "
                    "device-table endpoint — skipping",
                    name,
                )

        devices: list[DeviceSpec] = []
        for name, (host, port) in endpoints.items():
            # In subscribed mode a device absent from the get-map still gets an
            # (empty) include list: nothing is monitored, but include_settable
            # keeps its settable control surface — losing every :SP PV because
            # no variable is logged per shot was a real deployment gap.
            include = sub_map.get(name, []) if subscribed_only else None
            try:
                spec = DeviceSpec.from_db_metadata(
                    name,
                    host,
                    port,
                    var_map.get(name, []),
                    include=include,
                    include_settable=include_settable and include is not None,
                    experiment=experiment,
                )
            except Exception:
                logger.warning(
                    "from_geecs_experiment: skipping %s (could not build spec)",
                    name,
                    exc_info=True,
                )
                continue
            if not spec.variables:
                logger.debug(
                    "from_geecs_experiment: %s exposes no variables — skipping",
                    name,
                )
                continue
            for var in spec.variables:
                alarm_limits = alarm_map.get((name, var.geecs_var))
                if alarm_limits is None:
                    continue
                if var.dtype not in ("float", "int"):
                    logger.warning(
                        "from_geecs_experiment: ignoring alarm limits for "
                        "%s/%s because dtype=%s is not numeric",
                        name,
                        var.geecs_var,
                        var.dtype,
                    )
                    continue
                var.alarm_limits = alarm_limits
            devices.append(spec)
        served = {(dev.name, var.geecs_var) for dev in devices for var in dev.variables}
        for device, variable in sorted(set(alarm_map) - served):
            logger.warning(
                "from_geecs_experiment: ignoring alarm limits for %s/%s because "
                "that variable is not served by the gateway",
                device,
                variable,
            )
        logger.info(
            "from_geecs_experiment(%s): built %d/%d device spec(s)",
            experiment,
            len(devices),
            len(endpoints),
        )
        return cls(devices=devices)
