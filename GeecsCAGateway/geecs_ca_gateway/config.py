"""Declarative configuration for the GEECS → EPICS CA gateway.

The gateway is driven entirely by a :class:`GatewayConfig`.  For the proof of
concept these are constructed by hand (see ``demo.py``); the intended next step
is to generate them from the GEECS experiment database dict, pulling per-variable
units/precision/limits from the attributes database.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .naming import normalize_pv_component

logger = logging.getLogger(__name__)

DType = Literal["float", "int", "string", "enum"]

# GEECS `variabletype` → gateway dtype. Types absent here (image, 1darray) are
# not scalar CA data and are skipped when building specs from the DB.
_VARTYPE_TO_DTYPE: dict[str, DType] = {
    "numeric": "float",
    "string": "string",
    "path": "string",
    "choice": "enum",
}
_SKIP_VARTYPES = {"image", "1darray"}

# EPICS DBR_ENUM limits: at most 16 states, each label at most 26 chars. A GEECS
# `choice` that exceeds either can't be a CA enum, so it degrades to a string PV
# (the option value still round-trips as text; only the dropdown is lost).
_MAX_ENUM_STATES = 16
_MAX_ENUM_STRING_LEN = 26


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
                continue
            if var_name in seen:  # DB can list a variable more than once
                continue
            seen.add(var_name)

            override = dtypes.get(var_name)
            vartype = (meta.get("variabletype") or "").strip().lower()
            raw_choices = (meta.get("choices") or "").strip()
            descriptor = raw_choices.lower()

            # The `choice` table's low IDs double as type descriptors, so when
            # `choices` is a bare descriptor word it is the AUTHORITATIVE type —
            # even when variabletype says otherwise (e.g. variabletype='choice'
            # with choices='image' is an image variable streaming raw bytes, not a
            # one-option enum). Otherwise trust variabletype; if it's blank, a real
            # option list is a choice, else fall back to numeric.
            if descriptor in _SKIP_VARTYPES or descriptor in (
                "numeric",
                "string",
                "path",
            ):
                effective = descriptor
            elif vartype:
                effective = vartype
            elif "," in raw_choices:
                effective = "choice"
            else:
                effective = "numeric"

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
                    deadband=meta.get("tolerance") or 0.0,
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
        include, dtypes, prefix
            Forwarded to :meth:`from_db_metadata`.

        Returns
        -------
        DeviceSpec
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(name)
        metadata = GeecsDb.get_device_variables(name)
        return cls.from_db_metadata(
            name,
            host,
            port,
            metadata,
            include=include,
            dtypes=dtypes,
            prefix=prefix,
            experiment=experiment,
        )


class GatewayConfig(BaseModel):
    """Top-level gateway configuration: the set of devices to serve."""

    devices: list[DeviceSpec] = Field(default_factory=list)

    @classmethod
    def from_geecs_experiment(
        cls,
        experiment: str,
        *,
        subscribed_only: bool = True,
        enabled_only: bool = True,
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

        Returns
        -------
        GatewayConfig
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        if subscribed_only:
            sub_map = GeecsDb.get_subscribed_variables(
                experiment, enabled_only=enabled_only
            )
            targets = list(sub_map.items())
        else:
            names = GeecsDb.list_devices(experiment, enabled_only=enabled_only)
            targets = [(name, None) for name in names]

        devices: list[DeviceSpec] = []
        for name, include in targets:
            try:
                devices.append(
                    DeviceSpec.from_geecs_db(
                        name, experiment=experiment, include=include
                    )
                )
            except Exception:
                logger.warning(
                    "from_geecs_experiment: skipping %s (could not build spec)",
                    name,
                    exc_info=True,
                )
        logger.info(
            "from_geecs_experiment(%s): built %d/%d device spec(s)",
            experiment,
            len(devices),
            len(targets),
        )
        return cls(devices=devices)
