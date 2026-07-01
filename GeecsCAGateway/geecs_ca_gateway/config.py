"""Declarative configuration for the GEECS → EPICS CA gateway.

The gateway is driven entirely by a :class:`GatewayConfig`.  For the proof of
concept these are constructed by hand (see ``demo.py``); the intended next step
is to generate them from the GEECS experiment database dict, pulling per-variable
units/precision/limits from the attributes database.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .naming import normalize_pv_component

DType = Literal["float", "int", "string"]


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

    @property
    def pv_suffix(self) -> str:
        """PV component for this variable (explicit ``pv`` or normalized var)."""
        if self.pv is not None:
            return self.pv
        return normalize_pv_component(self.geecs_var)


class DeviceSpec(BaseModel):
    """One GEECS device: connection endpoint plus the variables to expose."""

    name: str
    host: str
    port: int
    prefix: str | None = None
    variables: list[VariableSpec] = Field(default_factory=list)

    @property
    def pv_prefix(self) -> str:
        """EPICS PV prefix for this device (explicit ``prefix`` or device name)."""
        return self.prefix if self.prefix is not None else self.name

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
        for meta in variables_metadata:
            var_name = meta["name"]
            if include is not None and var_name not in include:
                continue
            specs.append(
                VariableSpec(
                    geecs_var=var_name,
                    dtype=dtypes.get(var_name, "float"),
                    settable=bool(meta.get("settable", False)),
                    egu=meta.get("units") or "",
                    lo=meta.get("min"),
                    hi=meta.get("max"),
                )
            )
        return cls(name=name, host=host, port=port, prefix=prefix, variables=specs)

    @classmethod
    def from_geecs_db(
        cls,
        name: str,
        *,
        include: list[str] | None = None,
        dtypes: dict[str, DType] | None = None,
        prefix: str | None = None,
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
            name, host, port, metadata, include=include, dtypes=dtypes, prefix=prefix
        )


class GatewayConfig(BaseModel):
    """Top-level gateway configuration: the set of devices to serve."""

    devices: list[DeviceSpec] = Field(default_factory=list)
