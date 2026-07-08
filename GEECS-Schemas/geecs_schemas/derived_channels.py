"""DerivedChannels — computed PV declarations for the CA gateway.

This document declares operator-curated read-only PVs that the CA gateway
computes from one source device's numeric push-frame values.  Use it for
semantic quantities such as a Convectron pressure PV derived from a DAQ analog
input voltage, where the result should be visible to Phoebus, archiving, and
CA-backed clients like any other gateway readback.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class DerivedInput(SchemaModel):
    """One source variable bound to a symbol in a derived-channel formula."""

    symbol: str = Field(
        description=(
            "Python-style symbol used in the expression, e.g. 'v' for a "
            "voltage input. Must be a valid identifier and must not shadow a "
            "reserved math function or constant."
        )
    )
    device: str = Field(
        description=(
            "GEECS source device that provides this input variable, e.g. "
            "'U_DaqPad1'. All inputs for one derived channel must come from "
            "the same source device in schema version 1."
        )
    )
    variable: str = Field(
        description=(
            "GEECS source variable on the input device, e.g. "
            "'Analog Input 10'. The gateway subscribes to it even if it is not "
            "exposed as its own raw readback PV."
        )
    )

    @model_validator(mode="after")
    def _validate_symbol(self) -> "DerivedInput":
        if not self.symbol.isidentifier():
            raise ValueError(f"input symbol {self.symbol!r} is not a valid identifier")
        if self.symbol in {
            "acos",
            "asin",
            "atan",
            "cos",
            "e",
            "exp",
            "isfinite",
            "log",
            "log10",
            "pi",
            "sin",
            "sqrt",
            "tan",
            "tau",
        }:
            raise ValueError(f"input symbol {self.symbol!r} is reserved")
        return self


class DerivedChannel(SchemaModel):
    """One read-only float PV computed from a numeric expression.

    The output PV is named from ``device`` and ``variable`` using the gateway's
    usual PV normalization, with the experiment prefix supplied by the gateway
    launch config unless this entry overrides it.
    """

    device: str = Field(
        description=(
            "Device component of the output PV, e.g. 'U_ChamberVac' for "
            "'Undulator:U_ChamberVac:Pressure'. This may be semantic and does "
            "not need to be a real GEECS hardware device."
        )
    )
    variable: str = Field(
        description=(
            "Variable component of the output PV, e.g. 'Pressure'. The gateway "
            "normalizes it using the same rules as raw GEECS variables."
        )
    )
    expression: str = Field(
        min_length=1,
        description=(
            "Numeric formula for the output value, using input symbols and the "
            "gateway's restricted arithmetic subset. Example: '10**(v - 5)'."
        ),
    )
    inputs: list[DerivedInput] = Field(
        min_length=1,
        description=(
            "Input variables available to the expression. In schema version 1 "
            "all inputs for a derived channel must come from one source device, "
            "so the calculation is coherent within a single push frame."
        ),
    )
    experiment: str | None = Field(
        None,
        description=(
            "Optional experiment prefix override for the output PV. Leave unset "
            "to use the gateway's launched experiment."
        ),
    )
    pv: str | None = Field(
        None,
        description=(
            "Optional explicit output PV variable component. Leave unset to use "
            "the 'variable' field."
        ),
    )
    egu: str = Field(
        "",
        description="Engineering units displayed by CA clients, e.g. 'Torr'.",
    )
    precision: int = Field(
        3,
        ge=0,
        description="Number of decimal places CA clients should display.",
    )
    lo: float | None = Field(
        None,
        description=(
            "Optional lower display limit for the output PV. This is metadata "
            "only; it is not an alarm or control limit."
        ),
    )
    hi: float | None = Field(
        None,
        description=(
            "Optional upper display limit for the output PV. This is metadata "
            "only; it is not an alarm or control limit."
        ),
    )
    deadband: float = Field(
        0.0,
        ge=0,
        description=(
            "Monitor deadband for the computed float value. Leave at 0.0 to "
            "post every changed value and suppress only exact repeats."
        ),
    )
    description: str = Field(
        "",
        description=(
            "Operator-facing note describing what the derived PV represents, "
            "for example the gauge model or calibration provenance."
        ),
    )

    @property
    def source_device(self) -> str:
        """Return the single source device for this derived channel."""
        return self.inputs[0].device

    @model_validator(mode="after")
    def _validate_channel(self) -> "DerivedChannel":
        symbols = [inp.symbol for inp in self.inputs]
        if len(set(symbols)) != len(symbols):
            raise ValueError("derived-channel input symbols must be unique")
        source_devices = {inp.device for inp in self.inputs}
        if len(source_devices) != 1:
            raise ValueError(
                "derived-channel inputs must come from one source device in v1"
            )
        return self


class DerivedChannels(VersionedSchemaModel):
    """A file of computed read-only PVs for the CA gateway.

    Load this file at gateway startup to add semantic, derived readbacks on top
    of the raw GEECS database-driven PV set.
    """

    derived_channels: list[DerivedChannel] = Field(
        default_factory=list,
        description=(
            "Derived PVs to expose. Each entry computes one read-only float PV "
            "from one source device's numeric push-frame values."
        ),
    )
