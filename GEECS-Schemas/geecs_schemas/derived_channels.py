"""DerivedChannels — computed PV declarations for the CA gateway.

This document declares operator-curated read-only PVs that the CA gateway
computes from numeric push-frame values.  Same-device inputs are evaluated from
one coherent source frame.  Cross-device inputs use latest-value semantics and
must declare a freshness window with ``stale_after``.
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
            "'U_DaqPad1'. Inputs may span devices only when the derived "
            "channel declares stale_after."
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
            "Input variables available to the expression. Inputs from one "
            "source device are frame-coherent; inputs spanning devices use "
            "latest-value semantics and require stale_after."
        ),
    )
    stale_after: float | None = Field(
        None,
        gt=0,
        description=(
            "Maximum input age in seconds for latest-value derived channels. "
            "Required when inputs span more than one source device. Leave unset "
            "for same-device frame-coherent expressions."
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
        """Return the first source device for backward-compatible consumers."""
        return self.inputs[0].device

    @property
    def source_devices(self) -> set[str]:
        """Return the set of source devices for this derived channel."""
        return {inp.device for inp in self.inputs}

    @property
    def is_cross_device(self) -> bool:
        """Return whether this derived channel spans multiple source devices."""
        return len(self.source_devices) > 1

    @model_validator(mode="after")
    def _validate_channel(self) -> "DerivedChannel":
        symbols = [inp.symbol for inp in self.inputs]
        if len(set(symbols)) != len(symbols):
            raise ValueError("derived-channel input symbols must be unique")
        if self.is_cross_device and self.stale_after is None:
            raise ValueError(
                "derived-channel stale_after is required when inputs span "
                "multiple source devices"
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
            "from numeric push-frame values. Cross-device entries use "
            "latest-value semantics with stale_after."
        ),
    )
