"""caproto channel construction for GEECS variables.

Channel kinds:

* **readback** — value pushed by the subscription stream
  (:func:`make_readback_channel`).
* **setpoint** — CA ``caput`` forwarded to the device over UDP *before* the value
  is stored (:func:`make_setpoint_channel`); internal code never writes to a
  setpoint channel, so there is no feedback loop.

Enum variables (GEECS ``choice`` type) use :class:`caproto.ChannelEnum`: the CA
value is the option *index* while GEECS speaks the option *string*, so readback
maps string → index (:func:`enum_index`) and setpoint maps index/string →
string (:func:`enum_geecs_value`).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from caproto import (
    ChannelData,
    ChannelDouble,
    ChannelEnum,
    ChannelInteger,
    ChannelString,
)

from .config import DType, VariableSpec

#: Async callable that pushes a value to a GEECS device (``udp.set``-shaped).
Setter = Callable[[Any], Awaitable[Any]]

_SCALAR_BASE: dict[str, type[ChannelData]] = {
    "float": ChannelDouble,
    "int": ChannelInteger,
    "string": ChannelString,
}


def _unwrap(value: Any) -> Any:
    """CA delivers put values as length-1 arrays/lists; return the scalar."""
    if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
        return value[0] if len(value) else value
    return value


def cast_value(dtype: DType, value: Any) -> Any:
    """Coerce a raw scalar GEECS value to the Python type for ``dtype``.

    For ``float``/``int``/``string`` only; enum values go through
    :func:`enum_index` / :func:`enum_geecs_value`.

    Parameters
    ----------
    dtype : {"float", "int", "string"}
        Target scalar data type.
    value : Any
        Raw value from the GEECS stream or a CA put (arrays are unwrapped).

    Returns
    -------
    Any
    """
    value = _unwrap(value)
    if dtype == "float":
        return float(value)
    if dtype == "int":
        return int(float(value))
    return str(value)


def enum_index(choices: list[str], value: Any) -> int | None:
    """Map a GEECS value (option string, or a numeric index) to the enum index.

    Returns ``None`` if it cannot be resolved, so the caller can skip the update.
    """
    value = _unwrap(value)
    text = str(value).strip()
    for i, choice in enumerate(choices):
        if choice == text:
            return i
    try:  # tolerate a numeric index arriving instead of the label
        i = int(float(text))
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < len(choices) else None


def enum_geecs_value(choices: list[str], value: Any) -> str:
    """Map a CA put (option index or label) to the GEECS option string."""
    value = _unwrap(value)
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        i = int(value)
        if 0 <= i < len(choices):
            return choices[i]
    text = str(value).strip()
    if text in choices:
        return text
    try:
        i = int(float(text))
        if 0 <= i < len(choices):
            return choices[i]
    except (TypeError, ValueError):
        pass
    return text


def _initial(dtype: DType) -> Any:
    """Placeholder value a channel holds before its first update."""
    if dtype == "float":
        return 0.0
    if dtype in ("int", "enum"):
        return 0
    return ""


def _metadata_kwargs(spec: VariableSpec) -> dict[str, Any]:
    """Caproto kwargs (value + EGU/precision/limits) for a scalar channel."""
    kwargs: dict[str, Any] = {"value": _initial(spec.dtype)}
    if spec.dtype == "float":
        kwargs["precision"] = spec.precision
    if spec.dtype in ("float", "int"):
        if spec.egu:
            kwargs["units"] = spec.egu
        if spec.lo is not None:
            kwargs["lower_ctrl_limit"] = spec.lo
            kwargs["lower_disp_limit"] = spec.lo
        if spec.hi is not None:
            kwargs["upper_ctrl_limit"] = spec.hi
            kwargs["upper_disp_limit"] = spec.hi
    return kwargs


def make_readback_channel(spec: VariableSpec) -> ChannelData:
    """Build a read-only channel populated by the subscription stream."""
    if spec.dtype == "enum":
        return ChannelEnum(value=0, enum_strings=list(spec.choices))
    return _SCALAR_BASE[spec.dtype](**_metadata_kwargs(spec))


def make_setpoint_channel(spec: VariableSpec, setter: Setter) -> ChannelData:
    """Build a writable channel that forwards CA puts to GEECS.

    Parameters
    ----------
    spec : VariableSpec
        The variable being exposed (dtype, metadata, enum choices).
    setter : Setter
        Async callable invoked with the value to send to GEECS.  If it raises,
        the value is not stored and the CA put fails — the correct semantics.
    """
    if spec.dtype == "enum":
        choices = list(spec.choices)

        class _GeecsEnumSetpoint(ChannelEnum):
            """Enum channel whose CA puts forward the option string to GEECS."""

            async def write(self, value: Any, **kwargs: Any) -> Any:
                """Forward the put (as the GEECS string), then store it."""
                await setter(enum_geecs_value(choices, value))
                return await super().write(value, **kwargs)

        return _GeecsEnumSetpoint(value=0, enum_strings=choices)

    base = _SCALAR_BASE[spec.dtype]
    dtype = spec.dtype

    class _GeecsSetpoint(base):  # type: ignore[valid-type,misc]
        """Scalar channel whose CA puts are forwarded to a GEECS device."""

        async def write(self, value: Any, **kwargs: Any) -> Any:
            """Forward the put to GEECS, then store the value locally."""
            await setter(cast_value(dtype, value))
            return await super().write(value, **kwargs)

    return _GeecsSetpoint(**_metadata_kwargs(spec))
