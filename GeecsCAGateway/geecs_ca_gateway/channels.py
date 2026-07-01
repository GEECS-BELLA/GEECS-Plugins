"""caproto channel construction for GEECS variables.

Two kinds of channel are built:

* **readback** — a plain :class:`caproto.ChannelData` subclass whose value is
  pushed by the TCP subscription stream.
* **setpoint** — a channel whose CA ``caput`` is forwarded to the GEECS device
  over UDP *before* the value is stored locally (:func:`make_setpoint_channel`).
  Internal code never writes to a setpoint channel, so there is no feedback loop.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from caproto import ChannelData, ChannelDouble, ChannelInteger, ChannelString

from .config import DType

#: Async callable that pushes a value to a GEECS device (``udp.set``-shaped).
Setter = Callable[[Any], Awaitable[Any]]

_READBACK_BASE: dict[str, type[ChannelData]] = {
    "float": ChannelDouble,
    "int": ChannelInteger,
    "string": ChannelString,
}


def cast_value(dtype: DType, value: Any) -> Any:
    """Coerce a raw GEECS value to the Python type matching ``dtype``.

    Handles both the GEECS stream (Python scalars) and CA puts, which arrive as
    length-1 arrays/lists — the leading element is extracted for a scalar PV.

    Parameters
    ----------
    dtype : {"float", "int", "string"}
        Target channel data type.
    value : Any
        Raw value from the GEECS stream or a CA put.

    Returns
    -------
    Any
        ``float``, ``int``, or ``str`` per ``dtype``.
    """
    # CA delivers put values as arrays; unwrap to a scalar (leave strings alone).
    if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
        value = value[0] if len(value) else value
    if dtype == "float":
        return float(value)
    if dtype == "int":
        return int(float(value))
    return str(value)


def initial_value(dtype: DType) -> Any:
    """Return the placeholder value a channel holds before its first update."""
    if dtype == "float":
        return 0.0
    if dtype == "int":
        return 0
    return ""


def _metadata_kwargs(
    spec_dtype: DType,
    *,
    egu: str,
    precision: int,
    lo: float | None,
    hi: float | None,
) -> dict[str, Any]:
    """Assemble the caproto channel kwargs (value + EGU/precision/limits)."""
    kwargs: dict[str, Any] = {"value": initial_value(spec_dtype)}
    if spec_dtype == "float":
        kwargs["precision"] = precision
    if egu:
        kwargs["units"] = egu
    if spec_dtype in ("float", "int"):
        if lo is not None:
            kwargs["lower_ctrl_limit"] = lo
            kwargs["lower_disp_limit"] = lo
        if hi is not None:
            kwargs["upper_ctrl_limit"] = hi
            kwargs["upper_disp_limit"] = hi
    return kwargs


def make_readback_channel(
    spec_dtype: DType,
    *,
    egu: str,
    precision: int,
    lo: float | None = None,
    hi: float | None = None,
) -> ChannelData:
    """Build a read-only channel populated by the subscription stream."""
    base = _READBACK_BASE[spec_dtype]
    return base(
        **_metadata_kwargs(spec_dtype, egu=egu, precision=precision, lo=lo, hi=hi)
    )


def make_setpoint_channel(
    spec_dtype: DType,
    setter: Setter,
    *,
    egu: str,
    precision: int,
    lo: float | None = None,
    hi: float | None = None,
) -> ChannelData:
    """Build a writable channel that forwards CA puts to GEECS.

    Parameters
    ----------
    spec_dtype : {"float", "int", "string"}
        Channel data type.
    setter : Setter
        Async callable invoked with the put value; expected to send the value to
        the GEECS device (e.g. ``GeecsUdpClient.set``).  If it raises, the value
        is *not* stored and the CA put fails — the correct error semantics.
    egu : str
        Engineering units string (``EGU``), or empty for none.
    precision : int
        Display precision for float channels.

    Returns
    -------
    caproto.ChannelData
        A channel instance whose ``write`` is GEECS-routed.
    """
    base = _READBACK_BASE[spec_dtype]

    class _GeecsSetpointChannel(base):  # type: ignore[valid-type,misc]
        """A ``base`` channel whose CA puts are forwarded to a GEECS device."""

        async def write(self, value: Any, **kwargs: Any) -> Any:
            """Forward the put to GEECS, then store the value locally."""
            await setter(cast_value(spec_dtype, value))
            return await super().write(value, **kwargs)

    return _GeecsSetpointChannel(
        **_metadata_kwargs(spec_dtype, egu=egu, precision=precision, lo=lo, hi=hi)
    )
