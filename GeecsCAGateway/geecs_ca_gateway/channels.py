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
    AccessRights,
    ChannelChar,
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

# Capacity of `path` (long-string / char-array) channels. EPICS DBR_STRING caps
# at 40 chars, so paths are served as char arrays per the standard EPICS
# long-string convention; 512 comfortably covers GEECS save paths.
_PATH_MAX_LENGTH = 512


def _to_text(value: Any) -> str:
    """Coerce a CA char-array put / GEECS value to ``str``.

    Char-array values arrive as ``bytes`` or as sequences of integer character
    codes (numpy arrays / lists); NUL padding is stripped.
    """
    value = _unwrap_char_array(value)
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def _unwrap_char_array(value: Any) -> Any:
    """Collapse a sequence of character codes to ``bytes`` (leave text alone)."""
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__len__"):
        try:
            return bytes(bytearray(int(v) for v in value))
        except (TypeError, ValueError):
            return value
    return value


def _unwrap(value: Any) -> Any:
    """CA delivers put values as length-1 arrays/lists; return the scalar."""
    if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
        return value[0] if len(value) else value
    return value


def cast_value(dtype: DType, value: Any) -> Any:
    """Coerce a raw scalar GEECS value to the Python type for ``dtype``.

    For ``float``/``int``/``string``/``path`` only; enum values go through
    :func:`enum_index` / :func:`enum_geecs_value`.

    Parameters
    ----------
    dtype : {"float", "int", "string", "path"}
        Target scalar data type.
    value : Any
        Raw value from the GEECS stream or a CA put (arrays are unwrapped;
        ``path`` additionally decodes char-array puts to text).

    Returns
    -------
    Any
    """
    if dtype == "path":
        return _to_text(value)
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


_READONLY_CACHE: dict[type, type] = {}


def read_only(channel_cls: type[ChannelData]) -> type[ChannelData]:
    """Return a subclass of *channel_cls* that denies CA client writes.

    Readback PVs mirror GEECS state; only the gateway may write them (it calls
    ``.write`` directly, which bypasses the server's access check).  Without
    this, a mistaken ``caput`` to a readback *sticks*: the deadband cache
    compares against the gateway's own last write, so the next unchanged
    hardware frame is suppressed and the PV shows the client's value until the
    hardware actually changes.  Setpoints live at ``…:SP``.
    """
    if channel_cls not in _READONLY_CACHE:

        def check_access(self, hostname: str, username: str) -> AccessRights:
            """Report READ-only access so client writes are denied cleanly."""
            _ = hostname, username
            return AccessRights.READ

        _READONLY_CACHE[channel_cls] = type(
            f"ReadOnly{channel_cls.__name__}",
            (channel_cls,),
            {"check_access": check_access},
        )
    return _READONLY_CACHE[channel_cls]


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
        # Display (informational) limits only — NOT control limits, which caproto
        # *enforces* on write and would reject faithful-but-out-of-range readbacks
        # (e.g. NaN from a failed analysis). GEECS is the authority on valid values.
        if spec.lo is not None:
            kwargs["lower_disp_limit"] = spec.lo
        if spec.hi is not None:
            kwargs["upper_disp_limit"] = spec.hi
    return kwargs


def make_readback_channel(spec: VariableSpec) -> ChannelData:
    """Build a client-read-only channel populated by the subscription stream."""
    if spec.dtype == "enum":
        return read_only(ChannelEnum)(value=0, enum_strings=list(spec.choices))
    if spec.dtype == "path":
        return read_only(ChannelChar)(
            value="", string_encoding="utf-8", max_length=_PATH_MAX_LENGTH
        )
    return read_only(_SCALAR_BASE[spec.dtype])(**_metadata_kwargs(spec))


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

    if spec.dtype == "path":

        class _GeecsPathSetpoint(ChannelChar):
            """Char-array channel whose CA puts forward the text to GEECS."""

            async def write(self, value: Any, **kwargs: Any) -> Any:
                """Forward the put (decoded to text), then store it."""
                text = _to_text(value)
                await setter(text)
                return await super().write(text, **kwargs)

        return _GeecsPathSetpoint(
            value="",
            string_encoding="utf-8",
            max_length=_PATH_MAX_LENGTH,
        )

    base = _SCALAR_BASE[spec.dtype]
    dtype = spec.dtype

    class _GeecsSetpoint(base):  # type: ignore[valid-type,misc]
        """Scalar channel whose CA puts are forwarded to a GEECS device."""

        async def write(self, value: Any, **kwargs: Any) -> Any:
            """Forward the put to GEECS, then store the value locally."""
            await setter(cast_value(dtype, value))
            return await super().write(value, **kwargs)

    return _GeecsSetpoint(**_metadata_kwargs(spec))
