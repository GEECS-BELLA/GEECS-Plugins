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

import logging
from typing import Any, Awaitable, Callable

from caproto import (
    AccessRights,
    AlarmSeverity,
    AlarmStatus,
    ChannelAlarm,
    ChannelChar,
    ChannelData,
    ChannelDouble,
    ChannelEnum,
    ChannelInteger,
    ChannelString,
)

# Not re-exported at caproto top level; it is the exact class caproto's own
# verify_value raises, reused so pre-forward rejections are indistinguishable
# from caproto-native control-limit rejections to CA clients. Guarded because
# the private module may move under the uncapped caproto pin — the fallback
# subclasses the same public base, so client-observable failure is unchanged.
try:
    from caproto._data import CannotExceedLimits
except ImportError:  # pragma: no cover — future caproto moved the private module
    from caproto import CaprotoValueError

    class CannotExceedLimits(CaprotoValueError):  # type: ignore[no-redef]
        """Control-limit rejection (local stand-in for caproto's own class)."""


from .config import DType, VariableSpec

logger = logging.getLogger(__name__)

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
    """Map a GEECS readback value to its enum index.

    Resolution order:

    1. **Exact label match** — the wire text equals a choice verbatim.
    2. **Numeric-value match** — if the wire text parses as a number, compare
       it against every choice that itself parses as a number ("2.000000"
       matches the label "2").  Devices stream numeric option values in
       arbitrary float formatting, so a numeric label must be matched by
       *value*, never mistaken for an index (DG645-style trigger configs have
       labels like ["1", "2", "5"], where treating "2.0" as index 2 would
       select "5").
    3. **Index fallback** — only when *no* choice is numeric (e.g. labels
       ["Off", "On"] with wire value "1"), a numeric wire value is interpreted
       as the option index.

    If the choices contain numeric labels but none value-matches, returns
    ``None`` rather than guessing an index.  Returns ``None`` whenever the
    value cannot be resolved, so the caller can warn and skip the update.
    """
    value = _unwrap(value)
    text = str(value).strip()
    for i, choice in enumerate(choices):
        if choice == text:
            return i
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    # Numeric-value match against numeric labels.
    any_numeric_choice = False
    for i, choice in enumerate(choices):
        try:
            choice_number = float(choice)
        except (TypeError, ValueError):
            continue
        any_numeric_choice = True
        if choice_number == number:
            return i
    if any_numeric_choice:
        return None  # numeric labels exist but nothing matched — don't guess
    # No numeric labels at all: tolerate a numeric index arriving instead.
    i = int(number)
    return i if 0 <= i < len(choices) else None


def enum_geecs_value(choices: list[str], value: Any) -> str:
    """Map a CA put (option index or label) to the GEECS option string.

    A genuine numeric put (``int``/``float``/``bool``) is a CA enum *index* —
    standard Channel Access semantics.  **Text** puts resolve like
    :func:`enum_index`: exact label first, then numeric-value match against
    numeric labels (``"2.0"`` selects the label ``"2"``, never index 2), and
    index interpretation only when no label is numeric.  Unresolvable text is
    returned verbatim so GEECS itself rejects it (unchanged fallback).
    """
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
        number = float(text)
    except (TypeError, ValueError):
        return text
    any_numeric_choice = False
    for choice in choices:
        try:
            choice_number = float(choice)
        except (TypeError, ValueError):
            continue
        any_numeric_choice = True
        if choice_number == number:
            return choice
    if not any_numeric_choice:
        i = int(number)
        if 0 <= i < len(choices):
            return choices[i]
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


def _metadata_kwargs(
    spec: VariableSpec,
    *,
    initial_severity: AlarmSeverity | None = None,
    initial_status: AlarmStatus | None = None,
    control_limits: bool = False,
) -> dict[str, Any]:
    """Caproto kwargs (value + EGU/precision/limits) for a scalar channel.

    Parameters
    ----------
    spec : VariableSpec
        The variable being exposed.
    initial_severity, initial_status : optional
        Initial alarm state for the channel.
    control_limits : bool
        Additionally serve ``spec.lo``/``spec.hi`` as *control* limits
        (``DBR_CTRL``) — **setpoint channels only**. Readbacks must stay
        display-limits-only: caproto enforces control limits on every write,
        including the gateway's own stream mirroring, and would reject
        faithful-but-out-of-range readbacks (e.g. NaN from a failed
        analysis). Only well-formed spans (both bounds present, ``lo < hi``)
        are served — a degenerate or one-sided DB row must not brick an axis
        (``lo == hi`` is caproto's "limits disabled" sentinel).
    """
    kwargs: dict[str, Any] = {"value": _initial(spec.dtype)}
    if initial_severity is not None or initial_status is not None:
        kwargs["alarm"] = ChannelAlarm(
            severity=initial_severity or AlarmSeverity.NO_ALARM,
            status=initial_status or AlarmStatus.NO_ALARM,
        )
    if spec.dtype == "float":
        kwargs["precision"] = spec.precision
    if spec.dtype in ("float", "int"):
        if spec.egu:
            kwargs["units"] = spec.egu
        # Display (informational) limits on every scalar channel; control
        # limits only where `control_limits` says so (see the docstring).
        if spec.lo is not None:
            kwargs["lower_disp_limit"] = spec.lo
        if spec.hi is not None:
            kwargs["upper_disp_limit"] = spec.hi
        if (
            control_limits
            and spec.lo is not None
            and spec.hi is not None
            and spec.lo < spec.hi
        ):
            kwargs["lower_ctrl_limit"] = spec.lo
            kwargs["upper_ctrl_limit"] = spec.hi
    return kwargs


def make_readback_channel(
    spec: VariableSpec,
    *,
    initial_severity: AlarmSeverity | None = None,
    initial_status: AlarmStatus | None = None,
) -> ChannelData:
    """Build a client-read-only channel populated by the subscription stream."""
    if spec.dtype == "enum":
        kwargs: dict[str, Any] = {"value": 0, "enum_strings": list(spec.choices)}
        if initial_severity is not None or initial_status is not None:
            kwargs["alarm"] = ChannelAlarm(
                severity=initial_severity or AlarmSeverity.NO_ALARM,
                status=initial_status or AlarmStatus.NO_ALARM,
            )
        return read_only(ChannelEnum)(**kwargs)
    if spec.dtype == "path":
        kwargs = {
            "value": "",
            "string_encoding": "utf-8",
            "max_length": _PATH_MAX_LENGTH,
        }
        if initial_severity is not None or initial_status is not None:
            kwargs["alarm"] = ChannelAlarm(
                severity=initial_severity or AlarmSeverity.NO_ALARM,
                status=initial_status or AlarmStatus.NO_ALARM,
            )
        return read_only(ChannelChar)(**kwargs)
    return read_only(_SCALAR_BASE[spec.dtype])(
        **_metadata_kwargs(
            spec,
            initial_severity=initial_severity,
            initial_status=initial_status,
        )
    )


# EPICS DBR_STRING (the .DESC field's native type) caps at 40 characters.
_DESC_MAX_LENGTH = 40


def make_description_channel(description: str) -> ChannelData:
    """Build a read-only string channel serving a PV's ``.DESC`` field value.

    The gateway serves ``.DESC`` the simple way: a plain ``<pv>.DESC`` entry in
    the pvdb dict, which the CA server resolves via its first-line name lookup
    (no record machinery needed).  The text is clipped to the EPICS DBR_STRING
    limit; callers that source from ``VariableSpec.description`` are already
    clipped, but derived-channel descriptions are not, so clip here too.

    Parameters
    ----------
    description : str
        The description text (``.DESC`` field value).

    Returns
    -------
    ChannelData
        A client-read-only ``ChannelString`` holding the description.
    """
    return read_only(ChannelString)(value=description[:_DESC_MAX_LENGTH])


async def _forward_set(channel: ChannelData, setter: Setter, geecs_value: Any) -> None:
    """Forward one GEECS set, mirroring its outcome onto the channel alarm.

    A failed forward stamps ``WRITE``/``INVALID`` on the setpoint channel and
    re-raises (failing the CA put) — published, so monitor-based clients that
    never wait for put-completion (plain ``caput``, default Phoebus writes)
    still observe the failure.  The next successful forward clears the alarm;
    the clear is not published here because the value store that follows
    publishes it.  Both directions write on transition only, so a repeated
    failure does not re-publish an identical alarm state.

    Only the GEECS forward outcome is mirrored: value-shape errors raised
    *before* the setter (an unknown enum label, an uncastable scalar) are
    client errors, not device state, and leave the alarm untouched.
    """
    try:
        await setter(geecs_value)
    except Exception:
        alarm = channel.alarm
        if (alarm.severity, alarm.status) != (
            AlarmSeverity.INVALID_ALARM,
            AlarmStatus.WRITE,
        ):
            try:
                await alarm.write(
                    status=AlarmStatus.WRITE, severity=AlarmSeverity.INVALID_ALARM
                )
            except Exception:
                # Alarm bookkeeping is best-effort — the caput must fail with
                # the real set error, never a publish-side one.
                logger.debug("failed to publish set-failure alarm", exc_info=True)
        raise
    alarm = channel.alarm
    if (alarm.severity, alarm.status) != (
        AlarmSeverity.NO_ALARM,
        AlarmStatus.NO_ALARM,
    ):
        await alarm.write(
            status=AlarmStatus.NO_ALARM,
            severity=AlarmSeverity.NO_ALARM,
            publish=False,
        )


def make_setpoint_channel(spec: VariableSpec, setter: Setter) -> ChannelData:
    """Build a writable channel that forwards CA puts to GEECS.

    Parameters
    ----------
    spec : VariableSpec
        The variable being exposed (dtype, metadata, enum choices).
    setter : Setter
        Async callable invoked with the value to send to GEECS.  If it raises,
        the value is not stored and the CA put fails — the correct semantics —
        and the channel alarm goes ``WRITE``/``INVALID`` until the next
        successful set (:func:`_forward_set`), so the failure is observable
        without put-completion.
    """
    if spec.dtype == "enum":
        choices = list(spec.choices)

        class _GeecsEnumSetpoint(ChannelEnum):
            """Enum channel whose CA puts forward the option string to GEECS."""

            async def write(self, value: Any, **kwargs: Any) -> Any:
                """Forward the put (as the GEECS string), then store it."""
                await _forward_set(self, setter, enum_geecs_value(choices, value))
                return await super().write(value, **kwargs)

        return _GeecsEnumSetpoint(value=0, enum_strings=choices)

    if spec.dtype == "path":

        class _GeecsPathSetpoint(ChannelChar):
            """Char-array channel whose CA puts forward the text to GEECS."""

            async def write(self, value: Any, **kwargs: Any) -> Any:
                """Forward the put (decoded to text), then store it."""
                text = _to_text(value)
                await _forward_set(self, setter, text)
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
            """Forward the put to GEECS, then store the value locally.

            Control limits are checked **before** the GEECS forward, not
            merely via caproto's own ``verify_value`` (which runs inside
            ``super().write`` — i.e. *after* the UDP set has gone out; with
            DB limits tighter than the device's, the hardware would move and
            the put would then fail). The pre-forward check is the same
            semantics caproto applies: equal limits mean disabled.
            """
            geecs_value = cast_value(dtype, value)
            if dtype in ("float", "int"):
                lo, hi = self.lower_ctrl_limit, self.upper_ctrl_limit
                if lo != hi and not lo <= geecs_value <= hi:
                    # Pre-forward rejection: a client error, not a device
                    # outcome — no UDP, no :SP alarm, no LAST_SET_ERROR.
                    raise CannotExceedLimits(
                        f"Cannot write data {geecs_value}. Limits are set to "
                        f"{lo} and {hi}."
                    )
            await _forward_set(self, setter, geecs_value)
            return await super().write(value, **kwargs)

    return _GeecsSetpoint(**_metadata_kwargs(spec, control_limits=True))


#: Enum states of the ``CAGateway:RESTART`` control PV.
RESTART_STATES = ["Idle", "Restart"]


def make_restart_channel(on_restart: Callable[[], None]) -> ChannelData:
    """Build the ``CAGateway:RESTART`` control channel.

    A CA put of ``Restart`` (label or index 1) invokes *on_restart* — the
    gateway's clean-shutdown hook, which under systemd becomes a restart via
    the unit's ``RestartForceExitStatus``.  This is the devIocStats
    ``SYSRESET`` pattern: any CA client (Phoebus button, a GUI menu action, a
    one-line ``caput``) can bounce the service, which is also how it resyncs
    with the GEECS database after a device-set edit.  Writing ``Idle``/0 is a
    no-op, so two-state GUI widgets are safe to wire to it.
    """

    class _RestartChannel(ChannelEnum):
        """Enum channel whose ``Restart`` puts trigger the shutdown hook."""

        async def write(self, value: Any, **kwargs: Any) -> Any:
            """Store the put; fire *on_restart* when it selects ``Restart``."""
            if enum_geecs_value(RESTART_STATES, value) == "Restart":
                on_restart()
            return await super().write(value, **kwargs)

    return _RestartChannel(value=0, enum_strings=list(RESTART_STATES))
