"""Policy objects for known benign device command warnings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterable

from geecs_python_api.controls.interface.geecs_errors import GeecsDeviceCommandFailed


@dataclass(frozen=True)
class SuppressedSetErrorRule:
    """Rule for a known benign device set warning.

    ``variable=None`` means any set variable on the exact device can match,
    provided the hardware error text also contains all required fragments.
    """

    device: str
    variable: str | None
    required_fragments: tuple[str, ...]

    def applies_to(self, *, device_name: str, variable: str) -> bool:
        """Return True when this rule targets the device variable."""
        return device_name == self.device and (
            self.variable is None or variable == self.variable
        )

    def matches(
        self,
        *,
        device_name: str,
        variable: str,
        error_text: str = "",
    ) -> bool:
        """Return True when this rule applies to the set failure."""
        if not self.applies_to(device_name=device_name, variable=variable):
            return False
        return all(fragment in error_text for fragment in self.required_fragments)


@dataclass(frozen=True)
class SuppressedSetResult:
    """Return marker for a suppressed set warning and its reported readback."""

    readback: Any


DEFAULT_SUPPRESSED_SET_ERROR_RULES = (
    SuppressedSetErrorRule(
        device="HTT-B-Zaber_Chain-B",
        variable="Position.Ch6",
        required_fragments=(
            "Zaber A Series.lvlib:Error Query.vi<ERR>",
            "reports 1 warnings:",
            " WL",
        ),
    ),
)


class DeviceErrorSuppressionPolicy:
    """Match device command failures that may be logged instead of escalated."""

    def __init__(self, rules: Iterable[SuppressedSetErrorRule] = ()):
        self.rules = tuple(rules)

    @classmethod
    def default(cls) -> "DeviceErrorSuppressionPolicy":
        """Return scanner defaults for known benign command warnings."""
        return cls(rules=DEFAULT_SUPPRESSED_SET_ERROR_RULES)

    def suppresses_set_failure(self, device_name: str, variable: str) -> bool:
        """Return True when any rule targets this device variable."""
        return any(
            rule.applies_to(device_name=device_name, variable=variable)
            for rule in self.rules
        )

    def suppress_result_for(
        self,
        *,
        device_name: str,
        variable: str,
        exc: GeecsDeviceCommandFailed,
    ) -> SuppressedSetResult | None:
        """Return a suppressed result when *exc* matches a known benign warning."""
        if not exc.command.lower().startswith("set"):
            return None

        error_text = "\n".join(
            str(part)
            for part in (exc.error_detail, exc.actual_value, str(exc))
            if part is not None
        )
        for rule in self.rules:
            if rule.variable is not None and exc.command != f"set{variable}":
                continue
            if rule.matches(
                device_name=device_name,
                variable=variable,
                error_text=error_text,
            ):
                return SuppressedSetResult(_coerce_actual_value(exc.actual_value))

        return None


def _coerce_actual_value(value: str | None) -> Any:
    """Best-effort conversion of a device response value into a numeric readback."""
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return float(value)
        except ValueError:
            return value
