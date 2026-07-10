"""Shared fixtures for engine unit tests.

All helpers here are network-free: FakeScanDevice duck-types the real
ScanDevice for DeviceCommandExecutor tests.
"""

from __future__ import annotations

from typing import Any, Optional

from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceCommandRejected,
    GeecsDeviceExeTimeout,
)


# ---------------------------------------------------------------------------
# FakeScanDevice
# ---------------------------------------------------------------------------


class FakeScanDevice:
    """Duck-typed replacement for ScanDevice used in unit tests.

    Parameters
    ----------
    name : str
        Device name (used in log messages).
    is_composite : bool
        When True, tolerance is ignored by the executor.
    fail_after : int or None
        If set, ``set()`` raises ``GeecsDeviceCommandFailed`` on the
        *fail_after*-th call (1-indexed).
    timeout_after : int or None
        If set, ``set()`` raises ``GeecsDeviceExeTimeout`` on the
        *timeout_after*-th call (1-indexed).
    set_returns : float or None
        Value returned by ``set()``.  Defaults to the requested value so
        that tolerance checks pass.
    """

    def __init__(
        self,
        name: str = "FakeDevice",
        is_composite: bool = False,
        fail_after: Optional[int] = None,
        timeout_after: Optional[int] = None,
        reject_after: Optional[int] = None,
        set_returns: Optional[float] = None,
    ):
        self.name = name
        self.is_composite = is_composite
        self._fail_after = fail_after
        self._timeout_after = timeout_after
        self._reject_after = reject_after
        self._set_returns = set_returns
        self._set_call_count = 0

    def set(self, variable: str, value: Any, **_kwargs) -> Any:
        """Simulate setting a device variable; raises on injected failure modes."""
        self._set_call_count += 1
        if self._fail_after is not None and self._set_call_count == self._fail_after:
            raise GeecsDeviceCommandFailed(
                self.name, f"set {variable}", f"injected at call {self._set_call_count}"
            )
        if (
            self._timeout_after is not None
            and self._set_call_count == self._timeout_after
        ):
            raise GeecsDeviceExeTimeout(self.name, f"set {variable}", timeout=5.0)
        if (
            self._reject_after is not None
            and self._set_call_count == self._reject_after
        ):
            raise GeecsDeviceCommandRejected(self.name, f"set {variable}")
        return self._set_returns if self._set_returns is not None else value
