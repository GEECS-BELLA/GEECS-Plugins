"""Shared fixtures for data_acquisition unit tests.

All helpers here are network-free: FakeScanDevice duck-types the real
ScanDevice, and FakeDeviceManager provides the subset of DeviceManager
that ScanStepExecutor touches.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceCommandFailed,
    GeecsDeviceExeTimeout,
)
from geecs_scanner.data_acquisition.scan_options import ScanOptions


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
        set_returns: Optional[float] = None,
    ):
        self.name = name
        self.is_composite = is_composite
        self._fail_after = fail_after
        self._timeout_after = timeout_after
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
        return self._set_returns if self._set_returns is not None else value


# ---------------------------------------------------------------------------
# FakeDeviceManager
# ---------------------------------------------------------------------------


class FakeDeviceManager:
    """Duck-typed replacement for DeviceManager.

    Parameters
    ----------
    devices : dict[str, FakeScanDevice]
        Pre-populated device registry.
    statistic_noscan : bool
        Return value for ``is_statistic_noscan``.
    """

    def __init__(
        self,
        devices: Optional[Dict[str, FakeScanDevice]] = None,
        statistic_noscan: bool = False,
    ):
        self.devices: Dict[str, FakeScanDevice] = devices or {}
        self._statistic_noscan = statistic_noscan

    @staticmethod
    def is_statistic_noscan(variable_name) -> bool:
        """Return False — tests that need noscan behaviour set statistic_noscan."""
        return False


# ---------------------------------------------------------------------------
# ScanStepExecutor factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_options() -> ScanOptions:
    """Return a ScanOptions with all defaults."""
    return ScanOptions()


@pytest.fixture()
def make_executor(monkeypatch):
    """Return a factory that builds a ScanStepExecutor backed by fake hardware.

    Usage::

        def test_something(make_executor):
            device = FakeScanDevice("Dev", fail_after=1)
            executor = make_executor({"Dev": device})
            executor.move_devices_parallel_by_device({"Dev:Var1": 5.0}, False)
    """
    from geecs_scanner.data_acquisition.scan_executor import ScanStepExecutor

    def _factory(
        devices: Optional[Dict[str, FakeScanDevice]] = None,
        statistic_noscan: bool = False,
        options: Optional[ScanOptions] = None,
        exp_info_devices: Optional[Dict] = None,
    ) -> ScanStepExecutor:
        device_manager = FakeDeviceManager(
            devices=devices or {},
            statistic_noscan=statistic_noscan,
        )
        stop_event = threading.Event()
        pause_event = threading.Event()
        pause_event.set()

        executor = ScanStepExecutor(
            device_manager=device_manager,
            data_logger=MagicMock(),
            scan_data_manager=MagicMock(),
            options=options or ScanOptions(),
            stop_scanning_thread_event=stop_event,
            pause_scan_event=pause_event,
        )

        # Patch GeecsDevice.exp_info so tolerance lookups succeed without a DB
        fake_exp_info: Dict = {"devices": {}}
        if exp_info_devices:
            fake_exp_info["devices"].update(exp_info_devices)
        monkeypatch.setattr(
            "geecs_scanner.data_acquisition.scan_executor.GeecsDevice.exp_info",
            fake_exp_info,
        )
        return executor

    return _factory
