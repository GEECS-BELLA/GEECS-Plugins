"""Hardware integration tests for GeecsDevice.

Run with:
    cd GEECS-PythonAPI
    poetry run pytest --hardware tests/

Target device: U_S1H (power supply, variable: Current).
All tests are skipped unless --hardware is passed.

These tests intentionally cover the two known bugs in geecs_device.py:

  Bug 1 — get() / set() can return None instead of raising when the
           device is in an invalid state.  test_get_does_not_return_none
           and test_get_on_invalid_state_raises document the expected
           behaviour; they will FAIL until the bugs are fixed.

  Bug 2 — UDP port not released between device lifecycles (WinError 10048).
           test_rapid_reconnect_no_socket_error exercises five
           open/get/close cycles in a tight loop.
"""

from __future__ import annotations

import pytest

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE_NAME = "U_S1H"
VARIABLE = "Current"


def _open(geecs_exp_info) -> GeecsDevice:  # noqa: ARG001
    """Open U_S1H using the default alias mode (use_alias_in_TCP_subscription=False)."""
    return GeecsDevice(DEVICE_NAME)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
class TestGeecsDeviceHappyPath:
    def test_device_opens_without_error(self, geecs_exp_info):
        device = _open(geecs_exp_info)
        try:
            assert device.is_valid(), "device.is_valid() is False after successful open"
        finally:
            device.close()

    def test_get_returns_numeric(self, geecs_exp_info):
        """get('Current') on a live device must return a float, never None."""
        device = _open(geecs_exp_info)
        try:
            value = device.get(VARIABLE)
            assert value is not None, (
                "get() returned None — Bug 1: _state_value() found no entry, "
                "likely because use_alias_in_TCP_subscription=True stored the "
                "response under a wrong alias key"
            )
            assert isinstance(value, (int, float)), (
                f"Expected numeric value, got {type(value)}: {value!r}"
            )
        finally:
            device.close()

    def test_get_default_alias_mode_works(self, geecs_exp_info):
        """Default use_alias_in_TCP_subscription=False: get() must return a numeric
        without any manual flag override.
        """
        device = GeecsDevice(
            DEVICE_NAME
        )  # default: use_alias_in_TCP_subscription=False
        try:
            value = device.get(VARIABLE)
            assert value is not None, (
                "get() returned None with default alias mode — "
                "use_alias_in_TCP_subscription default may have regressed to True"
            )
            assert isinstance(value, (int, float))
        finally:
            device.close()

    def test_invalid_device_raises_at_init(self, geecs_exp_info):
        """Connecting to a non-existent device must raise, not silently succeed."""
        with pytest.raises(GeecsDeviceInstantiationError):
            GeecsDevice("U_NONEXISTENT_DEVICE_XYZ_99999")


# ---------------------------------------------------------------------------
# Regression tests for known bugs
# ---------------------------------------------------------------------------


@pytest.mark.hardware
class TestGeecsDeviceKnownBugs:
    def test_get_does_not_return_none(self, geecs_exp_info):
        """Bug 1 regression: get() must never silently return None.

        Currently FAILS because _state_value() can return None when the alias
        map isn't populated, or because _process_command() silently returns
        (dev_udp is None race) without raising.  Fix: make both paths raise
        GeecsDeviceInstantiationError or GeecsDeviceCommandFailed instead of
        falling through to None.
        """
        device = _open(geecs_exp_info)
        try:
            value = device.get(VARIABLE)
            assert value is not None, "get() returned None — Bug 1 not yet fixed"
        finally:
            device.close()

    def test_get_on_invalid_state_raises_not_returns_none(self, geecs_exp_info):
        """Bug 1 regression: when is_valid() is False, get() must raise.

        Simulates the invalid-device path by clearing dev_ip after a successful
        open (so is_valid() returns False without touching the real network).
        Currently _execute() returns None → get() returns None.
        After the fix: raises GeecsDeviceInstantiationError.
        """
        device = _open(geecs_exp_info)
        try:
            # Simulate post-close / invalid state without actually closing the socket
            device.dev_ip = ""  # forces is_valid() → False
            with pytest.raises(GeecsDeviceInstantiationError):
                device.get(VARIABLE)
        finally:
            # Restore so close() works cleanly
            device.dev_ip = ""
            device.close()

    def test_rapid_reconnect_no_socket_error(self, geecs_exp_info):
        """Bug 2 regression: UDP port must be fully released between device lifecycles.

        On Windows this manifests as WinError 10048 (address already in use).
        On macOS it may surface as OSError or silent bind failure.  Five rapid
        open/get/close cycles should all succeed without socket errors.
        """
        for i in range(5):
            device = _open(geecs_exp_info)
            try:
                value = device.get(VARIABLE)
                assert value is not None, f"get() returned None on reconnect #{i + 1}"
            finally:
                device.close()
