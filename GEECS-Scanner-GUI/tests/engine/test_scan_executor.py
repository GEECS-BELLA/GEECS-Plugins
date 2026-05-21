"""Tests for ScanStepExecutor device-movement logic.

These tests exercise move_devices_parallel_by_device using FakeScanDevice
and FakeDeviceManager so no real hardware or DB connection is needed.
"""

from __future__ import annotations


from geecs_python_api.controls.interface.geecs_errors import GeecsDeviceCommandFailed

from tests.engine.conftest import FakeScanDevice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOLERANCE = 100.0  # wide tolerance so all tests pass unless we want failure


def _exp_info_for(device_name: str, var_name: str, tolerance: float = _TOLERANCE):
    """Return a minimal fake exp_info dict for one device variable."""
    return {device_name: {var_name: {"tolerance": tolerance}}}


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestMoveDevicesHappyPath:
    def test_single_device_single_var(self, make_executor):
        device = FakeScanDevice("Dev1")
        executor = make_executor(
            devices={"Dev1": device},
            exp_info_devices=_exp_info_for("Dev1", "Var1"),
        )
        executor.move_devices_parallel_by_device({"Dev1:Var1": 3.0}, False)
        assert device._set_call_count == 1

    def test_two_devices_parallel(self, make_executor):
        dev_a = FakeScanDevice("A")
        dev_b = FakeScanDevice("B")
        executor = make_executor(
            devices={"A": dev_a, "B": dev_b},
            exp_info_devices={
                **_exp_info_for("A", "X"),
                **_exp_info_for("B", "Y"),
            },
        )
        executor.move_devices_parallel_by_device({"A:X": 1.0, "B:Y": 2.0}, False)
        assert dev_a._set_call_count == 1
        assert dev_b._set_call_count == 1

    def test_two_vars_same_device_set_sequentially(self, make_executor):
        device = FakeScanDevice("Dev")
        executor = make_executor(
            devices={"Dev": device},
            exp_info_devices={
                "Dev": {
                    "V1": {"tolerance": _TOLERANCE},
                    "V2": {"tolerance": _TOLERANCE},
                }
            },
        )
        executor.move_devices_parallel_by_device({"Dev:V1": 1.0, "Dev:V2": 2.0}, False)
        assert device._set_call_count == 2

    def test_empty_vars_is_noop(self, make_executor):
        executor = make_executor()
        # Should not raise even with no devices registered
        executor.move_devices_parallel_by_device({}, False)

    def test_composite_device_skips_tolerance_lookup(self, make_executor):
        device = FakeScanDevice("CompDev", is_composite=True)
        # No exp_info entry needed — composite path uses hardcoded 10000 tol
        executor = make_executor(devices={"CompDev": device})
        executor.move_devices_parallel_by_device({"CompDev": 5.0}, is_composite=True)
        assert device._set_call_count == 1

    def test_unknown_device_skips_silently(self, make_executor):
        executor = make_executor(devices={})
        # Device "Ghost" is not registered — should log warning and not raise
        executor.move_devices_parallel_by_device({"Ghost:Var": 1.0}, False)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retries_on_rejected(self, make_executor):
        """GeecsDeviceCommandRejected is the only error that triggers retry."""
        device = FakeScanDevice("Dev", reject_after=1)
        executor = make_executor(
            devices={"Dev": device},
            exp_info_devices=_exp_info_for("Dev", "V"),
        )
        # max_retries=3 default — second attempt succeeds
        executor.move_devices_parallel_by_device({"Dev:V": 1.0}, False)
        assert device._set_call_count == 2

    def test_timeout_escalates_immediately_no_retry(self, make_executor):
        """GeecsDeviceExeTimeout escalates immediately — only one set() call is made."""
        device = FakeScanDevice("Dev", timeout_after=1)
        executor = make_executor(
            devices={"Dev": device},
            exp_info_devices=_exp_info_for("Dev", "V"),
        )
        executor.move_devices_parallel_by_device({"Dev:V": 1.0}, False)
        assert device._set_call_count == 1  # no retry — escalated on first failure

    def test_hardware_failure_escalates_immediately_no_retry(self, make_executor):
        """GeecsDeviceCommandFailed escalates immediately — only one set() call is made."""
        device = FakeScanDevice("Dev", fail_after=1)
        executor = make_executor(
            devices={"Dev": device},
            exp_info_devices=_exp_info_for("Dev", "V"),
        )
        executor.move_devices_parallel_by_device({"Dev:V": 1.0}, False)
        assert device._set_call_count == 1  # no retry — escalated on first failure


# ---------------------------------------------------------------------------
# Exhausted retries → DeviceCommandError + stop event
# ---------------------------------------------------------------------------


class TestRetryExhaustion:
    def _always_failing_device(self) -> FakeScanDevice:
        """A device that fails on every set() call."""

        class _AlwaysFail(FakeScanDevice):
            def set(self, variable, value, **_kw):
                self._set_call_count += 1
                raise GeecsDeviceCommandFailed(
                    "BadDev", f"set {variable}", "always fails"
                )

        return _AlwaysFail("BadDev")

    def test_escalation_with_no_callback_does_not_raise(self, make_executor):
        device = self._always_failing_device()
        executor = make_executor(
            devices={"BadDev": device},
            exp_info_devices=_exp_info_for("BadDev", "V"),
        )
        # on_escalate=None → logs warning, returns False — no exception propagates
        executor.move_devices_parallel_by_device({"BadDev:V": 1.0}, False)
        assert device._set_call_count == 1  # immediate escalation, no retry

    def test_stop_event_set_when_escalation_returns_abort(self, make_executor):
        device = self._always_failing_device()
        executor = make_executor(
            devices={"BadDev": device},
            exp_info_devices=_exp_info_for("BadDev", "V"),
        )
        executor.cmd_executor.on_escalate = lambda exc, ctx: True
        executor.move_devices_parallel_by_device({"BadDev:V": 1.0}, False)
        assert executor.stop_scanning_thread_event.is_set()
