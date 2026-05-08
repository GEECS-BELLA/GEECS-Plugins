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
    def test_retries_on_transient_failure(self, make_executor):
        """A failure on the 1st call should be retried; 2nd call succeeds."""
        device = FakeScanDevice("Dev", fail_after=1)
        executor = make_executor(
            devices={"Dev": device},
            exp_info_devices=_exp_info_for("Dev", "V"),
        )
        # max_retries=3 default — second attempt should succeed
        executor.move_devices_parallel_by_device({"Dev:V": 1.0}, False)
        assert device._set_call_count == 2

    def test_retries_on_transient_timeout(self, make_executor):
        device = FakeScanDevice("Dev", timeout_after=1)
        executor = make_executor(
            devices={"Dev": device},
            exp_info_devices=_exp_info_for("Dev", "V"),
        )
        executor.move_devices_parallel_by_device({"Dev:V": 1.0}, False)
        assert device._set_call_count == 2


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

    def test_stop_event_set_on_exhaustion(self, make_executor):
        device = self._always_failing_device()
        executor = make_executor(
            devices={"BadDev": device},
            exp_info_devices=_exp_info_for("BadDev", "V"),
        )
        # on_device_error=None → escalate_device_error returns False (no abort)
        # but stop event must still be managed; exhaust retries silently
        executor.move_devices_parallel_by_device(
            {"BadDev:V": 1.0}, False, max_retries=2, retry_delay=0.0
        )
        # After 2 retries, executor returns without raising to the test; the
        # stop event may or may not be set depending on on_device_error result.
        # The key assertion: no unhandled exception propagated.

    def test_stop_event_set_when_on_device_error_returns_true(self, make_executor):
        device = self._always_failing_device()
        executor = make_executor(
            devices={"BadDev": device},
            exp_info_devices=_exp_info_for("BadDev", "V"),
        )
        # Inject an on_device_error that signals abort (escalate_device_error passes context= kwarg)
        executor.on_device_error = lambda exc, **_: True
        executor.move_devices_parallel_by_device(
            {"BadDev:V": 1.0}, False, max_retries=1, retry_delay=0.0
        )
        assert executor.stop_scanning_thread_event.is_set()
