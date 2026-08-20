"""Opt-in live-lab tests: the real MySQL DB and a real GEECS device.

Both tiers are deselected by default (``addopts`` in ``pyproject.toml``) and
self-skip cleanly when the lab is unreachable, so they are safe to invoke
blindly. Run them deliberately, on the lab network or VPN (check with
``/lab-status`` first):

    poetry run pytest tests/test_live_lab.py -m integration   # DB only
    poetry run pytest tests/test_live_lab.py -m hardware      # device I/O

The hardware tier targets one known device/variable, overridable via env vars
(same convention as GeecsBluesky's ``GEECS_HW_*`` family)::

    GEECS_HW_DEVICE   (default U_S1H)      GEECS_HW_VAR   (default Current)

Reads and subscription only by default. Set ``GEECS_HW_ALLOW_SET=1`` to also
exercise ``set`` — it writes back the value just read (a no-op move), but that
still commands real hardware, so it stays opt-in on top of opt-in.
"""

from __future__ import annotations

import os

import pytest

from geecs_core.client import GeecsDevice
from geecs_core.db.geecs_db import GeecsDb
from geecs_core.exceptions import GeecsError

HW_DEVICE = os.environ.get("GEECS_HW_DEVICE", "U_S1H")
HW_VAR = os.environ.get("GEECS_HW_VAR", "Current")

# Config missing / off-network / connector absent → skip, never error. The
# bounded DB connect (CONNECT_TIMEOUT_S) is what keeps the off-network skip
# fast instead of a ~75 s hang.
_LAB_UNAVAILABLE = (FileNotFoundError, KeyError, ImportError, OSError, GeecsError)


def _find_device_or_skip(name: str) -> tuple[str, int]:
    try:
        return GeecsDb.find_device(name)
    except _LAB_UNAVAILABLE as exc:
        pytest.skip(f"lab DB unavailable or {name} unknown: {exc!r}")
    except Exception as exc:  # mysql.connector errors (lazy import, no base)
        pytest.skip(f"lab DB unreachable: {exc!r}")


@pytest.mark.integration
class TestLiveDb:
    def test_find_device_returns_endpoint(self) -> None:
        host, port = _find_device_or_skip(HW_DEVICE)
        assert host and port > 0

    def test_device_variables_have_metadata_shape(self) -> None:
        _find_device_or_skip(HW_DEVICE)
        variables = GeecsDb.get_device_variables(HW_DEVICE)
        assert variables, f"{HW_DEVICE} declares no variables"
        assert {"name", "units", "settable", "variabletype"} <= variables[0].keys()


@pytest.mark.hardware
class TestLiveDevice:
    """End-to-end through the full new stack: DB lookup → UDP/TCP → device."""

    @pytest.fixture
    def device(self):
        _find_device_or_skip(HW_DEVICE)
        dev = GeecsDevice(HW_DEVICE)
        try:
            yield dev
        finally:
            dev.close()

    def test_get_returns_numeric(self, device: GeecsDevice) -> None:
        value = device.get(HW_VAR)
        assert isinstance(value, (int, float)), f"{HW_VAR} read as {value!r}"
        assert device.state[HW_VAR] == value

    def test_subscribe_delivers_frames_with_shot_number(
        self, device: GeecsDevice
    ) -> None:
        import time

        device.subscribe([HW_VAR])
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and "shot number" not in device.state:
            time.sleep(0.1)
        assert device.state.get("connected") is True
        assert isinstance(device.state.get("shot number"), int)
        assert HW_VAR in device.state

    @pytest.mark.skipif(
        os.environ.get("GEECS_HW_ALLOW_SET") != "1",
        reason="set-back test requires GEECS_HW_ALLOW_SET=1",
    )
    def test_set_back_current_value(self, device: GeecsDevice) -> None:
        current = device.get(HW_VAR)
        result = device.set(HW_VAR, current)
        assert result == pytest.approx(current, abs=abs(current) * 0.05 + 0.05)
