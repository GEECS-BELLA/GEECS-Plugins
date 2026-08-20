"""Generalized ShotController tests (from_writes) — no CA needed, CI-safe.

Pins the ordered multi-device transition semantics
(:class:`~geecs_bluesky.models.shot_control.ShotControlWrites` +
``ShotController.from_writes``): writes replayed strictly in the profile's
declared order, each completing before the next; setters cached per
(device, variable); strict-mode validation; and the single-device
regression (one-device profiles land the same values per state as the
legacy controller).
"""

from __future__ import annotations

from typing import Any

import pytest
from bluesky import RunEngine
from ophyd_async.core import AsyncStatus

from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.shot_controller import ShotController


class _OrderRecordingSetter:
    """Mock setter that appends (name, value) to a shared journal on set()."""

    def __init__(self, name: str, journal: list) -> None:
        self.name = name
        self._journal = journal

    def set(self, value: Any) -> AsyncStatus:
        self._journal.append((self.name, value))

        async def _noop() -> None:
            pass

        return AsyncStatus(_noop())


MULTI_DEVICE_WRITES = ShotControlWrites(
    name="spans",
    states={
        "SCAN": [
            ("U_DG645", "Amplitude.Ch AB", "4.0"),
            ("U_PLC", "DO.Ch9", "on"),
        ],
        "STANDBY": [
            ("U_PLC", "DO.Ch9", "off"),
            ("U_DG645", "Amplitude.Ch AB", "0.5"),
        ],
        "ARMED": [("U_DG645", "Trigger.Source", "Single shot")],
        "SINGLESHOT": [("U_DG645", "Trigger.ExecuteSingleShot", "on")],
    },
)


def _writes_controller(
    writes: ShotControlWrites,
) -> tuple[RunEngine, ShotController, list]:
    journal: list = []
    controller = ShotController.from_writes(
        writes,
        setter_factory=lambda device, variable: _OrderRecordingSetter(
            f"{device}:{variable}", journal
        ),
    )
    return RunEngine(), controller, journal


class TestGeneralizedShotController:
    def test_multi_device_transition_preserves_declared_order(self) -> None:
        """SCAN writes both devices, in the profile's top-to-bottom order."""
        re, controller, journal = _writes_controller(MULTI_DEVICE_WRITES)
        re(controller.arm())  # SCAN
        assert journal == [
            ("U_DG645:Amplitude.Ch AB", "4.0"),
            ("U_PLC:DO.Ch9", "on"),
        ]

    def test_reverse_declared_order_is_respected_per_state(self) -> None:
        """STANDBY declares the devices in the opposite order — kept."""
        re, controller, journal = _writes_controller(MULTI_DEVICE_WRITES)
        re(controller.disarm())  # STANDBY
        assert journal == [
            ("U_PLC:DO.Ch9", "off"),
            ("U_DG645:Amplitude.Ch AB", "0.5"),
        ]

    def test_writes_are_sequential_each_completing_before_the_next(self) -> None:
        """Message-level: every set is followed by a wait on its own group."""
        commands = []
        plan = ShotController.from_writes(
            MULTI_DEVICE_WRITES,
            setter_factory=lambda d, v: _OrderRecordingSetter(f"{d}:{v}", []),
        ).set_state("SCAN")
        try:
            msg = plan.send(None)
            while True:
                commands.append((msg.command, msg.kwargs.get("group")))
                msg = plan.send(None)
        except StopIteration:
            pass
        assert [c for c, _ in commands] == ["set", "wait", "set", "wait"]
        # Each wait targets the group of the set immediately before it.
        assert commands[0][1] == commands[1][1]
        assert commands[2][1] == commands[3][1]
        assert commands[0][1] != commands[2][1]

    def test_state_without_writes_is_a_noop(self) -> None:
        re, controller, journal = _writes_controller(MULTI_DEVICE_WRITES)
        re(controller.quiesce())  # OFF: not declared
        assert journal == []

    def test_setters_cached_per_device_variable(self) -> None:
        created: list[str] = []

        def factory(device: str, variable: str) -> _OrderRecordingSetter:
            created.append(f"{device}:{variable}")
            return _OrderRecordingSetter(f"{device}:{variable}", [])

        ShotController.from_writes(MULTI_DEVICE_WRITES, setter_factory=factory)
        # DO.Ch9 and Amplitude.Ch AB each appear in two states but get ONE setter.
        assert sorted(created) == sorted(
            [
                "U_DG645:Amplitude.Ch AB",
                "U_PLC:DO.Ch9",
                "U_DG645:Trigger.Source",
                "U_DG645:Trigger.ExecuteSingleShot",
            ]
        )

    def test_require_strict_single_shot_on_writes(self) -> None:
        _re, controller, _journal = _writes_controller(MULTI_DEVICE_WRITES)
        controller.require_strict_single_shot()  # ARMED + SINGLESHOT present
        no_armed = ShotControlWrites(
            name="incomplete", states={"SCAN": [("U_DG645", "A", "1")]}
        )
        with pytest.raises(Exception, match="ARMED"):
            ShotController.from_writes(
                no_armed,
                setter_factory=lambda d, v: _OrderRecordingSetter(d, []),
            ).require_strict_single_shot()

    def test_single_device_writes_behave_like_the_legacy_controller(self) -> None:
        """Regression: a one-device profile lands the same values per state."""
        single = ShotControlWrites(
            name="htu",
            states={
                "SCAN": [
                    ("U_DG645_ShotControl", "Trigger.Source", "External rising edges")
                ],
                "OFF": [
                    (
                        "U_DG645_ShotControl",
                        "Trigger.Source",
                        "Single shot external rising edges",
                    )
                ],
            },
        )
        re, controller, journal = _writes_controller(single)
        re(controller.arm())
        re(controller.quiesce())
        assert journal == [
            ("U_DG645_ShotControl:Trigger.Source", "External rising edges"),
            ("U_DG645_ShotControl:Trigger.Source", "Single shot external rising edges"),
        ]
        assert controller.defines_state("SCAN")
        assert not controller.defines_state("STANDBY")
