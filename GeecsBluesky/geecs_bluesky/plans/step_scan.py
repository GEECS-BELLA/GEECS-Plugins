"""geecs_step_scan — a Bluesky step-scan plan for GEECS hardware.

Moves a motor through a sequence of positions, collects ``shots_per_step``
shots from one or more detectors at each step, and emits Bluesky event
documents for downstream consumers (live callbacks, Databroker, etc.).

How it fits into Bluesky
------------------------
* :func:`bluesky.plan_stubs.mv` moves the motor by calling ``motor.set(pos)``
  and waiting for the returned :class:`~ophyd_async.core.AsyncStatus`.
* :func:`bluesky.plan_stubs.trigger_and_read` calls ``trigger()`` on every
  :class:`~bluesky.protocols.Triggerable` detector, waits for all triggers
  to complete, then reads all devices in the list (including the motor, so
  motor position is recorded alongside detector data in each event document).

Shot synchronisation
--------------------
Detectors that inherit from
:class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable` complete their
``trigger()`` call by waiting for the hardware ``acq_timestamp`` variable to
advance — the real shot timestamp from the DG645 delay generator.  This is
robust to device restarts (shot numbers drift; timestamps don't).

Example::

    import numpy as np
    from bluesky import RunEngine
    from geecs_bluesky.devices.motor import GeecsMotor
    from geecs_bluesky.devices.camera import GeecsCameraBase
    from geecs_bluesky.plans.step_scan import geecs_step_scan

    RE = RunEngine()

    motor = GeecsMotor("U_ESP_JetXYZ", "Position.Axis 1",
                       "192.168.8.198", 65158,
                       name="jet_x", units="mm")
    cam = GeecsCameraBase("U_ProbeCam", "192.168.8.50", 64000,
                          name="probe_cam")

    await motor.connect()
    await cam.connect()

    RE(geecs_step_scan(
        motor=motor,
        positions=np.linspace(0, 5, 6),
        detectors=[cam],
        shots_per_step=5,
        md={"sample": "He jet", "operator": "jdoe"},
    ))
"""

from __future__ import annotations

from typing import Any, Iterable

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


def geecs_step_scan(
    motor: Any,
    positions: Iterable[float],
    detectors: list[Any],
    shots_per_step: int = 5,
    md: dict[str, Any] | None = None,
):
    """Step-scan plan: move *motor* through *positions*, collect *shots_per_step* shots.

    Parameters
    ----------
    motor:
        A :class:`~bluesky.protocols.Movable` device (e.g.
        :class:`~geecs_bluesky.devices.motor.GeecsMotor`).
    positions:
        Iterable of motor positions to visit.
    detectors:
        List of :class:`~bluesky.protocols.Readable` / Triggerable devices
        to read at each shot (e.g. camera devices).  The motor is included
        automatically so its position is recorded in every event document.
    shots_per_step:
        Number of shots to collect at each motor position.  Default: ``5``.
    md:
        Extra metadata merged into the RunEngine ``start`` document.

    Yields
    ------
    Bluesky messages — pass this generator to a :class:`~bluesky.RunEngine`.

    Notes
    -----
    * The motor is appended to the read list for every shot so that each
      event document records the actual motor readback alongside detector
      data.
    * Non-Triggerable devices in *detectors* are read without calling
      ``trigger()`` (standard :func:`bluesky.plan_stubs.trigger_and_read`
      behaviour).
    """
    _positions = list(positions)
    _read_devices = list(detectors) + [motor]

    _md: dict[str, Any] = {
        "plan_name": "geecs_step_scan",
        "motor": getattr(motor, "name", str(motor)),
        "detectors": [getattr(d, "name", str(d)) for d in detectors],
        "positions": _positions,
        "shots_per_step": shots_per_step,
        "num_points": len(_positions),
        **(md or {}),
    }

    @bpp.run_decorator(md=_md)
    def _inner():
        for pos in _positions:
            yield from bps.mv(motor, pos)
            for _shot in range(shots_per_step):
                yield from bps.trigger_and_read(_read_devices)

    yield from _inner()
