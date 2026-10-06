"""The served-set re-read (#943): a running instance follows the DB.

Hermetic like ``test_server.py``: the roster resolver is a scripted fake
(no DB), the cameras are the neighbouring suites' binary fake push servers,
and the PVA server runs isolated.  Assertions are on what a PVA client sees
— the device PVs coming and going, the ``:devices`` instance PV — and on the
log lines the runbook points at.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time

import pytest
from p4p.client.thread import Context

from geecs_pva_gateway import server as server_module
from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig
from geecs_pva_gateway.file_plugin import PLUGIN_SUFFIX
from geecs_pva_gateway.server import GeecsPvaGateway
from tests.test_file_plugin import StampedCamera, _wait_until
from tests.test_server import DEVICE, IMG, FakeCamera

pytestmark = pytest.mark.fake_server

CAMERA = DEVICE.decode()
IMAGE_PV = "testexp:uc_testcam:image"
PREFIX = "testexp:pvagateway:127_0_0_1"
LOGGER = "geecs_pva_gateway.server"
TICK = 0.1


class ScriptedRoster:
    """A roster resolver the test steers: ``answer`` is the DB's next verdict.

    ``error`` makes every read raise and ``delay`` makes it slow, until
    cleared; ``calls`` counts reads.  Called off-loop, like the DB.
    """

    def __init__(self, answer: list[DeviceSpec]) -> None:
        self.answer = list(answer)
        self.error: Exception | None = None
        self.delay = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> list[DeviceSpec]:
        with self._lock:
            self.calls += 1
            error, delay, answer = self.error, self.delay, list(self.answer)
        if delay:
            time.sleep(delay)
        if error is not None:
            raise error
        return answer


def _spec(device: str, port: int, *variables: str) -> DeviceSpec:
    return DeviceSpec(
        device=device,
        host="127.0.0.1",
        port=port,
        experiment="testexp",
        image_variables=list(variables) or ["image"],
    )


async def _start(
    initial: list[DeviceSpec], roster: ScriptedRoster
) -> tuple[GeecsPvaGateway, asyncio.Task]:
    # host= names the identity PVs even while nothing is served.
    config = PvaGatewayConfig(experiment="testexp", host="127.0.0.1", devices=initial)
    gateway = GeecsPvaGateway(config, roster_resolver=roster, roster_interval_s=TICK)
    task = asyncio.create_task(gateway.run(isolate=True))
    for _ in range(100):
        await asyncio.sleep(0.05)
        try:
            gateway.conf()
            break
        except AssertionError:
            continue
    return gateway, task


async def _shutdown(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _get(ctx: Context, pv: str, timeout: float = 2.0):
    """A PVA get off the loop (p4p's thread client blocks)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ctx.get(pv, timeout=timeout))


async def _devices(ctx: Context) -> list[str]:
    return [str(d) for d in await _get(ctx, f"{PREFIX}:devices")]


def _messages(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


@pytest.mark.timeout(30)
async def test_a_device_enabled_in_the_db_is_served_without_a_restart(caplog):
    """An instance started with nothing serves a device the DB later names:
    its PVs appear, its gated subscription works, ``:devices`` lists it."""
    cam = FakeCamera()
    await cam.start()
    roster = ScriptedRoster([])
    gateway, task = await _start([], roster)
    try:
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            assert await _devices(ctx) == []
            with caplog.at_level(logging.INFO, logger=LOGGER):
                roster.answer = [_spec(CAMERA, cam.port)]
                await _wait_until(lambda: gateway.served_devices == [CAMERA])
            assert await _devices(ctx) == [CAMERA]
            assert _messages(caplog, f"roster: +{CAMERA} (serving 1 devices: {CAMERA})")

            # The live-added worker gates and subscribes exactly as at start.
            got = threading.Event()
            sub = ctx.monitor(
                IMAGE_PV, lambda v: got.set() if v.shape == IMG.shape else None
            )
            await asyncio.wait_for(cam.connected.wait(), 5)
            await asyncio.get_running_loop().run_in_executor(None, got.wait, 10)
            assert got.is_set()
            assert str(await _get(ctx, f"{IMAGE_PV}:connected")) == "Connected"
            sub.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_a_device_disabled_in_the_db_is_dropped_with_its_subscription(caplog):
    """A served device the DB no longer names loses its PVs (stream, state and
    plugin) and its GEECS subscription; the other device and the identity PVs
    are untouched."""
    cam = FakeCamera()
    await cam.start()
    keep, gone = _spec("UC_Keep", cam.port), _spec(CAMERA, cam.port)
    roster = ScriptedRoster([keep, gone])
    gateway, task = await _start([keep, gone], roster)
    try:
        worker = next(w for w in gateway._workers if w.device == CAMERA)
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            sub = ctx.monitor(IMAGE_PV, lambda v: None)
            await asyncio.wait_for(cam.connected.wait(), 5)  # a watcher holds it
            with caplog.at_level(logging.INFO, logger=LOGGER):
                roster.answer = [keep]
                await _wait_until(lambda: gateway.served_devices == ["UC_Keep"])
            # Released by the removal — the watcher is still attached.
            await asyncio.wait_for(cam.disconnected.wait(), 10)
            assert _messages(caplog, f"roster: -{CAMERA} (serving 1 devices: UC_Keep)")
            # Its plugin writer threads were joined, not leaked (a device
            # toggled in the DB builds a fresh worker each time it returns).
            assert not any(p._thread.is_alive() for p in worker.plugins.values())
            sub.close()
        finally:
            ctx.close()
        # A fresh client's search gets no answer for any of its PVs.
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            for pv in (
                IMAGE_PV,
                f"{IMAGE_PV}:connected",
                f"{IMAGE_PV}{PLUGIN_SUFFIX}Capture",
            ):
                with pytest.raises(TimeoutError):
                    await _get(ctx, pv, timeout=0.5)
            assert str(await _get(ctx, "testexp:uc_keep:image:connected")) == "Idle"
            assert await _devices(ctx) == ["UC_Keep"]
            assert int(await _get(ctx, f"{PREFIX}:heartbeat")) >= 0
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_an_empty_db_answer_leaves_the_instance_pvs_and_heals(caplog):
    """Every device disabled: nothing served, the identity PVs stay up (part 1
    of #943), and re-enabling brings the device back — no restart anywhere."""
    cam = FakeCamera()
    await cam.start()
    spec = _spec(CAMERA, cam.port)
    roster = ScriptedRoster([spec])
    gateway, task = await _start([spec], roster)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            roster.answer = []
            await _wait_until(lambda: gateway.served_devices == [])
        assert _messages(caplog, "serving 0 devices: none; idling on the instance PVs")
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            assert await _devices(ctx) == []
            assert await _get(ctx, f"{PREFIX}:version") == server_module.__version__
            assert int(await _get(ctx, f"{PREFIX}:heartbeat")) >= 0
            with pytest.raises(TimeoutError):
                await _get(ctx, IMAGE_PV, timeout=0.5)
        finally:
            ctx.close()
        roster.answer = [spec]
        await _wait_until(lambda: gateway.served_devices == [CAMERA])
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            assert str(await _get(ctx, f"{IMAGE_PV}:connected")) == "Idle"
            assert await _devices(ctx) == [CAMERA]
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(60)
async def test_removal_waits_while_a_capture_session_is_open(tmp_path, caplog):
    """A device mid-capture through the file plugin is never torn down: the
    removal is deferred (and said so) until the session closes, and the
    session's frames keep landing meanwhile."""
    cam = StampedCamera()
    await cam.start()
    spec = _spec(CAMERA, cam.port)
    roster = ScriptedRoster([spec])
    gateway, task = await _start([spec], roster)
    plugin = gateway._workers[0].plugins["image"]
    ctx = Context("pva", conf=gateway.conf(), useenv=False)
    loop = asyncio.get_running_loop()
    hdf = IMAGE_PV + PLUGIN_SUFFIX

    async def put(suffix: str, value) -> None:
        await loop.run_in_executor(None, lambda: ctx.put(hdf + suffix, value))

    try:
        await put("FilePath", str(tmp_path) + os.sep)
        await put("FileName", CAMERA)
        cam.push(IMG, time.time() - 5.0)
        await put("Capture", True)
        await _wait_until(lambda: plugin.capturing)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            roster.answer = []
            calls = roster.calls
            await _wait_until(lambda: roster.calls >= calls + 3)
            await asyncio.sleep(TICK)
        assert gateway.served_devices == [CAMERA]
        deferred = _messages(caplog, "removal deferred to the next tick")
        assert (
            deferred
            and f"{CAMERA} left the DB set but image has a capture" in deferred[0]
        )
        t = time.time()
        cam.push(IMG, t)
        cam.push(IMG + 1, t + 1)
        await _wait_until(lambda: plugin.value("NumCaptured_RBV") == 2)
        await put("Capture", False)
        await _wait_until(lambda: gateway.served_devices == [], timeout=5)
        assert (tmp_path / f"{CAMERA}.h5").exists()  # finished, not cut short
    finally:
        ctx.close()
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_a_failed_or_slow_read_keeps_the_last_good_set_and_logs_once(
    monkeypatch, caplog
):
    """A raising resolver, then one slower than its budget: the set never
    shrinks, the streak is logged once, reads are never doubled, and the
    loop is alive afterwards (recovery logged, the next change applied)."""
    monkeypatch.setattr(server_module, "_ROSTER_RESOLVE_TIMEOUT_S", 0.2)
    cam = FakeCamera()
    await cam.start()
    spec = _spec(CAMERA, cam.port)
    roster = ScriptedRoster([spec])
    gateway, task = await _start([spec], roster)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            roster.error = RuntimeError("db down")
            calls = roster.calls
            await _wait_until(lambda: roster.calls >= calls + 3)
            await asyncio.sleep(TICK)
            assert gateway.served_devices == [CAMERA]
            failed = _messages(caplog, "roster re-read failed")
            assert len(failed) == 1 and "RuntimeError: db down" in failed[0]

            roster.error, roster.delay = None, 0.6  # three budgets long
            calls = roster.calls
            await asyncio.sleep(0.5)  # five ticks
            assert roster.calls - calls <= 1  # one read in flight, never doubled
            assert gateway.served_devices == [CAMERA]
            assert len(_messages(caplog, "roster re-read failed")) == 1  # one streak

            roster.delay = 0.0
            await _wait_until(lambda: _messages(caplog, "recovered after"), timeout=5)
            assert gateway.served_devices == [CAMERA]
            roster.answer = []
            await _wait_until(lambda: gateway.served_devices == [])
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_a_device_in_both_sets_keeps_its_worker(caplog):
    """A device the DB still names is left alone: an endpoint move is the
    supervisor's business (#854), and a changed stream-variable set is
    logged once, never churned — the watcher's subscription never drops."""
    cam = FakeCamera()
    await cam.start()
    spec = _spec(CAMERA, cam.port)
    roster = ScriptedRoster([spec])
    gateway, task = await _start([spec], roster)
    try:
        worker = gateway._workers[0]
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            sub = ctx.monitor(IMAGE_PV, lambda v: None)
            await asyncio.wait_for(cam.connected.wait(), 5)
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                roster.answer = [spec.model_copy(update={"port": spec.port + 1})]
                calls = roster.calls
                await _wait_until(lambda: roster.calls >= calls + 3)
                roster.answer = [
                    spec.model_copy(
                        update={"image_variables": ["image", "processed image"]}
                    )
                ]
                calls = roster.calls
                await _wait_until(lambda: roster.calls >= calls + 3)
                await asyncio.sleep(TICK)
            assert gateway._workers == [worker]
            assert cam.connections == 1 and cam.total_connections == 1
            drift = _messages(caplog, "stream variables changed in the DB")
            assert (
                len(drift) == 1
                and "['image'] -> ['image', 'processed image']" in drift[0]
            )
            sub.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_a_new_device_colliding_with_a_served_pv_is_refused_once(caplog):
    """The startup collision guard holds for the re-read: a device whose PV
    names normalize onto a served device's is refused (logged once), the
    served set stands."""
    a, b = _spec("UC_Cam-A", 1), _spec("UC_Cam_A", 2)
    roster = ScriptedRoster([a])
    gateway, task = await _start([a], roster)
    try:
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            roster.answer = [a, b]
            calls = roster.calls
            await _wait_until(lambda: roster.calls >= calls + 3)
            await asyncio.sleep(TICK)
        assert gateway.served_devices == ["UC_Cam-A"]
        refused = _messages(caplog, "roster: refusing UC_Cam_A")
        assert len(refused) == 1 and "collision" in refused[0]
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            assert await _devices(ctx) == ["UC_Cam-A"]
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
