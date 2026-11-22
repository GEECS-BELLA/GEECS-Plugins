"""The monitor-delivery watchdog: the caproto stall it exists for, and its verdicts.

Two layers:

* unit tests of ``geecs_ca_gateway.watchdog`` (no sockets): the counting
  queue, the stall verdict on a fake clock (stalled, busy, quiet, startup),
  the hard-exit timer and the evidence report;
* localhost tests (``fake_server``: sockets on 127.0.0.1 only) with a real
  caproto server and raw CA clients: the reproduction of the stall (one
  client that stops reading parks monitor delivery for every client, and the
  park outlives that client's disconnect), and the gateway exiting with the
  watchdog's status on that stall while a healthy busy or quiet gateway runs on.

The stall needs a client's bounded per-circuit queue to fill. caproto sizes it
from ``caproto.MAX_TOTAL_SUBSCRIPTION_BACKLOG`` (10000) when a circuit opens;
the localhost tests set it to 50 so the stall arrives in a few hundred
updates instead of ten thousand. The mechanism is the same at any size.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from types import SimpleNamespace
from typing import Any

import caproto as ca
import pytest
from caproto import ChannelDouble
from caproto.asyncio.server import Context

import geecs_ca_gateway.gateway as gateway_module
from geecs_ca_gateway import watchdog
from geecs_ca_gateway.config import GatewayConfig
from geecs_ca_gateway.gateway import RESTART_EXIT_CODE, GeecsCaGateway
from geecs_ca_gateway.watchdog import (
    WATCHDOG_EXIT_CODE,
    CountingQueue,
    DeliveryWatchdog,
    arm_hard_exit,
    install_counting_queue,
    stall_report,
)

# ----------------------------------------------------------------------
# Unit: the counting queue and the verdict
# ----------------------------------------------------------------------


class FakeClock:
    """A settable monotonic clock."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_counting_queue_counts_every_take() -> None:
    """``gets`` advances on get and get_nowait, not on put."""
    queue = CountingQueue()
    for item in range(3):
        queue.put_nowait(item)
    assert queue.gets == 0
    assert await queue.get() == 0
    assert queue.get_nowait() == 1
    assert queue.gets == 2


async def test_install_replaces_the_context_queue() -> None:
    """The context's queue becomes a counting one; a queue in use is refused."""
    context = Context({}, ["127.0.0.1"])
    queue = install_counting_queue(context)
    assert context.subscription_queue is queue
    queue.put_nowait("update")
    with pytest.raises(RuntimeError):
        install_counting_queue(context)


def test_a_stalled_queue_trips_at_the_limit() -> None:
    """Updates queued and none taken: the stall grows until the limit."""
    clock = FakeClock()
    queue = CountingQueue()
    dog = DeliveryWatchdog(queue, stall_limit_s=30.0, clock=clock)
    queue.put_nowait("update")
    for t in (1.0, 10.0, 29.0):
        clock.now = 1000.0 + t
        assert dog.stalled_for() == pytest.approx(t)
    clock.now = 1030.0
    assert dog.stalled_for() >= dog.stall_limit_s


def test_a_busy_queue_never_trips() -> None:
    """A loop taking updates as fast as they come is progress at every read."""
    clock = FakeClock()
    queue = CountingQueue()
    dog = DeliveryWatchdog(queue, stall_limit_s=30.0, clock=clock)
    for second in range(1, 600):
        for _ in range(50):
            queue.put_nowait("update")
        queue.get_nowait()  # the loop falls behind, but it is taking
        clock.now = 1000.0 + second
        assert dog.stalled_for() == 0.0
    assert queue.qsize() > 0  # a deep queue that drains is not a stall


def test_a_quiet_queue_never_trips() -> None:
    """No updates at all (no subscribers, nothing changing) is idle, not stalled."""
    clock = FakeClock()
    dog = DeliveryWatchdog(CountingQueue(), stall_limit_s=30.0, clock=clock)
    for second in range(1, 3600, 7):
        clock.now = 1000.0 + second
        assert dog.stalled_for() == 0.0


def test_startup_is_not_a_stall() -> None:
    """The first read after construction counts from construction, not from 0."""
    clock = FakeClock(now=50_000.0)
    queue = CountingQueue()
    for _ in range(100):
        queue.put_nowait("startup burst")
    dog = DeliveryWatchdog(queue, stall_limit_s=30.0, clock=clock)
    assert dog.stalled_for() == 0.0
    clock.now += 5.0
    assert dog.stalled_for() == pytest.approx(5.0)


def test_progress_resets_the_stall() -> None:
    """One update taken ends a stall that had not reached the limit."""
    clock = FakeClock()
    queue = CountingQueue()
    dog = DeliveryWatchdog(queue, stall_limit_s=30.0, clock=clock)
    queue.put_nowait("a")
    queue.put_nowait("b")
    clock.now += 25.0
    assert dog.stalled_for() == pytest.approx(25.0)
    queue.get_nowait()
    clock.now += 1.0
    assert dog.stalled_for() == 0.0


def test_the_thresholds_clear_a_healthy_worst_case() -> None:
    """The limit sits far above caproto's 1 s batch cap and below minutes."""
    assert 10 * 1.2 <= watchdog.STALL_LIMIT_S <= 120.0
    assert watchdog.CHECK_PERIOD_S <= watchdog.STALL_LIMIT_S / 10


async def test_watch_returns_on_a_stall_and_not_while_draining() -> None:
    """``watch`` keeps reading while a consumer drains; it returns on a stall."""
    queue = CountingQueue()
    dog = DeliveryWatchdog(queue, stall_limit_s=0.3)
    stop = asyncio.Event()

    async def produce_and_consume() -> None:
        while not stop.is_set():
            queue.put_nowait("update")
            await queue.get()
            await asyncio.sleep(0.01)

    worker = asyncio.create_task(produce_and_consume())
    watch = asyncio.create_task(dog.watch(check_period_s=0.05))
    await asyncio.sleep(1.0)
    assert not watch.done()  # draining for 3x the limit: no verdict
    stop.set()
    await worker
    queue.put_nowait("stuck")  # queued, and now nobody takes it
    stalled = await asyncio.wait_for(watch, timeout=2.0)
    assert stalled >= 0.3


def test_hard_exit_fires_after_the_grace_unless_cancelled() -> None:
    """The timer calls the exit function with the status once the grace runs out."""
    fired: list[int] = []
    done = threading.Event()

    def fake_exit(code: int) -> None:
        fired.append(code)
        done.set()

    arm_hard_exit(WATCHDOG_EXIT_CODE, grace_s=0.05, exit_fn=fake_exit)
    assert done.wait(timeout=2.0)
    assert fired == [WATCHDOG_EXIT_CODE]

    cancelled: list[int] = []
    timer = arm_hard_exit(WATCHDOG_EXIT_CODE, grace_s=0.2, exit_fn=cancelled.append)
    assert timer.daemon  # never keeps a finished process alive
    timer.cancel()
    timer.join(timeout=1.0)
    assert cancelled == []


def _fake_circuit(port: int, *, depth: int, cap: int = 3, connected: bool = True):
    queue: asyncio.Queue = asyncio.Queue(cap)
    for _ in range(depth):
        queue.put_nowait("update")
    events_on = asyncio.Event()
    events_on.set()
    return SimpleNamespace(
        circuit=SimpleNamespace(address=("10.0.0.1", port)),
        subscription_queue=queue,
        events_on=events_on,
        subscriptions={"spec": ["sub-a", "sub-b"]},
        connected=connected,
        _sub_task=None,
    )


async def test_the_report_names_the_waited_circuit_and_the_parked_await() -> None:
    """The ERROR block carries the queue, every circuit and the parked ``put``."""
    stuck = _fake_circuit(5001, depth=3, connected=False)
    healthy = _fake_circuit(5002, depth=0)
    context = SimpleNamespace(circuits=[stuck, healthy])

    async def _subscription_queue_send(circuit: Any) -> None:
        await circuit.subscription_queue.put("one more")  # parks: the queue is full

    async def subscription_queue_loop() -> None:
        await _subscription_queue_send(stuck)

    parked = asyncio.create_task(subscription_queue_loop(), name="ctx-delivery")
    await asyncio.sleep(0.01)
    queue = CountingQueue()
    queue.put_nowait("waiting")
    try:
        report = stall_report(context, queue, 31.5)
    finally:
        parked.cancel()
        await asyncio.gather(parked, return_exceptions=True)

    assert "STALLED" in report and "31.5 s" in report and "depth 1" in report
    assert f"status {WATCHDOG_EXIT_CODE}" in report
    waited = report.split("\n")[1]
    assert waited.startswith("delivery loop is waiting on circuit: 10.0.0.1:5001")
    assert "connected=False queue=3/3 FULL events_on=True subscriptions=2" in waited
    assert "circuits (2):" in report and "10.0.0.1:5002" in report
    # the await chain reaches the parked put, not just the task's own frame
    assert "in _subscription_queue_send" in report
    assert "queues.py" in report and "in put" in report
    assert "ctx-delivery" in report


# ----------------------------------------------------------------------
# Localhost: a real caproto server and raw CA clients
# ----------------------------------------------------------------------

SMALL_BACKLOG = 50
BIG_LENGTH = 16_000  # 128 kB per update: a few fill any socket buffer


class RawClient:
    """A minimal CA client over a plain socket, on caproto's sans-I/O layer.

    ``rcvbuf`` shrinks the socket's receive buffer; a client that then never
    calls :meth:`pump` stands in for one that stopped reading (a suspended
    process, a sleeping laptop, a wedged client thread).
    """

    def __init__(self, port: int, *, rcvbuf: int | None = None) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if rcvbuf:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self.sock.connect(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.local_port = self.sock.getsockname()[1]
        self.circ = ca.VirtualCircuit(ca.CLIENT, ("127.0.0.1", port), 0)
        self.events = 0
        self.reads: list[float] = []

    def send(self, *commands: Any) -> None:
        self.sock.sendall(b"".join(self.circ.send(*commands)))

    async def pump(self, until, timeout: float) -> bool:
        """Read and process until ``until()`` holds; False on timeout."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while not until():
            remaining = end - loop.time()
            if remaining <= 0:
                return False
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(self.sock, 1 << 20), remaining
                )
            except asyncio.TimeoutError:
                return False
            commands, _ = self.circ.recv(data)
            for command in commands:
                self.circ.process_command(command)
                if isinstance(command, ca.EventAddResponse):
                    self.events += 1
                elif isinstance(command, ca.ReadNotifyResponse):
                    self.reads.append(float(command.data[0]))
        return True

    async def drain(self, quiet_s: float = 0.5, timeout: float = 10.0) -> None:
        """Read until nothing has arrived for ``quiet_s`` (the backlog is out)."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            if not await self.new_event(quiet_s):
                return
        raise AssertionError("the client's backlog never ran dry")

    async def new_event(self, timeout: float) -> bool:
        """True if at least one monitor event arrives within ``timeout``."""
        before = self.events
        return await self.pump(lambda: self.events > before, timeout)

    async def get(self, chan: Any, timeout: float = 2.0) -> float | None:
        """A CA get on this circuit; None if no reply."""
        before = len(self.reads)
        self.send(chan.read(data_type=ca.ChannelType.DOUBLE))
        await self.pump(lambda: len(self.reads) > before, timeout)
        return self.reads[-1] if len(self.reads) > before else None

    async def open(self, name: str) -> Any:
        self.send(
            ca.VersionRequest(priority=0, version=13),
            ca.HostNameRequest("host"),
            ca.ClientNameRequest("user"),
        )
        chan = ca.ClientChannel(name, self.circ)
        self.send(chan.create())
        assert await self.pump(lambda: chan.states[ca.CLIENT] is ca.CONNECTED, 5.0)
        return chan

    async def monitor(self, name: str) -> Any:
        """Open ``name`` and subscribe to it."""
        chan = await self.open(name)
        self.send(chan.subscribe(data_type=ca.ChannelType.DOUBLE))
        return chan

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def localhost_ca(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve CA on 127.0.0.1 alone, on a free port, with a small client backlog."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setenv("EPICS_CA_SERVER_PORT", str(port))
    monkeypatch.setenv("EPICS_CAS_INTF_ADDR_LIST", "127.0.0.1")
    monkeypatch.setenv("EPICS_CAS_BEACON_ADDR_LIST", "127.0.0.1")
    monkeypatch.setenv("EPICS_CAS_AUTO_BEACON_ADDR_LIST", "NO")
    monkeypatch.setattr(ca, "MAX_TOTAL_SUBSCRIPTION_BACKLOG", SMALL_BACKLOG)


def _test_pvs() -> dict[str, ChannelDouble]:
    return {
        "test:big": ChannelDouble(value=[0.0] * BIG_LENGTH, max_length=BIG_LENGTH),
        "test:beat": ChannelDouble(value=0.0),
    }


async def _await_port(context_or_none) -> int:
    for _ in range(500):
        context = context_or_none()
        if context is not None and context.port:
            await asyncio.sleep(0.05)
            return context.port
        await asyncio.sleep(0.01)
    raise AssertionError("the CA server did not start")


async def _flood_until(pvs: dict[str, ChannelDouble], done, limit: int = 5000) -> int:
    """Post big and small updates until ``done()``; return the count posted."""
    for i in range(1, limit + 1):
        await pvs["test:big"].write([float(i)] * BIG_LENGTH)
        await pvs["test:beat"].write(float(i))
        await asyncio.sleep(0)
        if done():
            return i
    raise AssertionError(f"{limit} updates posted and the condition never held")


async def _stop(task: asyncio.Task, contexts: list[Context]) -> None:
    """Cancel a server task and close its circuits' sockets.

    caproto's cancelled server leaves accepted connections open; closing them
    here keeps their transports from being collected after the loop closes.
    """
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    for context in contexts:
        for circuit in list(context.circuits):
            circuit.client.writer.close()
    await asyncio.sleep(0)


def _circuit_for(context: Context, client: RawClient) -> Any:
    for circuit in context.circuits:
        if circuit.circuit.address[1] == client.local_port:
            return circuit
    raise AssertionError("no circuit for that client")


pytestmark_sockets = pytest.mark.fake_server


@pytestmark_sockets
async def test_one_unread_client_parks_monitors_for_every_client(localhost_ca) -> None:
    """The fault reproduced on caproto alone.

    A client that subscribes and stops reading fills its circuit's bounded
    queue; caproto's one delivery loop then waits on that queue's ``put``
    for good. Every other client stops receiving monitors while its gets
    still answer, the context queue grows without bound, and nothing is
    logged. The park outlives the stuck client's disconnect: its delivery
    task is cancelled with the queue still full, so nothing ever takes from
    it again.
    """
    pvs = _test_pvs()
    context = Context(pvs, ["127.0.0.1"])
    server = asyncio.create_task(context.run())
    try:
        port = await _await_port(lambda: context)
        healthy = RawClient(port)
        beat = await healthy.monitor("test:beat")
        # the value on subscribe
        assert await healthy.pump(lambda: healthy.events >= 1, 5.0)
        await pvs["test:beat"].write(-1.0)
        assert await healthy.new_event(2.0)  # monitors flow before the stuck client

        stuck = RawClient(port, rcvbuf=4096)
        await stuck.monitor("test:big")  # and never reads again
        stuck_circuit = _circuit_for(context, stuck)
        await _flood_until(
            pvs,
            lambda: stuck_circuit.subscription_queue.full()
            and context.subscription_queue.qsize() > 20,
        )

        await healthy.drain()  # what was sent before the park
        await pvs["test:beat"].write(-3.0)
        assert not await healthy.new_event(1.5)  # no monitors for the healthy client
        assert await healthy.get(beat) is not None  # gets still answer
        depth = context.subscription_queue.qsize()
        await _flood_until(pvs, lambda: True)
        assert context.subscription_queue.qsize() > depth  # the context queue grows

        stacks = watchdog.task_stacks(asyncio.all_tasks())
        assert any("in _subscription_queue_send" in line for line in stacks)

        stuck.close()
        for _ in range(100):
            if not stuck_circuit.connected:
                break
            await asyncio.sleep(0.02)
        assert not stuck_circuit.connected
        assert stuck_circuit.subscription_queue.full()  # never drained
        await pvs["test:beat"].write(-2.0)
        assert not await healthy.new_event(1.5)  # the park outlives the disconnect
        healthy.close()
    finally:
        await _stop(server, [context])


def _gateway_with_test_pvs() -> tuple[GeecsCaGateway, dict[str, ChannelDouble]]:
    gw = GeecsCaGateway(GatewayConfig(devices=[]))
    pvs = _test_pvs()
    gw.pvdb.update(pvs)
    return gw, pvs


@pytest.fixture
def fast_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1 s stall limit read every 0.1 s, so a stall is acted on in seconds."""
    monkeypatch.setattr(watchdog, "STALL_LIMIT_S", 1.0)
    monkeypatch.setattr(watchdog, "CHECK_PERIOD_S", 0.1)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Context]:
    """The caproto contexts the gateway starts, recorded as they are built."""
    contexts: list[Context] = []

    def recording(context: Context) -> CountingQueue:
        contexts.append(context)
        return install_counting_queue(context)

    monkeypatch.setattr(gateway_module, "install_counting_queue", recording)
    return contexts


@pytestmark_sockets
async def test_the_gateway_logs_the_evidence_and_exits_on_a_stall(
    localhost_ca, fast_watchdog, served, caplog: pytest.LogCaptureFixture
) -> None:
    """On the reproduced stall, run() logs one ERROR block and returns 87."""
    caplog.set_level(logging.ERROR, logger="geecs_ca_gateway.gateway")
    gw, pvs = _gateway_with_test_pvs()
    run = asyncio.create_task(gw.run())
    try:
        port = await _await_port(lambda: served[0] if served else None)
        healthy = RawClient(port)
        await healthy.monitor("test:beat")
        assert await healthy.pump(lambda: healthy.events >= 1, 5.0)
        stuck = RawClient(port, rcvbuf=4096)
        await stuck.monitor("test:big")

        await _flood_until(pvs, run.done, limit=200_000)
        assert await asyncio.wait_for(run, timeout=5.0) == WATCHDOG_EXIT_CODE
        healthy.close()
        stuck.close()
    finally:
        await _stop(run, served)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    block = errors[0].getMessage()
    assert "CA monitor delivery STALLED" in block
    assert f"delivery loop is waiting on circuit: 127.0.0.1:{stuck.local_port}" in block
    assert f"queue={SMALL_BACKLOG}/{SMALL_BACKLOG} FULL" in block
    assert "circuits (2):" in block
    assert "in _subscription_queue_send" in block and "in put" in block


@pytestmark_sockets
async def test_a_busy_then_quiet_gateway_runs_on(
    localhost_ca, fast_watchdog, served
) -> None:
    """Healthy clients under a flood, then silence, never trip the watchdog."""
    gw, pvs = _gateway_with_test_pvs()
    run = asyncio.create_task(gw.run())
    try:
        port = await _await_port(lambda: served[0] if served else None)
        readers = [RawClient(port) for _ in range(3)]
        for reader in readers:
            await reader.monitor("test:big")
            await reader.monitor("test:beat")

        # Stopped by an event, not by cancel: on Python 3.11, wait_for (in
        # pump) can swallow a cancel that lands as its read completes.
        stop_reading = asyncio.Event()

        async def keep_reading() -> None:
            while not stop_reading.is_set():
                for reader in readers:
                    await reader.pump(lambda: False, 0.01)

        reading = asyncio.create_task(keep_reading())
        loop = asyncio.get_running_loop()
        end = loop.time() + 2.5  # 2.5 stall limits of steady load
        await _flood_until(pvs, lambda: loop.time() > end or run.done(), limit=200_000)
        assert not run.done()
        await asyncio.sleep(2.5)  # then 2.5 stall limits of silence
        assert not run.done()
        stop_reading.set()
        await asyncio.wait_for(reading, timeout=5.0)
        assert all(reader.events > 10 for reader in readers)

        await gw.pvdb["cagateway:restart"].write("Restart")
        assert await asyncio.wait_for(run, timeout=5.0) == RESTART_EXIT_CODE
        for reader in readers:
            reader.close()
    finally:
        await _stop(run, served)
