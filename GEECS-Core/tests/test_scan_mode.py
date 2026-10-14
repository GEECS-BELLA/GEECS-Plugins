"""The scan-mode protocol pieces: messages, multicaster, acknowledgement listener, path server.

Every wire byte asserted here is the one captured from Master Control
(Undulator Scan006, 2026-10-13); the fake devices play the device side.
"""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from geecs_core.exceptions import GeecsConnectionError
from geecs_core.transport._framing import read_frame
from geecs_core.testing.fake_scan_mode_device import (
    FakeScanModeDevice,
    FakeScanModeGroup,
)
from geecs_core.transport.scan_mode import (
    KEEPALIVE_PAYLOAD,
    SCAN_END,
    SCAN_MODE_TCP_PORT,
    SCAN_START,
    ScanModeAckListener,
    ScanModeMulticaster,
    ScanModeServer,
    _bounded_drain,
    _wait_event,
    device_shot_path,
    frame,
    parse_acknowledgement,
    path_message,
    scan_end_message,
    scan_start_message,
)
from geecs_core.transport.udp_client import detect_local_ip

pytestmark = pytest.mark.fake_server

LOOPBACK = "127.0.0.1"
SCAN_FOLDER = r"z:\data\Undulator\Y2026\10-Oct\26_1013\scans\Scan006"
CAPTURED_PATH = SCAN_FOLDER + r"\xxx\Scan006_xxx_001.fileextension"
CAPTURED_ACK = (
    b"U_DG645_ShotControl>>ScanStart>>Undulator>>Shot #>>192.168.7.203>>no error,"
)


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((LOOPBACK, 0))
        return sock.getsockname()[1]


async def _until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


class _Stack:
    """Server + fake group + ack listener + multicaster, all on loopback."""

    def __init__(self, *devices: FakeScanModeDevice) -> None:
        self.devices = devices
        self.names = [device.name for device in devices]

    async def __aenter__(self) -> _Stack:
        self.server = ScanModeServer(LOOPBACK, 0)
        await self.server.start()
        for device in self.devices:
            device.tcp_port = self.server.port
        self.group = FakeScanModeGroup(self.devices, host=LOOPBACK)
        await self.group.start()
        self.listener = ScanModeAckListener(
            LOOPBACK, [device.command_port for device in self.devices]
        )
        await self.listener.open()
        self.multicaster = ScanModeMulticaster(
            LOOPBACK, self.group.port, local_ip=LOOPBACK
        )
        await self.multicaster.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.multicaster.close()
        await self.listener.close()
        await self.group.stop()
        await self.server.stop()

    async def start_scan(self) -> dict:
        since = asyncio.get_running_loop().time()
        await self.multicaster.announce_scan_start("Undulator", "Shot #")
        return await self.listener.wait_for(SCAN_START, self.names, 2.0, since=since)


def _devices(*names: str, **kwargs) -> list[FakeScanModeDevice]:
    return [FakeScanModeDevice(name, _free_udp_port(), **kwargs) for name in names]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_scan_start_bytes_match_the_capture(self) -> None:
        assert (
            scan_start_message("Undulator", "Shot #", "192.168.7.203")
            == b"ScanStart>>Undulator>>Shot #>>192.168.7.203"
        )

    def test_scan_end_bytes_match_the_capture(self) -> None:
        assert (
            scan_end_message("Undulator", "U_ProbeCamStage Home", "192.168.7.203")
            == b"ScanEnd>>Undulator>>U_ProbeCamStage Home>>192.168.7.203"
        )

    def test_path_frame_matches_the_capture(self) -> None:
        framed = frame(path_message(CAPTURED_PATH))
        assert framed[:4] == b"\x00\x00\x00\x5c"
        assert framed[4:] == b"Path>>" + CAPTURED_PATH.encode("ascii")
        assert len(framed) == 96

    def test_keepalive_frame_matches_the_capture(self) -> None:
        assert frame(KEEPALIVE_PAYLOAD) == b"\x00\x00\x00\x05dummy"

    def test_device_shot_path_builds_the_captured_template(self) -> None:
        assert device_shot_path(SCAN_FOLDER, 6, 1) == CAPTURED_PATH
        assert device_shot_path(SCAN_FOLDER + "\\", 6, 1) == CAPTURED_PATH
        assert device_shot_path(SCAN_FOLDER, 6, 1234).endswith(
            r"\Scan006_xxx_1234.fileextension"
        )

    def test_parse_acknowledgement_reads_the_captured_reply(self) -> None:
        ack = parse_acknowledgement(CAPTURED_ACK)
        assert ack is not None
        assert (ack.device, ack.command, ack.status) == (
            "U_DG645_ShotControl",
            SCAN_START,
            "no error,",
        )
        assert ack.ok

    def test_parse_acknowledgement_carries_an_error_verdict(self) -> None:
        ack = parse_acknowledgement(
            b'U_DG645_ShotControl>>ScanEnd>>Undulator>>x>>1.2.3.4>>error, Trigger Source must be "Single Shot"'
        )
        assert ack is not None
        assert ack.command == SCAN_END
        assert not ack.ok
        assert "Single Shot" in ack.status

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"Empty",
            b"setfire>>fire>>force>>accepted",
            b">>ScanStart>>x",
            b"garbage",
        ],
    )
    def test_parse_acknowledgement_rejects_everything_else(self, data: bytes) -> None:
        assert parse_acknowledgement(data) is None

    def test_the_port_is_the_framework_constant(self) -> None:
        assert SCAN_MODE_TCP_PORT == 6342

    async def test_read_frame_round_trips_and_refuses_a_corrupt_header(self) -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(frame(b"dummy") + frame(b"") + b"\xff\xff\xff\xff" + b"Path")
        reader.feed_eof()
        assert await read_frame(reader) == b"dummy"
        assert await read_frame(reader) == b""
        with pytest.raises(ValueError, match="corrupt"):
            await read_frame(reader)


# ---------------------------------------------------------------------------
# Entry: ScanStart, acknowledgements, connections
# ---------------------------------------------------------------------------


class TestEntry:
    async def test_every_device_acknowledges_scan_start_once_deduplicated(self) -> None:
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            acks = await stack.start_scan()
            assert set(acks) == {"U_A", "U_B"}
            assert all(ack.ok for ack in acks.values())
            assert (
                acks["U_A"].raw
                == "U_A>>ScanStart>>Undulator>>Shot #>>127.0.0.1>>no error,"
            )
            # The real device sends the datagram twice; one record per device.
            assert await _until(
                lambda: all(len(d.acks_sent) == 2 for d in stack.devices)
            )
            assert set(stack.listener.acknowledgements(SCAN_START)) == {"U_A", "U_B"}
            assert all(device.in_scan_mode for device in stack.devices)
            assert all(device.lock_ip == LOOPBACK for device in stack.devices)

    async def test_the_sequence_is_empty_empty_scan_start(self) -> None:
        async with _Stack(*_devices("U_A")) as stack:
            since = asyncio.get_running_loop().time()
            sent = await stack.multicaster.announce_scan_start("Undulator", "Shot #")
            await stack.listener.wait_for(SCAN_START, stack.names, 2.0, since=since)
            assert stack.group.received == [b"Empty", b"Empty", sent]
            assert sent == b"ScanStart>>Undulator>>Shot #>>127.0.0.1"

    async def test_a_silent_device_is_absent_after_the_timeout(self) -> None:
        devices = [*_devices("U_A"), *_devices("U_Silent", acknowledge=False)]
        async with _Stack(*devices) as stack:
            started = time.monotonic()
            since = asyncio.get_running_loop().time()
            await stack.multicaster.announce_scan_start("Undulator", "Shot #")
            acks = await stack.listener.wait_for(
                SCAN_START, stack.names, 0.3, since=since
            )
            assert set(acks) == {"U_A"}
            assert time.monotonic() - started >= 0.3
            assert not devices[1].in_scan_mode
            assert not devices[1].connected

    async def test_an_error_acknowledgement_is_reported_not_hidden(self) -> None:
        devices = _devices(
            "U_DG645", ack_status='error, Trigger Source must be "Single Shot"'
        )
        async with _Stack(*devices) as stack:
            acks = await stack.start_scan()
            assert set(acks) == {"U_DG645"}
            assert not acks["U_DG645"].ok
            assert "Single Shot" in acks["U_DG645"].status

    async def test_listener_refuses_a_port_another_process_holds(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as holder:
            holder.bind((LOOPBACK, 0))
            port = holder.getsockname()[1]
            listener = ScanModeAckListener(LOOPBACK, [_free_udp_port(), port])
            with pytest.raises(GeecsConnectionError, match=str(port)):
                await listener.open()
        # A failed open leaves the listener closed: no socket, and wait_for says so.
        with pytest.raises(RuntimeError, match="open"):
            await listener.wait_for(SCAN_START, [], 0.01, since=0.0)

    async def test_listener_collapses_duplicate_ports(self) -> None:
        port = _free_udp_port()
        listener = ScanModeAckListener(LOOPBACK, [port, port, str(port)])
        assert listener.ports == (port,)
        async with listener:
            pass

    async def test_devices_connect_to_the_path_server_after_scan_start(self) -> None:
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            assert stack.server.connection_count == 0
            await stack.start_scan()
            assert (
                await stack.server.wait_for_hosts({LOOPBACK}, timeout=2.0)
                == frozenset()
            )
            assert await _until(lambda: stack.server.connection_count == 2)
            assert stack.server.peer_hosts == {LOOPBACK}
            for device in stack.devices:
                assert await device.wait_connected(1.0)

    async def test_a_previous_runs_repeat_cannot_stand_in_for_a_silent_device(
        self,
    ) -> None:
        """The device acknowledges twice; the repeat must not pass the next run's check."""
        device = FakeScanModeDevice("U_A", _free_udp_port(), ack_gap_s=0.3)
        async with _Stack(device) as stack:
            loop = asyncio.get_running_loop()
            first_run = loop.time()
            await stack.multicaster.announce_scan_start("Undulator", "Shot #")
            acks = await stack.listener.wait_for(
                SCAN_START, ["U_A"], 2.0, since=first_run
            )
            assert set(acks) == {"U_A"}
            assert acks["U_A"].received_at >= first_run
            stack.listener.clear()
            assert await _until(lambda: len(device.acks_sent) == 2)
            await asyncio.sleep(0.05)
            # The repeat repopulated the bucket after clear() — the trap.
            assert set(stack.listener.acknowledgements(SCAN_START)) == {"U_A"}
            # The next run reads the clock before announcing; the device stays silent.
            device.acknowledge = False
            second_run = loop.time()
            await stack.multicaster.announce_scan_start("Undulator", "Shot #")
            acks = await stack.listener.wait_for(
                SCAN_START, ["U_A"], 0.2, since=second_run
            )
            assert acks == {}
            assert stack.listener.acknowledgements(SCAN_START, since=second_run) == {}

    async def test_opening_twice_is_refused_not_leaked(self) -> None:
        async with ScanModeMulticaster(LOOPBACK, 1, local_ip=LOOPBACK) as sender:
            with pytest.raises(RuntimeError, match="already open"):
                await sender.open()
        async with ScanModeAckListener(LOOPBACK, [_free_udp_port()]) as listener:
            with pytest.raises(RuntimeError, match="already open"):
                await listener.open()
        async with ScanModeServer(LOOPBACK, 0) as server:
            with pytest.raises(RuntimeError, match="already started"):
                await server.start()

    async def test_wait_for_hosts_names_the_hosts_still_missing(self) -> None:
        async with ScanModeServer(LOOPBACK, 0) as server:
            missing = await server.wait_for_hosts({"10.0.0.9", LOOPBACK}, timeout=0.1)
            assert missing == {"10.0.0.9", LOOPBACK}

    async def test_server_refuses_a_port_in_use(self) -> None:
        async with ScanModeServer(LOOPBACK, 0) as first:
            second = ScanModeServer(LOOPBACK, first.port)
            with pytest.raises(GeecsConnectionError, match=str(first.port)):
                await second.start()
            # A failed start leaves it not started: wait_for_hosts says so.
            with pytest.raises(RuntimeError, match="start"):
                await second.wait_for_hosts([], 0.01)


# ---------------------------------------------------------------------------
# The path channel
# ---------------------------------------------------------------------------


class TestPathChannel:
    async def test_keepalives_tick_about_every_40_ms_and_replies_are_drained(
        self,
    ) -> None:
        async with _Stack(*_devices("U_A")) as stack:
            await stack.start_scan()
            device = stack.devices[0]
            assert await device.wait_connected(1.0)
            await asyncio.sleep(0.3)
            # 0.3 s / 40 ms ≈ 7 ticks; allow for a loaded CI host.
            assert device.keepalives >= 4
            assert (
                stack.server.connection_count == 1
            )  # the zero-byte replies never choke it

    async def test_push_path_reaches_every_device_with_the_captured_framing(
        self,
    ) -> None:
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            await stack.start_scan()
            assert await stack.server.wait_for_hosts({LOOPBACK}, 2.0) == frozenset()
            assert await _until(lambda: stack.server.connection_count == 2)
            assert await stack.server.push_path(CAPTURED_PATH) == 2
            for device in stack.devices:
                assert await device.wait_for_paths(1, 1.0) == [CAPTURED_PATH]
                assert device.shot_numbers == [1]
                assert device.stamp == 1

    async def test_a_repeated_shot_number_is_not_re_acquired(self) -> None:
        async with _Stack(*_devices("U_A")) as stack:
            await stack.start_scan()
            device = stack.devices[0]
            assert await device.wait_connected(1.0)
            await stack.server.push_path(device_shot_path(SCAN_FOLDER, 6, 1))
            await stack.server.push_path(device_shot_path(SCAN_FOLDER, 6, 1))
            await stack.server.push_path(device_shot_path(SCAN_FOLDER, 6, 2))
            assert len(await device.wait_for_paths(3, 1.0)) == 3
            assert device.shot_numbers == [1, 1, 2]
            assert device.stamp == 2

    async def test_push_path_with_nobody_connected_reaches_nobody(self) -> None:
        async with ScanModeServer(LOOPBACK, 0) as server:
            assert await server.push_path(CAPTURED_PATH) == 0

    async def test_a_device_that_drops_its_connection_is_forgotten(self) -> None:
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            await stack.start_scan()
            assert await _until(lambda: stack.server.connection_count == 2)
            await stack.devices[0].stop()
            assert await _until(lambda: stack.server.connection_count == 1)
            assert await stack.server.push_path(CAPTURED_PATH) == 1
            assert await stack.devices[1].wait_for_paths(1, 1.0) == [CAPTURED_PATH]

    async def test_a_stalled_peer_does_not_delay_the_others(self) -> None:
        """One connection whose drain stalls costs only itself; the rest keep their cadence."""
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            stack.server._write_timeout_s = 0.3
            await stack.start_scan()
            assert await _until(lambda: stack.server.connection_count == 2)
            stalled = next(iter(stack.server._writers))

            async def never_drains() -> None:
                await asyncio.sleep(5.0)

            stalled.drain = never_drains  # type: ignore[method-assign]
            # The healthy peer must have its path long before the stalled drain is
            # judged: a serial loop would hand it over only after the 0.3 s stall.
            started = time.monotonic()
            push = asyncio.create_task(stack.server.push_path(CAPTURED_PATH))
            paths = [await device.wait_for_paths(1, 0.1) for device in stack.devices]
            assert [CAPTURED_PATH] in paths and time.monotonic() - started < 0.15
            took = await push
            elapsed = time.monotonic() - started
            assert took == 1
            assert 0.25 <= elapsed < 1.0  # bounded by the stall timeout
            assert (
                stack.server.connection_count == 1
            )  # only the stalled one was dropped

    async def test_fake_device_stop_closes_its_connection_without_the_collector(
        self,
    ) -> None:
        """Review: the fake used to lose its writer when cancelling the reader and left the socket to GC."""
        async with _Stack(*_devices("U_A")) as stack:
            await stack.start_scan()
            device = stack.devices[0]
            assert await device.wait_connected(1.0)
            keep_alive = (
                device._writer
            )  # a strong reference: only an explicit close can end it
            assert keep_alive is not None
            await device.stop()
            assert keep_alive.is_closing()
            assert await _until(lambda: stack.server.connection_count == 0, timeout=0.5)

    async def test_a_raw_peer_that_only_connects_and_leaves_is_tolerated(self) -> None:
        """A device may open a second connection and let it go (captured: 53 attempts, 26 kept)."""
        async with ScanModeServer(LOOPBACK, 0) as server:
            _, writer = await asyncio.open_connection(LOOPBACK, server.port)
            assert await _until(lambda: server.connection_count == 1)
            writer.close()
            await writer.wait_closed()
            assert await _until(lambda: server.connection_count == 0)

    async def test_scan_end_is_acknowledged_and_devices_free_run_again(self) -> None:
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            await stack.start_scan()
            assert await _until(lambda: stack.server.connection_count == 2)
            since = asyncio.get_running_loop().time()
            sent = await stack.multicaster.announce_scan_end("Undulator", "Shot #")
            assert sent == b"ScanEnd>>Undulator>>Shot #>>127.0.0.1"
            acks = await stack.listener.wait_for(
                SCAN_END, stack.names, 2.0, since=since
            )
            assert set(acks) == {"U_A", "U_B"} and all(ack.ok for ack in acks.values())
            assert await _until(lambda: not any(d.in_scan_mode for d in stack.devices))
            assert await _until(lambda: stack.server.connection_count == 0)
            # The ScanStart record survives beside the ScanEnd one.
            assert set(stack.listener.acknowledgements(SCAN_START)) == {"U_A", "U_B"}
            stack.listener.clear()
            assert stack.listener.acknowledgements(SCAN_START) == {}

    async def test_stop_closes_every_connection(self) -> None:
        async with _Stack(*_devices("U_A", "U_B")) as stack:
            await stack.start_scan()
            assert await _until(lambda: stack.server.connection_count == 2)
            await stack.server.stop()
            assert stack.server.connection_count == 0
            assert await _until(lambda: not any(d.connected for d in stack.devices))

    async def test_stop_is_bounded_even_if_the_ticker_ignores_cancellation(
        self,
    ) -> None:
        """Live 2026-10-13: wait_for swallowed the ticker's cancellation and stop() hung."""
        server = ScanModeServer(LOOPBACK, 0, keepalive_interval_s=0.01)
        await server.start()
        original = server._keepalive_loop

        async def stubborn() -> None:
            while True:
                try:
                    await original()
                    return
                except asyncio.CancelledError:
                    continue  # the swallowed cancellation, exaggerated

        assert server._keepalive_task is not None
        server._keepalive_task.cancel()
        await asyncio.sleep(0)
        server._keepalive_task = asyncio.create_task(stubborn())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await asyncio.wait_for(server.stop(), timeout=5.0)
        assert time.monotonic() - started < 3.0

    async def test_bounded_drain_propagates_a_same_tick_cancellation(self) -> None:
        """The swallowed-cancellation window: drain completes in the tick the caller is cancelled."""
        async with ScanModeServer(LOOPBACK, 0) as server:
            _, writer = await asyncio.open_connection(LOOPBACK, server.port)
            try:
                task = asyncio.create_task(_bounded_drain(writer, 1.0))
                await asyncio.sleep(0)  # the drain future is scheduled
                await asyncio.sleep(0)  # ... and completes in this tick
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                writer.close()

    async def test_wait_event_propagates_a_same_tick_cancellation(self) -> None:
        event = asyncio.Event()
        task = asyncio.create_task(_wait_event(event, 1.0))
        await asyncio.sleep(0)
        event.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_listener_wait_for_honours_an_abort_at_once(self) -> None:
        """An abort landing as the acks arrive must not be swallowed into a full timeout."""
        async with _Stack(*_devices("U_A")) as stack:
            since = asyncio.get_running_loop().time()
            waiter = asyncio.create_task(
                stack.listener.wait_for(
                    SCAN_START, ["U_A", "U_Never"], 5.0, since=since
                )
            )
            await stack.multicaster.announce_scan_start("Undulator", "Shot #")
            await _until(lambda: "U_A" in stack.listener.acknowledgements(SCAN_START))
            started = time.monotonic()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert time.monotonic() - started < 0.5

    async def test_client_ip_is_what_scan_start_names(self) -> None:
        """A host the devices resolve to a name announces that name; the bind address stays the IP."""
        async with _Stack(*_devices("U_A")) as stack:
            await stack.multicaster.close()
            stack.multicaster = ScanModeMulticaster(
                LOOPBACK, stack.group.port, local_ip=LOOPBACK, client_ip="localhost"
            )
            await stack.multicaster.open()
            assert stack.multicaster.client_ip == "localhost"
            sent = await stack.multicaster.announce_scan_start("Undulator", "Shot #")
            assert sent == b"ScanStart>>Undulator>>Shot #>>localhost"
            assert await stack.devices[0].wait_connected(1.0)  # resolved and connected
            assert (
                await stack.multicaster.announce_scan_end("Undulator", "x")
            ).endswith(b">>localhost")

    async def test_stop_keepalive_stops_the_ticker_but_keeps_connections(self) -> None:
        async with _Stack(*_devices("U_A")) as stack:
            await stack.start_scan()
            device = stack.devices[0]
            assert await device.wait_connected(1.0)
            await asyncio.sleep(0.15)
            assert device.keepalives >= 2
            await stack.server.stop_keepalive()
            seen = device.keepalives
            await asyncio.sleep(0.15)
            assert device.keepalives == seen  # no more ticks
            assert stack.server.connection_count == 1  # still connected
            assert await stack.server.push_path(CAPTURED_PATH) == 1  # pushes still work
            assert await device.wait_for_paths(1, 0.5) == [CAPTURED_PATH]
            await stack.server.stop()

    async def test_inbound_bytes_counts_what_devices_send(self) -> None:
        async with _Stack(*_devices("U_A")) as stack:
            await stack.start_scan()
            device = stack.devices[0]
            assert await device.wait_connected(1.0)
            await asyncio.sleep(0.15)
            (count,) = stack.server.inbound_bytes().values()
            # six bytes on connect plus six per keepalive answered
            assert count == 6 * (1 + device.keepalives) or count >= 12

    async def test_unopened_objects_say_so(self) -> None:
        with pytest.raises(RuntimeError, match="open"):
            ScanModeMulticaster(LOOPBACK, 1, local_ip=LOOPBACK).send(b"Empty")
        with pytest.raises(RuntimeError, match="open"):
            await ScanModeAckListener(LOOPBACK, []).wait_for(
                SCAN_START, [], 0.01, since=0.0
            )
        with pytest.raises(RuntimeError, match="start"):
            await ScanModeServer(LOOPBACK, 0).wait_for_hosts([], 0.01)


# ---------------------------------------------------------------------------
# Real multicast (skipped where the host cannot loop a group back)
# ---------------------------------------------------------------------------


class TestMulticast:
    async def test_a_real_group_round_trip(self) -> None:
        local_ip = detect_local_ip(
            "192.0.2.1"
        )  # TEST-NET-1: any routed address picks the interface
        if not local_ip:
            pytest.skip("no routable interface to send multicast from")
        group, port = "239.192.57.57", _free_udp_port()
        device = FakeScanModeDevice("U_A", _free_udp_port(), acknowledge=True)
        try:
            fake_group = FakeScanModeGroup(
                [device], host=local_ip, port=port, group=group
            )
            await fake_group.start()
        except OSError as exc:
            pytest.skip(f"cannot join a multicast group here: {exc}")
        try:
            async with ScanModeAckListener(local_ip, [device.command_port]) as listener:
                async with ScanModeMulticaster(
                    group, port, local_ip=local_ip
                ) as sender:
                    since = asyncio.get_running_loop().time()
                    await sender.announce_scan_start("Undulator", "Shot #")
                    acks = await listener.wait_for(
                        SCAN_START, ["U_A"], 1.0, since=since
                    )
            if not fake_group.received:
                pytest.skip("multicast loopback is not routed on this host")
            assert set(acks) == {"U_A"}
            assert device.lock_ip == local_ip
        finally:
            await fake_group.stop()
