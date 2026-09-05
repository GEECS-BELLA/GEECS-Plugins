"""The GUI hop of both background workers (services/background.py).

The daemon thread never emits toward the consumer: the result hops onto
the GUI thread through the worker's own queued signal and the public
signal is emitted there — ``BackgroundResult.result_ready`` since 0.28.0,
``HealthPoller.report_ready`` since 0.28.2 (#767).  These pin the
properties the mechanism exists for — where the consumer's slot runs, that
a raising callable still releases the in-flight hold, that overlapping
calls on one worker keep the hold until the last one lands, that the
poller's in-flight skip lasts until the report has landed, and that a
consumer dropped mid-flight is safe.
"""

from __future__ import annotations

import gc
import threading
import time

from PySide6.QtCore import QObject, Qt, Slot

from geecs_console.services import background
from geecs_console.services.background import BackgroundResult, HealthPoller


class Consumer(QObject):
    """A throwaway receiver recording the thread its slot ran on."""

    def __init__(self):
        super().__init__()
        self.results: list = []
        self.threads: list[int] = []

    @Slot(object)
    def take(self, result):
        self.results.append(result)
        self.threads.append(threading.get_ident())


def _drain(qtbot, predicate, timeout=3000):
    qtbot.waitUntil(predicate, timeout=timeout)


class TestGuiHop:
    def test_result_ready_is_emitted_on_the_gui_thread(self, qtbot):
        worker = BackgroundResult()
        consumer = Consumer()
        # A DIRECT connection runs the slot on the *emitting* thread — so
        # the recorded thread is where result_ready was emitted from.
        worker.result_ready.connect(consumer.take, Qt.ConnectionType.DirectConnection)
        worker.run_async(lambda: 42, name="hop-test")
        _drain(qtbot, lambda: consumer.results == [42])
        assert consumer.threads == [threading.main_thread().ident]

    def test_raising_callable_emits_nothing_but_releases_the_hold(self, qtbot):
        worker = BackgroundResult()
        consumer = Consumer()
        worker.result_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)

        def boom():
            raise RuntimeError("nope")

        worker.run_async(boom, name="hop-raise")
        _drain(qtbot, lambda: worker not in background._INFLIGHT)
        assert consumer.results == []

    def test_overlapping_calls_hold_the_worker_until_the_last_lands(self, qtbot):
        worker = BackgroundResult()
        consumer = Consumer()
        worker.result_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        gate = threading.Event()

        def slow():
            gate.wait(3.0)
            return "slow"

        worker.run_async(slow, name="hop-slow")
        worker.run_async(lambda: "fast", name="hop-fast")
        assert background._INFLIGHT[worker] == 2
        _drain(qtbot, lambda: consumer.results == ["fast"])
        # The fast landing released ONE hold; the slow call keeps the worker.
        _drain(qtbot, lambda: background._INFLIGHT.get(worker) == 1)
        gate.set()
        _drain(qtbot, lambda: consumer.results == ["fast", "slow"])
        _drain(qtbot, lambda: worker not in background._INFLIGHT)

    def test_consumer_dropped_mid_flight_is_safe(self, qtbot):
        """The teardown race that motivated the hop: the consumer dies while
        the call is out; the result lands nowhere and nothing crashes."""
        worker = BackgroundResult()
        consumer = Consumer()
        worker.result_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        gate = threading.Event()

        def slow():
            gate.wait(3.0)
            return "late"

        worker.run_async(slow, name="hop-orphan")
        del consumer
        gc.collect()
        gate.set()
        _drain(qtbot, lambda: worker not in background._INFLIGHT)


class GatedProbe:
    """A probe whose ``poll()`` blocks on a gate and counts its calls."""

    def __init__(self, report="report"):
        self.report = report
        self.gate = threading.Event()
        self.gate.set()  # blocks only when a test clears it
        self.calls = 0

    def poll(self):
        self.calls += 1
        self.gate.wait(3.0)
        if isinstance(self.report, Exception):
            raise self.report
        return self.report


def _wait_for_poll_thread_exit(timeout=3.0):
    """Block (no event pumping) until no ``console-health-poll`` thread is alive.

    The daemon thread exits right after hopping its report, so once it is
    gone the report is posted but not yet delivered — the window in which
    the pre-0.28.2 poller had already cleared ``_busy``.
    """
    deadline = time.monotonic() + timeout
    while any(t.name == "console-health-poll" for t in threading.enumerate()):
        assert time.monotonic() < deadline, "poll thread did not exit"
        time.sleep(0.005)


class TestHealthPollerGuiHop:
    def test_report_ready_is_emitted_on_the_gui_thread(self, qtbot):
        poller = HealthPoller(GatedProbe("r"))
        consumer = Consumer()
        # DIRECT: the slot runs where report_ready was emitted from.
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.DirectConnection)
        poller.poll_async()
        _drain(qtbot, lambda: consumer.results == ["r"])
        assert consumer.threads == [threading.main_thread().ident]

    def test_poll_in_flight_is_skipped_until_the_report_has_landed(self, qtbot):
        probe = GatedProbe("r")
        probe.gate.clear()
        poller = HealthPoller(probe)
        consumer = Consumer()
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        poller.poll_async()
        poller.poll_async()  # skipped: one is out
        _drain(qtbot, lambda: probe.calls == 1)
        assert background._INFLIGHT[poller] == 1
        probe.gate.set()
        # The poll has returned and the report is posted but NOT delivered
        # (no events pumped): the poller must still count as in flight —
        # clearing ``_busy`` on the daemon thread would poll again here.
        _wait_for_poll_thread_exit()
        poller.poll_async()
        assert probe.calls == 1
        assert background._INFLIGHT[poller] == 1
        _drain(qtbot, lambda: consumer.results == ["r"])
        _drain(qtbot, lambda: poller not in background._INFLIGHT)
        assert probe.calls == 1
        # Landed ⇒ no longer busy: the next tick polls again.
        poller.poll_async()
        _drain(qtbot, lambda: consumer.results == ["r", "r"])
        assert probe.calls == 2

    def test_raising_probe_emits_nothing_but_releases_hold_and_busy(self, qtbot):
        probe = GatedProbe(RuntimeError("probe down"))
        poller = HealthPoller(probe)
        consumer = Consumer()
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        poller.poll_async()
        _drain(qtbot, lambda: probe.calls == 1 and poller not in background._INFLIGHT)
        assert consumer.results == []
        assert poller._busy is False

    def test_health_reports_are_stamped_with_the_poll_sequence(self, qtbot):
        from geecs_console.services.health import HealthReport, HealthStatus

        probe = GatedProbe(HealthReport(gateway=HealthStatus.OK))
        poller = HealthPoller(probe)
        consumer = Consumer()
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        assert poller.polls_started == 0
        poller.poll_async()
        _drain(qtbot, lambda: len(consumer.results) == 1)
        poller.poll_async()
        _drain(qtbot, lambda: len(consumer.results) == 2)
        assert [r.sequence for r in consumer.results] == [1, 2]
        assert all(r.gateway is HealthStatus.OK for r in consumer.results)
        assert poller.polls_started == 2
        assert probe.report.sequence == 0  # the probe's own report is untouched

    def test_skipped_polls_do_not_take_a_sequence(self, qtbot):
        probe = GatedProbe("r")
        probe.gate.clear()
        poller = HealthPoller(probe)
        consumer = Consumer()
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        poller.poll_async()
        poller.poll_async()  # skipped: one is out
        assert poller.polls_started == 1
        probe.gate.set()
        _drain(qtbot, lambda: consumer.results == ["r"])  # non-reports pass through

    def test_none_report_is_not_emitted(self, qtbot):
        probe = GatedProbe(None)
        poller = HealthPoller(probe)
        consumer = Consumer()
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        poller.poll_async()
        _drain(qtbot, lambda: poller not in background._INFLIGHT)
        assert consumer.results == []
        assert poller._busy is False

    def test_consumer_dropped_mid_flight_is_safe(self, qtbot):
        """The #767 teardown race: the window dies while a poll is out."""
        probe = GatedProbe("late")
        probe.gate.clear()
        poller = HealthPoller(probe)
        consumer = Consumer()
        poller.report_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        poller.poll_async()
        del consumer
        gc.collect()
        probe.gate.set()
        _drain(qtbot, lambda: poller not in background._INFLIGHT)

    def test_poller_dropped_mid_flight_survives_until_the_hop_lands(self, qtbot):
        """The hold outlives the owner's reference: a poller whose only
        Python reference is dropped mid-poll is not torn down under the
        pending hop (the C++ object would go with the wrapper)."""
        probe = GatedProbe("late")
        probe.gate.clear()
        poller = HealthPoller(probe)
        poller.poll_async()
        del poller
        gc.collect()
        assert (
            len([w for w in background._INFLIGHT if isinstance(w, HealthPoller)]) == 1
        )
        probe.gate.set()
        _drain(
            qtbot,
            lambda: not [
                w for w in background._INFLIGHT if isinstance(w, HealthPoller)
            ],
        )


class TestGuiRelay:
    """The long-lived producer's hop (#787): post from any thread, land on the GUI."""

    def test_posted_payloads_land_on_the_gui_thread_in_order(self, qtbot):
        from geecs_console.services.background import GuiRelay

        relay = GuiRelay()
        consumer = Consumer()
        relay.delivered.connect(consumer.take, Qt.ConnectionType.QueuedConnection)

        def producer():
            for index in range(5):
                relay.post((index, index * 1.5))

        thread = threading.Thread(target=producer)
        thread.start()
        thread.join()
        _drain(qtbot, lambda: len(consumer.results) == 5)
        assert consumer.results == [(i, i * 1.5) for i in range(5)]
        assert set(consumer.threads) == {threading.get_ident()}
        _drain(qtbot, lambda: relay not in background._INFLIGHT)

    def test_close_drops_in_flight_payloads_and_releases_the_hold(self, qtbot):
        from geecs_console.services.background import GuiRelay

        relay = GuiRelay()
        consumer = Consumer()
        relay.delivered.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        relay.post("in flight")
        relay.close()
        relay.post("after close")  # never even posted
        _drain(qtbot, lambda: relay not in background._INFLIGHT)
        assert consumer.results == []

    def test_concurrent_posters_keep_the_hold_count_consistent(self, qtbot):
        # Two threads posting at once exercise the locked read-modify-write
        # of the in-flight counter; every hold must be released.
        from geecs_console.services.background import GuiRelay

        relay = GuiRelay()
        received: list = []
        relay.delivered.connect(received.append, Qt.ConnectionType.QueuedConnection)
        threads = [
            threading.Thread(target=lambda: [relay.post(n) for n in range(200)])
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _drain(qtbot, lambda: len(received) == 800, timeout=10000)
        _drain(qtbot, lambda: relay not in background._INFLIGHT)


class TestDisconnectQuietly:
    def test_never_connected_signal_does_not_warn(self, qtbot, recwarn):
        from geecs_console.services.background import disconnect_quietly

        worker = BackgroundResult()
        disconnect_quietly(worker.result_ready)
        assert not [w for w in recwarn if "Failed to disconnect" in str(w.message)]

    def test_detaches_a_connected_slot(self, qtbot):
        from geecs_console.services.background import disconnect_quietly

        worker = BackgroundResult()
        consumer = Consumer()
        worker.result_ready.connect(consumer.take, Qt.ConnectionType.QueuedConnection)
        disconnect_quietly(worker.result_ready, consumer.take)
        worker.run_async(lambda: 1, "t")
        _drain(qtbot, lambda: worker not in background._INFLIGHT)
        assert consumer.results == []
