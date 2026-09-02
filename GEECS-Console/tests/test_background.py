"""The BackgroundResult GUI hop (services/background.py, 0.28.0).

The daemon thread never emits toward the consumer: the result hops onto
the GUI thread through the worker's own queued signal and ``result_ready``
is emitted there.  These pin the properties the mechanism exists for —
where the consumer's slot runs, that a raising callable still releases
the in-flight hold, that overlapping calls on one worker keep the hold
until the last one lands, and that a consumer dropped mid-flight is safe.
"""

from __future__ import annotations

import gc
import threading

from PySide6.QtCore import QObject, Qt, Slot

from geecs_console.services import background
from geecs_console.services.background import BackgroundResult


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
