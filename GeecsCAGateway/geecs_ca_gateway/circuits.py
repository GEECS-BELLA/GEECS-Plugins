"""A client that takes no monitor update for a grace period is disconnected.

caproto's asyncio server hands every monitor update to each subscribed
circuit's bounded queue from ONE context-wide delivery loop, with an awaited
``put``. When a circuit's queue is full that ``put`` waits, which is also the
loop's yield to the circuit's own task that drains the queue; a client that
has stopped reading its socket never lets that task take an entry, and the
``put`` would wait forever: delivery stops for every client.

:class:`GatewayContext` serves its circuits as :class:`GatewayCircuit`, whose
queue's ``put`` waits as caproto's does, but for :data:`UNREAD_GRACE_S` at
most. A burst, or a client that reads slowly, is paced as by caproto. A
circuit that takes nothing for the whole grace is disconnected (its socket
aborted, so caproto's own disconnect path cleans it up) with one WARNING, and
the update is dropped. A client that is alive reconnects, resubscribes, and
receives the current value on subscribe; dropping updates silently instead
would leave it showing stale values as live. A disconnecting circuit's queue
accepts nothing, so each stuck client costs every other client at most one
grace of delivery.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any

from caproto.asyncio.server import Context, VirtualCircuit

logger = logging.getLogger(__name__)

#: Seconds a full circuit queue may take no entry before its client is
#: disconnected. A circuit whose delivery task can send takes an entry within
#: milliseconds of the loop yielding; one blocked on a full socket takes none.
UNREAD_GRACE_S = 2.0

#: Seconds between cancels of a disconnected circuit's delivery task.
_CANCEL_RETRY_S = 0.1


class _BoundedWaitQueue(asyncio.Queue):
    """A circuit's subscription queue whose ``put`` waits for a grace at most.

    Parameters
    ----------
    maxsize : int
        The bound caproto gave the circuit's queue.
    circuit : GatewayCircuit
        The circuit it serves, disconnected when the grace runs out.
    """

    def __init__(self, maxsize: int, circuit: GatewayCircuit) -> None:
        super().__init__(maxsize)
        self._circuit = circuit

    async def put(self, item: Any) -> None:  # Any: caproto's weakref to a command
        """Queue ``item``; if the queue stays full for the grace, disconnect."""
        if self._circuit.disconnecting:
            return
        if not self.full():  # caproto's put, without a task for wait_for
            self.put_nowait(item)
            return
        try:
            await asyncio.wait_for(super().put(item), UNREAD_GRACE_S)
        except asyncio.TimeoutError:  # the item was not queued
            self._circuit.disconnect_unread()

    def _put(self, item: Any) -> None:  # Any: as above
        if not self._circuit.disconnecting:
            super()._put(item)


class GatewayCircuit(VirtualCircuit):
    """A caproto circuit that is disconnected when its client stops reading."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.disconnecting = False
        self.subscription_queue = _BoundedWaitQueue(
            self.subscription_queue.maxsize, self
        )

    def disconnect_unread(self) -> None:
        """Abort the client's socket, logging one WARNING, once."""
        if self.disconnecting:
            return
        self.disconnecting = True
        host, port = self.circuit.address[:2]
        logger.warning(
            "CA client %s:%s took no monitor update for %.1f s with its queue "
            "full (%d updates, %d subscriptions). Disconnecting it so delivery "
            "to every other client continues; it must reconnect.",
            host,
            port,
            UNREAD_GRACE_S,
            self.subscription_queue.maxsize,
            sum(len(subs) for subs in self.subscriptions.values()),
        )
        # abort, not close: close waits to flush a buffer this client never reads.
        self.client.writer.transport.abort()

    async def _on_disconnect(self) -> None:
        """Run caproto's disconnect, then end the delivery task for certain.

        caproto cancels the circuit's delivery task once and awaits it. On
        Python 3.11 a cancel that meets a ``get`` already completed inside
        ``wait_for`` is swallowed, so the task runs on, then waits on an empty
        queue for good, and the disconnect awaiting it never finishes (the
        circuit stays in ``context.circuits``). Here the task is taken from
        caproto, which then skips its cancel, and cancelled until it ends; the
        queue accepts nothing meanwhile.
        """
        self.disconnecting = True
        task, self._sub_task = self._sub_task, None
        await super()._on_disconnect()
        if task is None or task is asyncio.current_task():
            return  # the delivery task itself ends after this call
        while not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=_CANCEL_RETRY_S)


class _SnapshotDeque(deque):
    """A deque iterated over a copy, so it may change while it is iterated."""

    def __iter__(self) -> Iterator[Any]:  # Any: caproto's Subscription
        return iter(list(deque.__iter__(self)))


class GatewayContext(Context):
    """caproto's asyncio server context, serving :class:`GatewayCircuit`.

    The delivery loop iterates a subscription spec's subscribers while it
    waits on a full circuit queue; meanwhile other circuits subscribe,
    unsubscribe and disconnect, which would end the loop with "deque mutated
    during iteration". The subscribers are therefore iterated over a copy.
    """

    CircuitClass = GatewayCircuit

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.subscriptions = defaultdict(_SnapshotDeque)
