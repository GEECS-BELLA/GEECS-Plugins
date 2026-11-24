"""A full circuit is disconnected, never waited on.

caproto's asyncio server hands every monitor update to each subscribed
circuit's bounded queue from ONE context-wide delivery loop, with an awaited
``put``. A client that stops reading its socket leaves its circuit's queue
full, and that ``put`` then never returns: delivery stops for every client.

:class:`GatewayContext` serves its circuits as :class:`GatewayCircuit`, whose
queue never makes ``put`` wait. When it is full, the circuit is disconnected
(the socket aborted, so caproto's own disconnect path cleans it up) with one
WARNING, and the update is dropped. A client that is alive reconnects and
resubscribes, and receives the current value on subscribe; dropping updates
silently instead would leave it showing stale values as live. Updates for a
circuit already disconnecting are dropped, so no circuit can hold the loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from caproto.asyncio.server import Context, VirtualCircuit

logger = logging.getLogger(__name__)


class _DisconnectWhenFull(asyncio.Queue):
    """A circuit's subscription queue whose ``put`` never waits.

    Parameters
    ----------
    maxsize : int
        The bound caproto gave the circuit's queue.
    circuit : GatewayCircuit
        The circuit it serves, disconnected when the queue is full.
    """

    def __init__(self, maxsize: int, circuit: GatewayCircuit) -> None:
        super().__init__(maxsize)
        self._circuit = circuit

    async def put(self, item: Any) -> None:  # Any: caproto's weakref to a command
        """Queue ``item``, or drop it and disconnect the circuit if full."""
        if self._circuit.disconnecting or not self._circuit.connected:
            return
        if self.full():
            self._circuit.disconnect_unread()
            return
        self.put_nowait(item)


class GatewayCircuit(VirtualCircuit):
    """A caproto circuit that is disconnected when its client stops reading."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.disconnecting = False
        self.subscription_queue = _DisconnectWhenFull(
            self.subscription_queue.maxsize, self
        )

    def disconnect_unread(self) -> None:
        """Abort the client's socket, logging one WARNING, once."""
        if self.disconnecting:
            return
        self.disconnecting = True
        host, port = self.circuit.address[:2]
        logger.warning(
            "CA client %s:%s has not read its monitors: its queue is full "
            "(%d updates, %d subscriptions). Disconnecting it so delivery to "
            "every other client continues; it must reconnect.",
            host,
            port,
            self.subscription_queue.maxsize,
            sum(len(subs) for subs in self.subscriptions.values()),
        )
        # abort, not close: close waits to flush a buffer this client never reads.
        self.client.writer.transport.abort()
        # Empty, so caproto's cancel of the circuit's delivery task lands on a
        # get that cannot complete: on Python 3.11 a cancel that meets a
        # completed get inside wait_for is swallowed, and the task (and with it
        # the circuit's disconnect) would never finish.
        while not self.subscription_queue.empty():
            self.subscription_queue.get_nowait()

    async def _on_disconnect(self) -> None:
        """Run caproto's disconnect, finishing even if the delivery task ends cancelled.

        caproto awaits the circuit's cancelled delivery task here; a task
        cancelled mid-send (a client that stopped reading) re-raises
        ``CancelledError`` into the caller, which then never removes the
        circuit from the context. That error is absorbed unless this task is
        itself being cancelled.
        """
        try:
            await super()._on_disconnect()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise


class GatewayContext(Context):
    """caproto's asyncio server context, serving :class:`GatewayCircuit`."""

    CircuitClass = GatewayCircuit
