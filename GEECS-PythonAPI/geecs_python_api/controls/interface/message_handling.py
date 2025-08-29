"""Thread-safe network message helpers with logging-based error handling (ErrorAPI deprecated)."""

from __future__ import annotations

import inspect
import logging
import queue
import threading
from typing import Optional, List

from geecs_python_api.controls.interface.geecs_errors import ErrorAPI  # deprecated shim
from geecs_python_api.controls.interface.event_handler import EventHandler

logger = logging.getLogger(__name__)


class NetworkMessage:
    """Lightweight container for network messages with optional legacy error shim."""

    def __init__(
        self,
        tag: str = "",
        stamp: str = "",
        msg: str = "",
        err: Optional[ErrorAPI] = None,
    ) -> None:
        self.tag: str = tag
        self.stamp: str = stamp
        self.msg: str = msg
        self.err: Optional[ErrorAPI] = err


def next_msg(
    a_queue: queue.Queue, block: bool = False, timeout: Optional[float] = None
) -> NetworkMessage:
    """Pop one NetworkMessage from the queue or return an empty message on timeout/error."""
    try:
        net_msg: NetworkMessage = a_queue.get(block=block, timeout=timeout)
        a_queue.task_done()
        return net_msg
    except queue.Empty:
        logger.debug("next_msg timed out (block=%s, timeout=%s)", block, timeout)
        return NetworkMessage()
    except Exception:
        caller = inspect.stack()[1].function if len(inspect.stack()) > 1 else "unknown"
        logger.exception('Failed to dequeue next message (caller "%s")', caller)
        return NetworkMessage()


def flush_queue(a_queue: queue.Queue) -> List[NetworkMessage]:
    """Drain all items from the queue and return them as a list."""
    flushed: List[NetworkMessage] = []
    try:
        while True:
            try:
                item = a_queue.get_nowait()
            except queue.Empty:
                break
            else:
                flushed.append(item)
                a_queue.task_done()
    except Exception:
        logger.exception("Unexpected error while flushing queue")
    return flushed


# Legacy function maintained for backward compatibility with previous print-based handlers.
def async_msg_handler(
    message: NetworkMessage, a_queue: Optional[queue.Queue] = None
) -> None:
    """Legacy async handler that logs the message and optionally dequeues one item."""
    try:
        if a_queue is not None:
            _ = next_msg(a_queue)  # legacy behavior: consume one pending item

        has_legacy = bool(
            getattr(message.err, "is_error", False)
            or getattr(message.err, "is_warning", False)
        )
        if has_legacy:
            # Let the deprecated shim format/log as it used to, but prefer direct logging going forward.
            try:
                # Trigger the shim's handler without raising.
                message.err.error_handler()
            except Exception:
                logger.exception("ErrorAPI handler raised unexpectedly")
            return

        if message.stamp:
            logger.info(
                'Async UDP response to "%s":\n\t%s\n\t%s',
                message.tag,
                message.stamp,
                message.msg,
            )
        else:
            logger.info(
                'Async UDP response to "%s":\n\tno timestamp\n\t%s',
                message.tag,
                message.msg,
            )
    except Exception:
        logger.exception('Exception in async_msg_handler for tag "%s"', message.tag)


def broadcast_msg(
    net_msg: NetworkMessage,
    notifier: Optional[threading.Condition] = None,
    queue_msgs: Optional[queue.Queue] = None,
    publisher: Optional[EventHandler] = None,
    event_name: str = "",
) -> None:
    """Queue the message, notify any waiters, and publish via EventHandler if provided."""
    if notifier is not None:
        with notifier:
            if queue_msgs is not None:
                queue_msgs.put(net_msg)
            notifier.notify_all()
            if publisher is not None and event_name.strip():
                publisher.publish(event_name, net_msg, queue_msgs)
    else:
        if queue_msgs is not None:
            queue_msgs.put(net_msg)
        if publisher is not None and event_name.strip():
            publisher.publish(event_name, net_msg, queue_msgs)
