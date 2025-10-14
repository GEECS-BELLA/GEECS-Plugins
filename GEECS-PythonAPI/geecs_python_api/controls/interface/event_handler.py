"""
Simple pub/sub event handling example with logging integration.

Provides an `EventHandler` class for registering and publishing events,
a demo `SubscriberExample`, and a helper for variable name lookup.
Uses the standard Python logging framework instead of custom error handling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SubscriberExample:
    """Simple demo subscriber that logs received messages."""

    def __init__(self, name: str):
        """Initialize the subscriber with a display name."""
        self.name = name

    def print_message(self, message: str) -> None:
        """Log the received message."""
        logger.info('handler "%s" received message: "%s"', self.name, message)


class EventHandler:
    """Lightweight pub/sub event hub mapping event names to subscriber callables."""

    def __init__(self, events_names: Iterable[str]):
        """Create an event registry with optional initial events."""
        self.events: Dict[str, Dict[str, Optional[Callable]]] = {
            event_name: {} for event_name in events_names
        }

    def add_events(self, events_names: Iterable[str]) -> None:
        """Add events to the registry."""
        for event_name in events_names:
            if event_name not in self.events:
                self.events[event_name] = {}
                logger.debug("added event %s", event_name)

    def delete_events(self, events_names: Iterable[str]) -> None:
        """Delete events from the registry (no-op if absent)."""
        for event_name in events_names:
            removed = self.events.pop(event_name, None)
            if removed is not None:
                logger.debug("deleted event %s", event_name)

    def get_subscribers(self, event_name: str) -> Dict[str, Optional[Callable]]:
        """Return the subscriber map for an event."""
        if event_name not in self.events:
            self.events[event_name] = {}
            logger.debug("auto-created missing event %s", event_name)
        return self.events[event_name]

    def register(
        self,
        event_name: str,
        subscriber_name: str,
        subscriber_method: Optional[Callable] = None,
    ) -> None:
        """Register a subscriber callable under an event."""
        self.get_subscribers(event_name)[subscriber_name] = subscriber_method
        logger.debug('registered "%s" to event %s', subscriber_name, event_name)

    def unregister(self, event_name: str, subscriber_name: str) -> None:
        """Unregister a subscriber from an event (no-op if absent)."""
        subs = self.events.get(event_name)
        if not subs:
            logger.warning("attempted to unregister from unknown event %s", event_name)
            return
        subs.pop(subscriber_name, None)
        logger.debug('unregistered "%s" from event %s', subscriber_name, event_name)

    def unregister_all(self) -> None:
        """Remove all subscribers from all events."""
        for _, subscriptions in self.events.items():
            subscriptions.clear()
        logger.debug("cleared all subscriptions")

    def publish(self, event_name: str, *args, **kwargs) -> None:
        """Invoke all subscribers for a single event with provided args."""
        subs = self.events.get(event_name, {})
        if not subs:
            logger.debug("publish called on event %s with no subscribers", event_name)
        for subscriber_name, subscriber_method in subs.items():
            if subscriber_method is None:
                continue
            try:
                subscriber_method(*args, **kwargs)
            except Exception:
                logger.exception(
                    'subscriber "%s" failed for event %s', subscriber_name, event_name
                )

    def publish_all(self, *args, **kwargs) -> None:
        """Invoke all subscribers across all events with provided args."""
        for event_name, subscriptions in self.events.items():
            for subscriber_name, subscriber_method in subscriptions.items():
                if subscriber_method is None:
                    continue
                try:
                    subscriber_method(*args, **kwargs)
                except Exception:
                    logger.exception(
                        'subscriber "%s" failed for event %s',
                        subscriber_name,
                        event_name,
                    )


def var_to_name(var) -> Optional[str]:
    """Best-effort variable-name lookup in globals (debug helper; avoid in production)."""
    # noinspection PyTypeChecker
    dict_vars = dict(globals().items())
    for name, value in dict_vars.items():
        if value is var:
            return name
    return None


if __name__ == "__main__":
    # Library code should *not* configure logging; demo/CLI/GUI may:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logger.info("creating publisher...")
    registration = EventHandler(["new message"])

    logger.info("creating handler...")
    message_handler = SubscriberExample("Handler")

    logger.info('registering for incoming "new message"...')
    registration.register(
        "new message",
        var_to_name(message_handler) or "handler",
        message_handler.print_message,
    )

    time.sleep(1)
    registration.publish("new message", "this is a test message!")
    registration.publish_all("this is another test message!")
