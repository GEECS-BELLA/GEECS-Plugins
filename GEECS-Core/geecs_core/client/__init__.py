"""Entry-level synchronous client layer (DESIGN.md layer 3).

The one place in geecs-core where synchronous callers bridge to the async
transport. Scripts and notebooks import :class:`GeecsDevice`; services with
their own event loop use :mod:`geecs_core.transport` directly instead.
"""

from .geecs_device import GeecsDevice

__all__ = ["GeecsDevice"]
