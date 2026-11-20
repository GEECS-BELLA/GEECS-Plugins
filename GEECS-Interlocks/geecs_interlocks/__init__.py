"""GEECS software interlocks: watch devices and external signals against a named configuration and report the verdict.

The engine (:class:`~geecs_interlocks.engine.Engine`) evaluates every
interlock on one tick (:mod:`~geecs_interlocks.monitor`) over readings from
GEECS devices (:mod:`~geecs_interlocks.sources`, built on
``geecs_core.client.GeecsDevice``) and UDP status packets
(:mod:`~geecs_interlocks.external`) and serves the verdict to Master Control as a
length-prefixed text stream (:mod:`~geecs_interlocks.wire`). It only
reports: what a trip stops is Master Control's decision.
"""

from geecs_interlocks.conditions import Inputs, Verdict, evaluate
from geecs_interlocks.engine import Engine
from geecs_interlocks.monitor import InterlockStatus, Monitor, Snapshot

__all__ = [
    "Engine",
    "Inputs",
    "InterlockStatus",
    "Monitor",
    "Snapshot",
    "Verdict",
    "evaluate",
]
