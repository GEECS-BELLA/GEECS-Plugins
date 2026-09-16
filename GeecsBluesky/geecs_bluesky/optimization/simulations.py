"""Synthetic observables for beam-free acceptance with real movable readbacks."""

from collections.abc import Mapping


def aline_x_com(scalars: Mapping[str, float]) -> float:
    """Return a deterministic alignment centroid from s1v and emq measurements."""
    return 100.0 * (scalars["s1v"] - 1.0) * (1.0 + scalars["emq"])
