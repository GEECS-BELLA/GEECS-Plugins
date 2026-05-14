"""Helpers for asking a generator what it would propose next."""

from __future__ import annotations

from typing import Optional, Tuple


def next_candidate(generator) -> Optional[dict]:
    """Ask the generator for one next point and return the full candidate dict.

    Returns ``None`` and prints the underlying error if generation fails
    (e.g., the generator's data store is empty, the optimizer hits a
    numerical edge case, etc.). The dict has one entry per VOCS variable.
    """
    try:
        cands = generator.generate(1)
        if not cands:
            return None
        return cands[0]
    except Exception as exc:
        print(f"  next-candidate generation failed: {exc}")
        return None


def next_candidate_xy(
    generator, var_names: Tuple[str, str]
) -> Optional[Tuple[float, float]]:
    """Project the generator's next candidate onto two named variables.

    Convenience wrapper around :func:`next_candidate` for plotting on a
    2D slice. Returns ``(x, y)`` or ``None`` if generation failed.
    """
    pt = next_candidate(generator)
    if pt is None:
        return None
    vx, vy = var_names
    return float(pt[vx]), float(pt[vy])
