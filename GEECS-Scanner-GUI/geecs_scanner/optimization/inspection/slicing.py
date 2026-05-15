"""Choose which variables to slice over and how to fix the rest.

For 2D-slice visualizations of an N-D Xopt model, you need to pick two
variables to render and decide what value to hold every other variable at.
These helpers automate sensible defaults and keep the choice explicit.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
from xopt.vocs import VOCS


def pick_top_varied_pair(data: pd.DataFrame, vocs: VOCS) -> Tuple[str, str]:
    """Return the pair of VOCS variables that varied the most in the data.

    Useful default when you don't have an opinion about which 2 variables
    to plot — the two with the widest observed range are usually the most
    interesting in a tuning run.

    Raises
    ------
    ValueError
        If fewer than two variables have variation in the data.
    """
    ranges = {}
    for name in vocs.variable_names:
        if name in data.columns:
            s = data[name].dropna()
            if len(s) > 1:
                ranges[name] = float(s.max() - s.min())
    top = sorted(ranges, key=ranges.get, reverse=True)
    if len(top) < 2:
        raise ValueError(
            "Need at least two varying variables in the data to choose a plot pair."
        )
    return top[0], top[1]


def best_observed_point(data: pd.DataFrame, vocs: VOCS) -> Dict[str, float]:
    """Reference point at the row with the best observed objective.

    Direction follows ``vocs.objectives[obj]``: ``MAXIMIZE`` picks the row
    with the highest objective; otherwise the lowest.
    """
    obj = vocs.objective_names[0]
    direction = vocs.objectives[obj]
    row = (
        data.loc[data[obj].idxmax()]
        if str(direction).upper() == "MAXIMIZE"
        else data.loc[data[obj].idxmin()]
    )
    return {name: float(row[name]) for name in vocs.variable_names if name in row}


def resolve_slice_and_fixed(
    data: pd.DataFrame,
    vocs: VOCS,
    slice_vars: Optional[Tuple[str, str]] = None,
    fixed_overrides: Optional[Dict[str, float]] = None,
    fixed_default: str = "best",
) -> Tuple[Tuple[str, str], Dict[str, float]]:
    """Resolve which two variables to slice over and how to pin the rest.

    Parameters
    ----------
    data
        The Xopt data DataFrame (or anything with the same columns).
    vocs
        VOCS describing the input space.
    slice_vars
        ``(var_x, var_y)`` to slice over. If ``None``, auto-pick via
        :func:`pick_top_varied_pair`.
    fixed_overrides
        Partial mapping ``{var_name: value}`` to pin specific non-slice
        variables to particular values. Anything not in this mapping
        falls back to ``fixed_default``.
    fixed_default
        ``"best"`` (uses :func:`best_observed_point`) or ``"midpoint"``
        (uses the midpoint of each variable's bounds).

    Returns
    -------
    ((var_x, var_y), fixed_dict)
        The resolved slice pair and a complete dict covering every
        non-slice variable.
    """
    if slice_vars is None:
        slice_vars = pick_top_varied_pair(data, vocs)
    vx, vy = slice_vars

    if fixed_default == "best":
        base = best_observed_point(data, vocs)
    elif fixed_default == "midpoint":
        base = {n: 0.5 * (lo + hi) for n, (lo, hi) in vocs.variables.items()}
    else:
        raise ValueError(f"Unknown fixed_default: {fixed_default!r}")

    overrides = dict(fixed_overrides or {})
    fixed = {
        n: float(overrides.get(n, base[n]))
        for n in vocs.variable_names
        if n not in (vx, vy)
    }
    return (vx, vy), fixed


def print_slice_summary(
    slice_vars: Tuple[str, str],
    fixed: Dict[str, float],
    fixed_overrides: Optional[Dict[str, float]],
    data: pd.DataFrame,
) -> None:
    """Pretty-print which variables are being sliced vs. held.

    Useful when you have many controls — makes it unambiguous what the
    rendered slice means without forcing the caller to inspect the
    fixed-point dict by hand.
    """
    vx, vy = slice_vars
    overrides = fixed_overrides or {}
    print("slicing over:")
    for v in (vx, vy):
        s = data[v].dropna() if v in data.columns else pd.Series([], dtype=float)
        rng = f"{s.min():.4g} .. {s.max():.4g}" if len(s) else "no data"
        print(f"  {v}  (observed range: {rng})")
    if not fixed:
        return
    width = max(len(n) for n in fixed)
    print(f"fixing {len(fixed)} other variable(s):")
    for n, val in fixed.items():
        tag = "override" if n in overrides else "best-observed"
        print(f"  {n:{width}s} = {val:>10.4g}  ({tag})")
