"""Column name helpers: flatten MultiIndex, search, and resolve loose specs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Literal, Sequence, TypeAlias

import pandas as pd


def flatten_columns(dataframe: pd.DataFrame) -> List[str]:
    """
    Flatten pandas.DataFrame columns to strings, joining MultiIndex with ':'.

    Parameters
    ----------
    dataframe
        pandas.DataFrame

    Returns
    -------
    list of str
    """
    cols = dataframe.columns
    if getattr(cols, "nlevels", 1) > 1:
        return [":".join(map(str, tup)) for tup in cols.to_list()]
    return list(map(str, cols))


ColumnMatchMode: TypeAlias = Literal[
    "contains", "startswith", "endswith", "regex", "exact"
]


def find_cols(
    dataframe: pd.DataFrame,
    query: str | Sequence[str],
    *,
    mode: ColumnMatchMode = "contains",
    case_sensitive: bool = False,
) -> List[str]:
    """
    Flexible column search.

    Parameters
    ----------
    query
        String or list of strings to search for.
    mode
        Search mode: ``"contains"`` (default), ``"startswith"``, ``"endswith"``,
        ``"regex"``, or ``"exact"``.
    case_sensitive
        If True, match with case sensitivity.

    Returns
    -------
    list of str
        Matching column names (flattened form). May be empty.
    """
    cols = flatten_columns(dataframe)
    originals = cols
    hay = originals if case_sensitive else [c.lower() for c in originals]
    queries = [query] if isinstance(query, str) else list(query)

    matches: set[str] = set()
    for q in queries:
        needle = q if case_sensitive else str(q).lower()

        if mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            pat = re.compile(str(q), flags=flags)
            for og in originals:
                if pat.search(og):
                    matches.add(og)
            continue

        for og, h in zip(originals, hay):
            s = og if case_sensitive else h
            if (
                (mode == "contains" and needle in s)
                or (mode == "startswith" and s.startswith(needle))
                or (mode == "endswith" and s.endswith(needle))
                or (mode == "exact" and s == needle)
            ):
                matches.add(og)

    return sorted(matches)


@dataclass(frozen=True)
class ResolveColResult:
    """Outcome of :func:`resolve_col_detailed`."""

    column: str
    """Chosen column name (flattened string form)."""
    ambiguous: bool
    """True when several columns matched and a tie-break rule picked one."""
    candidates: tuple[str, ...] | None = None
    """When ``ambiguous`` is True, the hit set before tie-breaking (else ``None``)."""


def resolve_col_detailed(
    dataframe: pd.DataFrame,
    spec: str,
    *,
    mode: ColumnMatchMode = "contains",
    case_sensitive: bool = False,
    prefer_exact_ci: bool = True,
) -> ResolveColResult:
    """
    Resolve a loose column spec to a single column name, with ambiguity metadata.

    This function does not log. Callers (e.g. :class:`~geecs_data_utils.scan_data.ScanData`)
    may inspect :attr:`ResolveColResult.ambiguous` and emit warnings if desired.

    Parameters
    ----------
    dataframe
        pandas.DataFrame
    spec
        User-provided spec (partial name, regex pattern when ``mode="regex"``, etc.).
    mode
        Matching strategy used by :func:`find_cols`.
    case_sensitive
        If True, enforce case-sensitive matching for the chosen mode.
    prefer_exact_ci
        Prefer exact (case-insensitive) matches over substring/regex matches.

    Returns
    -------
    ResolveColResult

    Raises
    ------
    ValueError
        If no match is found.
    """
    cols = flatten_columns(dataframe)

    if prefer_exact_ci:
        eq = [c for c in cols if c.lower() == spec.lower()]
        if len(eq) == 1:
            return ResolveColResult(eq[0], ambiguous=False)

    hits = find_cols(dataframe, spec, mode=mode, case_sensitive=case_sensitive)

    if not hits and mode != "contains":
        hits = find_cols(
            dataframe, spec, mode="contains", case_sensitive=case_sensitive
        )

    if not hits:
        raise ValueError(
            f"No columns match spec {spec!r}. "
            f"Available (showing up to 6): {cols[:6]}..."
        )

    exact_ci = [h for h in hits if h.lower() == spec.lower()]
    if len(exact_ci) == 1:
        return ResolveColResult(exact_ci[0], ambiguous=False)

    if len(hits) > 1:
        winner = sorted(hits, key=lambda s: (len(s), s))[0]
        cand = tuple(sorted(hits))
        return ResolveColResult(winner, ambiguous=True, candidates=cand)

    return ResolveColResult(hits[0], ambiguous=False)


def resolve_col(
    dataframe: pd.DataFrame,
    spec: str,
    *,
    mode: ColumnMatchMode = "contains",
    case_sensitive: bool = False,
    prefer_exact_ci: bool = True,
) -> str:
    """
    Resolve a loose column spec to a single best column name.

    Same resolution rules as :func:`resolve_col_detailed` but returns only the
    column string. No logging is performed.

    Parameters
    ----------
    dataframe
        pandas.DataFrame
    spec
        User-provided spec (may be partial/regex per ``mode``).
    mode
        Matching strategy used by :func:`find_cols`.
    case_sensitive
        If True, enforce case-sensitive matching for the chosen mode.
    prefer_exact_ci
        Prefer exact (case-insensitive) matches over substring/regex matches.

    Returns
    -------
    str
        Selected column name.

    Raises
    ------
    ValueError
        If no match is found.
    """
    return resolve_col_detailed(
        dataframe,
        spec,
        mode=mode,
        case_sensitive=case_sensitive,
        prefer_exact_ci=prefer_exact_ci,
    ).column
