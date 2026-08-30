"""OR-of-AND row filters — GEECSplotter's filter model, one home.

The analysis-tabs filter vocabulary
(``Planning/data_portal/03_analysis_tabs_design.md``, W1b): named
**groups** of AND conditions, groups OR together, each condition a
``within``/``outside`` numeric bounds pair on one column — exactly the
LabVIEW tool's "outer indexes OR, inner AND" semantics, as Pydantic
models so the same object serializes into an endpoint query, a saved
analysis config, and the "show the code" snippet.

This module deliberately does **not** lower onto
:func:`geecs_data_utils.data.cleaning.apply_row_filters` (the 03 design
doc's original sketch): a mask-returning, OR-capable, explicit-NaN
primitive cannot be built on an AND-only frame-returning kernel, so the
comparisons are written inline here.  The legacy tuple vocabulary stays
for its existing consumers (``DatasetBuilder.prepare_frame`` still
takes ``list[RowFilterSpec]``; composing :class:`RowFilters` into the
dataset pipeline is future work).  The NaN policy is a declared field
precisely because the legacy behavior was op-dependent (comparisons
drop NaN rows, ``!=`` keeps them).

Coercion follows the package's established doctrine
(``tiled_schema.numeric_series``): datetime/timedelta columns and
duplicated column labels are refused loudly rather than silently
compared as nanosecond integers.

Porting fidelity note: GEECSplotter has per-*condition* enable toggles;
this model carries enable at the *group* level (the 03 design's
explicit choice) — importing a saved LabVIEW filter set drops disabled
condition rows rather than carrying their disabled state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class FilterCondition(BaseModel):
    """One numeric bounds condition on one column.

    ``within`` passes rows with ``low <= value <= high`` (inclusive);
    ``outside`` passes the complement.  NaN handling is the parent
    :class:`RowFilters`' ``nan_policy`` — deliberately not per-condition.
    """

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name, verbatim (any provenance).")
    mode: Literal["within", "outside"] = Field(
        default="within", description="Pass inside the bounds, or outside them."
    )
    low: float = Field(description="Lower bound (inclusive).")
    high: float = Field(description="Upper bound (inclusive).")

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "FilterCondition":
        import math

        # NaN bounds silently match nothing (NaN comparisons are all
        # False, so low > high never trips) — refuse them here. ±inf
        # stays legal: half-open ranges are legitimate.
        if math.isnan(self.low) or math.isnan(self.high):
            raise ValueError(f"NaN bound for column {self.column!r}")
        if self.low > self.high:
            raise ValueError(
                f"low ({self.low}) must not exceed high ({self.high}) "
                f"for column {self.column!r}"
            )
        return self


class FilterGroup(BaseModel):
    """A named group of AND conditions (one GEECSplotter filter block)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="", description='Display name; UIs default blanks to "filter N".'
    )
    enabled: bool = Field(default=True, description="Disabled groups are ignored.")
    conditions: List[FilterCondition] = Field(
        default_factory=list, description="ANDed together; an empty group is ignored."
    )


class RowFilters(BaseModel):
    """The full filter selection: groups OR together.

    With no active group (none enabled, or none with conditions) every
    row passes — "no filters" is the identity, never an empty result.
    """

    model_config = ConfigDict(extra="forbid")

    groups: List[FilterGroup] = Field(default_factory=list)
    nan_policy: Literal["exclude", "keep"] = Field(
        default="exclude",
        description=(
            "A NaN in a condition's column: 'exclude' fails that condition "
            "(the row needs another OR group to pass); 'keep' passes it."
        ),
    )

    def active_groups(self) -> List[FilterGroup]:
        """The groups that participate: enabled and non-empty."""
        return [g for g in self.groups if g.enabled and g.conditions]


def _condition_mask(
    frame: "pd.DataFrame", condition: FilterCondition, nan_policy: str
) -> "pd.Series":
    """Boolean mask for one condition."""
    if condition.column not in frame.columns:
        raise ValueError(
            f"filter column {condition.column!r} not in the frame "
            f"({len(frame.columns)} columns)"
        )
    import pandas as pd

    raw = frame[condition.column]
    if not isinstance(raw, pd.Series):
        # Duplicated column label — refuse with OUR message, not an
        # opaque pandas TypeError (numeric_series parity).
        raise ValueError(f"filter column {condition.column!r} is duplicated")
    if pd.api.types.is_datetime64_any_dtype(raw) or pd.api.types.is_timedelta64_dtype(
        raw
    ):
        # Coercing would compare ~1e18 nanosecond integers against the
        # bounds — wrong-but-plausible results (numeric_series doctrine).
        raise ValueError(
            f"filter column {condition.column!r} is a datetime/timedelta "
            "column — not filterable by numeric bounds"
        )
    series = pd.to_numeric(raw, errors="coerce")
    inside = (series >= condition.low) & (series <= condition.high)
    mask = inside if condition.mode == "within" else ~inside
    is_nan = series.isna()
    if nan_policy == "keep":
        return mask | is_nan
    # 'exclude': NaN fails the condition under BOTH modes (the ~inside
    # complement would otherwise silently pass NaN rows for 'outside').
    return mask & ~is_nan


def filter_mask(frame: "pd.DataFrame", filters: RowFilters) -> "pd.Series":
    """The boolean pass mask for *filters* over *frame*.

    The composable primitive: ``mask.sum()`` is the live pass count,
    ``frame[mask]`` the filtered view.  Groups AND internally and OR
    together; no active groups → all-True.

    Parameters
    ----------
    frame : pandas.DataFrame
        The scan frame (any provenance mix).
    filters : RowFilters
        The filter selection.

    Returns
    -------
    pandas.Series
        Boolean, aligned to ``frame.index``.
    """
    import pandas as pd

    active = filters.active_groups()
    if not active:
        return pd.Series(True, index=frame.index)
    combined = pd.Series(False, index=frame.index)
    for group in active:
        group_mask = pd.Series(True, index=frame.index)
        for condition in group.conditions:
            group_mask &= _condition_mask(frame, condition, filters.nan_policy)
        combined |= group_mask
    return combined


def apply_filters(frame: "pd.DataFrame", filters: RowFilters) -> "pd.DataFrame":
    """Return the rows of *frame* passing *filters* (a filtered copy).

    Parameters
    ----------
    frame : pandas.DataFrame
        The scan frame.
    filters : RowFilters
        The filter selection.

    Returns
    -------
    pandas.DataFrame
        ``frame[filter_mask(frame, filters)]``.
    """
    return frame[filter_mask(frame, filters)]
