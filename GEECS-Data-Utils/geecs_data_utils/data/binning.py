"""Per-bin aggregation — the pure core of the binned plot.

The analysis-tabs flagship primitive
(``Planning/data_portal/03_analysis_tabs_design.md``, W1c):
:func:`bin_frame` takes any frame plus a :class:`BinningConfig` and
returns centers with asymmetric error bands — frame in, result out,
no instance state.  It replaces the former stateful
``ScanData.binned_scalars`` implementation (145 lines on a property
with an ``id(df)`` cache key and a fragmented-insert output); the
``BinningConfig`` vocabulary — the good part — moved here unchanged,
and ``ScanData`` keeps thin compatibility wrappers.

Every number a web endpoint serves from this module is reproducible in
a notebook by the same one call — the arc's reproducibility doctrine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

AggT = Literal["mean", "median"]
ErrT = Literal["std", "stderr", "mad", "iqr", "percentile"]
DropT = Literal["any", "all"]


@dataclass(frozen=True)
class BinningConfig:
    """Configuration for per-bin aggregation.

    The binner computes, for each selected value column in each bin,
    three subcolumns: ``(value, "center")``, ``(value, "err_low")``,
    ``(value, "err_high")``.

    Parameters
    ----------
    bin_col
        Column name giving bin identity (or the source for numeric
        binning).
    value_cols
        Columns to aggregate; ``None`` selects all numeric columns
        except ``"Shotnumber"`` (the bin source column IS included —
        its per-bin center is the natural X axis).
    agg
        Center estimator per bin: ``"mean"`` or ``"median"``.
    err
        Error definition per bin:

        - ``"std"``       : sample standard deviation (symmetric)
        - ``"stderr"``    : std / sqrt(N) (symmetric)
        - ``"mad"``       : median absolute deviation (×1.4826 when
          ``scale_to_sigma``; symmetric)
        - ``"iqr"``       : inter-quantile offsets using `percentiles`
          (asymmetric: ``err_low = center − q_low``,
          ``err_high = q_high − center``, clipped at 0)
        - ``"percentile"``: same as ``"iqr"`` with arbitrary bounds
    ddof
        Degrees of freedom for ``"std"``/``"stderr"``.
    percentiles
        ``(low, high)`` quantiles for the asymmetric methods.
    scale_to_sigma
        Scale ``"mad"`` by 1.4826 toward σ.  Ignored elsewhere.
    min_count
        Minimum samples for a bin to be reported.
    dropna
        Row policy before grouping over the selected value columns:
        ``"any"`` drops a row when any (non-all-NaN) value column is NA;
        ``"all"`` drops only all-NA rows.  Columns that are entirely NaN
        never cause drops (they aggregate to NaN instead).

    Numeric binning (optional)
    --------------------------
    bin_edges
        Explicit edges for ``pd.cut``.
    bin_width
        Uniform width; edges built from the data range (or `origin`).
    quantile_bins
        e.g. 10 → deciles via ``pd.qcut``.
    right
        Include the right edge (``pd.cut`` semantics).
    label
        Numeric-bin labels: ``"interval"``, ``"left"``, ``"center"``
        (default), or ``"right"``.
    origin
        Starting point for width-bins; data minimum when ``None``.
    """

    bin_col: str = "Bin #"
    value_cols: Optional[Iterable[str]] = None
    agg: AggT = "median"
    err: ErrT = "iqr"
    ddof: int = 1
    percentiles: Tuple[float, float] = (0.25, 0.75)
    scale_to_sigma: bool = False
    min_count: int = 1
    dropna: DropT = "any"

    # numeric binning options
    bin_edges: Optional[Sequence[float]] = None
    bin_width: Optional[float] = None
    quantile_bins: Optional[int] = None
    right: bool = True
    label: Literal["interval", "left", "center", "right"] = "center"
    origin: Optional[float] = None


@dataclass(frozen=True)
class BinnedFrame:
    """The result of :func:`bin_frame`.

    Attributes
    ----------
    frame : pandas.DataFrame
        MultiIndex columns ``(value_col, {"center","err_low","err_high"})``,
        one row per surviving bin (``min_count`` applied).  The row
        index carries the bin labels (raw values for identity binning;
        interval-derived labels named ``"{src} (binned)"`` for numeric
        binning — the legacy naming, kept deliberately).
    counts : pandas.Series
        Samples per surviving bin — a separate series, not a pseudo
        column (the old ``("count", "center")`` shape special-case
        lives only in the ``ScanData`` compatibility wrapper).
    """

    frame: "pd.DataFrame"
    counts: "pd.Series" = field(repr=False, default=None)  # type: ignore[assignment]


def compute_bin_key(
    frame: "pd.DataFrame", cfg: BinningConfig
) -> "tuple[pd.Series, str]":
    """Return ``(bin_labels, bin_name)`` per the config's binning mode.

    Identity binning returns the source column untouched under its own
    name; numeric binning (explicit edges / uniform width / quantiles)
    returns interval-derived labels under ``"{src} (binned)"``.  A
    non-numeric source always bins by identity.

    Parameters
    ----------
    frame : pandas.DataFrame
        The rows to bin.
    cfg : BinningConfig
        The binning configuration.

    Returns
    -------
    tuple of (pandas.Series, str)
        The per-row bin labels and the grouping column name.

    Raises
    ------
    KeyError
        When ``cfg.bin_col`` is absent.
    """
    import numpy as np
    import pandas as pd

    src = cfg.bin_col
    if src not in frame.columns:
        raise KeyError(f"Bin column {src!r} not found in DataFrame.")
    s = frame[src]

    if not pd.api.types.is_numeric_dtype(s):
        return s, src

    if cfg.bin_edges is not None:
        bins = pd.cut(s, bins=list(cfg.bin_edges), right=cfg.right)
    elif cfg.bin_width is not None:
        vmin = s.min() if cfg.origin is None else cfg.origin
        vmax = s.max()
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        n = int(np.ceil((vmax - vmin) / float(cfg.bin_width))) or 1
        edges = vmin + np.arange(n + 1, dtype=float) * float(cfg.bin_width)
        bins = pd.cut(s, bins=edges, right=cfg.right, include_lowest=True)
    elif cfg.quantile_bins is not None:
        bins = pd.qcut(s, q=max(1, int(cfg.quantile_bins)), duplicates="drop")
    else:
        return s, src

    if cfg.label == "interval":
        labels = bins.astype(str)
    else:
        left = bins.cat.categories.left.values
        right = bins.cat.categories.right.values
        if cfg.label == "left":
            label_vals = left
        elif cfg.label == "right":
            label_vals = right
        else:  # "center"
            label_vals = (left + right) / 2.0
        labels = bins.map(dict(zip(bins.cat.categories, label_vals)))
    return labels, f"{src} (binned)"


def _select_value_cols(frame: "pd.DataFrame", cfg: BinningConfig) -> "list[str]":
    import numpy as np

    if cfg.value_cols is not None:
        return [str(c) for c in cfg.value_cols]
    numeric = frame.select_dtypes(include=[np.number]).columns
    return [str(c) for c in numeric if str(c) != "Shotnumber"]


def bin_frame(frame: "pd.DataFrame", cfg: BinningConfig) -> BinnedFrame:
    """Aggregate *frame* into bins — pure, stateless, vectorized.

    Parameters
    ----------
    frame : pandas.DataFrame
        The per-shot rows (any provenance mix; typically the filtered
        union frame).
    cfg : BinningConfig
        The aggregation configuration.

    Returns
    -------
    BinnedFrame
        Centers + asymmetric error bands per value column, and the
        per-bin counts.

    Raises
    ------
    KeyError
        When ``cfg.bin_col`` is absent.
    ValueError
        On an unknown ``cfg.err``.
    """
    import numpy as np
    import pandas as pd

    value_cols = _select_value_cols(frame, cfg)

    # Row policy over the selected MEASUREMENT columns: the bin key is
    # excluded (a grouping key is never a measurement — including it
    # would make dropna="all" vacuous), and entirely-NaN columns never
    # cause drops (they aggregate to NaN instead) — both applied to BOTH
    # policies, unlike the legacy implementation.
    valid_cols = [
        c
        for c in value_cols
        if c in frame.columns and c != cfg.bin_col and frame[c].notna().any()
    ]
    bin_labels, bin_name = compute_bin_key(frame, cfg)
    work = frame[
        [c for c in dict.fromkeys([*value_cols, cfg.bin_col]) if c in frame.columns]
    ]
    work = work.assign(**{"__bin__": bin_labels})
    if valid_cols:
        if cfg.dropna == "any":
            work = work.dropna(subset=valid_cols, how="any")
        else:
            work = work.dropna(subset=valid_cols, how="all")

    if work.empty:
        import pandas as pd  # noqa: F811 — local alias for the early exit

        empty = pd.DataFrame()
        empty.index.name = bin_name
        counts = pd.Series([], dtype=int)
        counts.index.name = bin_name
        return BinnedFrame(frame=empty, counts=counts)

    g = work.groupby("__bin__", dropna=False, observed=True)
    center = g[value_cols].median() if cfg.agg == "median" else g[value_cols].mean()

    if cfg.err in ("std", "stderr"):
        err = g[value_cols].std(ddof=cfg.ddof)
        if cfg.err == "stderr":
            counts_for_err = g.size().astype(float)
            err = err.div(np.sqrt(counts_for_err), axis=0)
        err_low, err_high = err, err.copy()
    elif cfg.err in ("iqr", "percentile"):
        p_lo, p_hi = float(cfg.percentiles[0]), float(cfg.percentiles[1])
        qtbl = g[value_cols].quantile(q=[p_lo, p_hi])
        lo = qtbl.xs(p_lo, level=-1)
        hi = qtbl.xs(p_hi, level=-1)
        err_low = (center - lo).clip(lower=0)
        err_high = (hi - center).clip(lower=0)
    elif cfg.err == "mad":
        # Vectorized MAD: per-row deviation from the row's bin median,
        # then the per-bin median of those deviations (the legacy
        # implementation looped a Python apply per group).
        med_per_row = g[value_cols].transform("median")
        dev = (work[value_cols] - med_per_row).abs()
        mad = dev.groupby(work["__bin__"], dropna=False, observed=True).median()
        if cfg.scale_to_sigma:
            mad = mad * 1.4826
        err_low, err_high = mad, mad.copy()
    else:
        raise ValueError(f"Unknown err: {cfg.err}")

    pieces = {
        col: pd.concat(
            {"center": center[col], "err_low": err_low[col], "err_high": err_high[col]},
            axis=1,
        )
        for col in value_cols
    }
    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=center.index)

    counts = g.size()
    if cfg.min_count > 1:
        keep = counts[counts >= cfg.min_count].index
        out = out.loc[keep]
        counts = counts.loc[keep]

    out = out.sort_index(axis=1)
    out.index.name = bin_name
    counts.index.name = bin_name
    return BinnedFrame(frame=out, counts=counts)
