"""Analysis-tab endpoint helpers: param parsing, JSON shaping, snippets.

The middle of the three-layer contract: the ``/api``
endpoints are one-liners over the pure data-utils primitives
(:func:`~geecs_data_utils.scan_frame.scan_frame`,
:func:`~geecs_data_utils.data.row_filters.filter_mask`,
:func:`~geecs_data_utils.data.binning.bin_frame`), and this module owns
the boundary chores around those calls: deserializing the URL-carried
``filters``/``bincfg`` params (the same JSON the "show the code"
snippet quotes), making pandas values JSON-safe (NaN → ``null``), and
generating the notebook snippet that reproduces each response exactly —
the arc's reproducibility doctrine.

No FastAPI imports here: bad params raise :class:`BadParam` and the app
maps that to 400.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Optional, Sequence

from geecs_data_utils.data.binning import BinningConfig, bin_frame
from geecs_data_utils.data.row_filters import RowFilters
from geecs_data_utils.scan_grid import GridConfig, GridResult


class BadParam(ValueError):
    """A malformed query parameter — the app renders it as HTTP 400."""


def parse_gridcfg(raw: str) -> GridConfig:
    """Validate all URL-carried grid fields before numerical work."""
    try:
        return GridConfig.model_validate_json(raw) if raw else GridConfig()
    except ValueError as exc:
        raise BadParam(f"bad gridcfg param: {exc}") from exc


def grid_code(
    uid: str,
    run_day: Optional[str],
    cfg: GridConfig,
    filters: RowFilters,
    pretty: dict[str, str] | None = None,
) -> str:
    """Reproduce geometry, filtered statistics and both figures in a notebook."""
    return (
        _snippet_prelude(uid, run_day)
        + "from geecs_data_utils.scan_grid import GridConfig, grid_scan\n"
        + "from geecs_data_utils.data.row_filters import RowFilters\n"
        + "from geecs_portal.figures import grid_figures\n"
        + f"cfg = GridConfig.model_validate_json({cfg.model_dump_json()!r})\n"
        + f"filters = RowFilters.model_validate_json({filters.model_dump_json()!r})\n"
        + "result = grid_scan(pf.frame, detail.start_doc, cfg, filters)\n"
        + f"figures = grid_figures(result, pretty={pretty!r})\n"
        + "for figure in figures.values():\n    figure.show()\n"
    )


def grid_document(result: GridResult, pretty: dict[str, str], code: str) -> dict:
    """A 2-D map's JSON answer: cells, axes, counts, both figures and *code*.

    One shape for the scan page's ``/api/run/{uid}/grid`` and the Plot
    page's ``/plot/api/map``; the numbers are ``grid_scan``'s.

    Parameters
    ----------
    result : GridResult
        ``geecs_data_utils.scan_grid.grid_scan``'s answer.
    pretty : dict[str, str]
        Column → the name the figures title it with.
    code : str
        The notebook snippet that rebuilds the maps.

    Returns
    -------
    dict
        JSON-safe (NaN → ``None``).
    """
    from geecs_portal import figures

    payload = {
        "cells": result.cells.to_dict("records"),
        "config": result.config.model_dump(),
        "kind": result.kind,
        "x_values": result.x_values,
        "y_values": result.y_values,
        "visits": result.visits,
        "pass": result.passing,
        "total": result.total,
        "bin_column": result.bin_column,
        "notes": result.notes,
        "error_label": figures.grid_error_label(result),
        "pretty": pretty,
        "figures": {
            name: figures.page_figure(fig)
            for name, fig in figures.grid_figures(
                result, pretty=pretty, palette=figures.THEMED_PALETTE
            ).items()
        },
        "code": code,
    }
    return jsonable_document(payload)


def parse_filters(raw: str) -> RowFilters:
    """Deserialize the ``filters`` query param (empty → no filters).

    Parameters
    ----------
    raw : str
        The URL-carried ``RowFilters`` JSON (``model_dump_json`` form).

    Returns
    -------
    RowFilters
        The validated selection.

    Raises
    ------
    BadParam
        On JSON or model validation failure.
    """
    if not raw:
        return RowFilters()
    try:
        return RowFilters.model_validate_json(raw)
    except Exception as exc:  # pydantic ValidationError or bad JSON
        raise BadParam(f"bad filters param: {exc}") from exc


_BIN_FIELDS = {f.name for f in dataclasses.fields(BinningConfig)}
_BIN_CHOICES = {
    "agg": {"mean", "median"},
    "err": {"std", "stderr", "mad", "iqr", "percentile"},
    "dropna": {"any", "all"},
    "label": {"interval", "left", "center", "right"},
}
#: Per-field shape checks: BinningConfig is a plain (non-validating)
#: dataclass, so type/arity discipline lives HERE — a wrong-typed field
#: must 400 at the boundary, never 500 (or silently coerce) inside
#: bin_frame.
_BIN_INTS = ("ddof", "min_count", "quantile_bins")
_BIN_FLOATS = ("bin_width", "origin")
_BIN_BOOLS = ("right", "scale_to_sigma")


def _bin_number(key: str, value: object, kind: type) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadParam(f"bad bincfg param: {key} must be a number")
    # json.loads admits NaN/Infinity literals and unbounded ints — all
    # three must 400 here, not overflow later.
    try:
        as_float = float(value)
    except OverflowError as exc:
        raise BadParam(f"bad bincfg param: {key} must be finite") from exc
    if not math.isfinite(as_float):
        raise BadParam(f"bad bincfg param: {key} must be finite")
    if kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise BadParam(f"bad bincfg param: {key} must be an integer")
        return int(value)
    return as_float


def parse_bincfg(raw: str) -> BinningConfig:  # noqa: C901, PLR0912
    """Deserialize the ``bincfg`` query param (empty → defaults).

    ``BinningConfig`` is a plain dataclass, so validation is explicit
    here — unknown keys, out-of-vocabulary choices, and wrong
    types/arities are all refused (a typo'd ``err`` or a one-element
    ``percentiles`` must 400 at the boundary, not 500 inside
    ``bin_frame``); JSON lists become the tuples the config declares.

    Parameters
    ----------
    raw : str
        The URL-carried JSON object of ``BinningConfig`` fields.

    Returns
    -------
    BinningConfig
        The frozen config.

    Raises
    ------
    BadParam
        On malformed JSON, unknown keys, or invalid field values.
    """
    if not raw:
        return BinningConfig()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BadParam(f"bad bincfg param: {exc}") from exc
    if not isinstance(payload, dict):
        raise BadParam("bad bincfg param: expected a JSON object")
    unknown = set(payload) - _BIN_FIELDS
    if unknown:
        raise BadParam(f"bad bincfg param: unknown fields {sorted(unknown)}")
    # Doctrine: an explicit JSON null means "field absent" — the
    # dataclass default stands.  (Nulls into non-Optional fields would
    # otherwise bypass every check below and crash inside bin_frame.)
    payload = {key: value for key, value in payload.items() if value is not None}
    for key, allowed in _BIN_CHOICES.items():
        # isinstance first: an unhashable value (list/dict) would raise
        # a raw TypeError out of the set-membership test.
        if key in payload and (
            not isinstance(payload[key], str) or payload[key] not in allowed
        ):
            raise BadParam(f"bad bincfg param: {key} must be one of {sorted(allowed)}")
    if "bin_col" in payload and not isinstance(payload["bin_col"], str):
        raise BadParam("bad bincfg param: bin_col must be a string")
    for key in _BIN_INTS:
        if payload.get(key) is not None:
            payload[key] = _bin_number(key, payload[key], int)
    for key in _BIN_FLOATS:
        if payload.get(key) is not None:
            payload[key] = _bin_number(key, payload[key], float)
    if payload.get("bin_width") is not None and payload["bin_width"] <= 0:
        # 0 divides to inf inside compute_bin_key (OverflowError → 500);
        # negative widths build empty edge arrays.
        raise BadParam("bad bincfg param: bin_width must be > 0")
    for key in _BIN_BOOLS:
        if key in payload and not isinstance(payload[key], bool):
            raise BadParam(f"bad bincfg param: {key} must be a boolean")
    if payload.get("percentiles") is not None:
        pair = payload["percentiles"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise BadParam("bad bincfg param: percentiles must be a [low, high] pair")
        payload["percentiles"] = tuple(
            _bin_number("percentiles", bound, float) for bound in pair
        )
    if payload.get("bin_edges") is not None:
        edges = payload["bin_edges"]
        if not isinstance(edges, list) or len(edges) < 2:
            raise BadParam("bad bincfg param: bin_edges must be a list of >= 2 edges")
        payload["bin_edges"] = tuple(
            _bin_number("bin_edges", edge, float) for edge in edges
        )
    if payload.get("value_cols") is not None:
        cols = payload["value_cols"]
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise BadParam("bad bincfg param: value_cols must be a list of strings")
        payload["value_cols"] = tuple(cols)
    try:
        return BinningConfig(**payload)
    except (TypeError, ValueError) as exc:
        raise BadParam(f"bad bincfg param: {exc}") from exc


_DISPLAY_BOOLS = ("logx", "logy")
# plo/phi = the image endpoints' percentile-window overrides (the plot
# figures ignore them, same one-display-state doctrine as cmap).
_DISPLAY_NUMBERS = (
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "msize",
    "width",
    "height",
    "plo",
    "phi",
)
#: Image view modes: "" (the processed / raw pixels as a windowed PNG)
#: or "rendered" (the analyzer's own matplotlib figure — overlays,
#: axes, colorbar; only meaningful with a `processing` selection).
DISPLAY_MODES = ("", "rendered")
_DISPLAY_FIELDS = {
    *_DISPLAY_BOOLS,
    *_DISPLAY_NUMBERS,
    "colors",
    "layout",
    "cmap",
    "mode",
}


def parse_display(raw: str) -> dict:  # noqa: C901, PLR0912
    """Deserialize the ``display`` query param (empty → no cosmetics).

    Types are the contract and 400 here (the ``bincfg`` precedent: a
    wrong-typed field must never 500 inside the figure builder); values
    keep the page's historical degrade semantics downstream — a non-hex
    color entry or a non-positive marker size falls back to the default
    in :mod:`geecs_portal.figures`, because display state rides shared
    links and a cosmetic value should never make a link fail.  The
    ``layout`` key is type-checked but otherwise opaque: it is the
    client-side Plotly-layout passthrough and is never applied on the
    server.

    Parameters
    ----------
    raw : str
        The URL-carried JSON object of plot-cosmetics fields.

    Returns
    -------
    dict
        The validated mapping (nulls dropped — field absent).

    Raises
    ------
    BadParam
        On malformed JSON, unknown keys, or wrong-typed fields.
    """
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BadParam(f"bad display param: {exc}") from exc
    if not isinstance(payload, dict):
        raise BadParam("bad display param: expected a JSON object")
    unknown = set(payload) - _DISPLAY_FIELDS
    if unknown:
        raise BadParam(f"bad display param: unknown fields {sorted(unknown)}")
    payload = {key: value for key, value in payload.items() if value is not None}
    for key in _DISPLAY_BOOLS:
        if key in payload and not isinstance(payload[key], bool):
            raise BadParam(f"bad display param: {key} must be a boolean")
    for key in _DISPLAY_NUMBERS:
        if key in payload:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BadParam(f"bad display param: {key} must be a number")
            # json.loads admits NaN/Infinity literals — degrade-vs-400
            # doesn't apply to non-finite numbers: they are type junk.
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise BadParam(f"bad display param: {key} must be finite")
    if "colors" in payload:
        colors = payload["colors"]
        if not isinstance(colors, list) or not all(isinstance(c, str) for c in colors):
            raise BadParam("bad display param: colors must be a list of strings")
    if "layout" in payload and not isinstance(payload["layout"], dict):
        raise BadParam("bad display param: layout must be an object")
    if "cmap" in payload and not isinstance(payload["cmap"], str):
        raise BadParam("bad display param: cmap must be a string")
    if "mode" in payload and payload["mode"] not in DISPLAY_MODES:
        raise BadParam(f"bad display param: mode must be one of {list(DISPLAY_MODES)}")
    return payload


def jsonable_values(series: Any) -> list:
    """A pandas Series as a JSON-safe list (NA/NaN/±inf/NaT → ``None``).

    ``json.dumps`` emits invalid JSON for float NaN and browsers refuse
    it — every array leaving an ``/api`` endpoint passes through here.
    ``pd.NA`` (an Int64 shot key on a union row the event side missed)
    must become ``None``, never the string ``"<NA>"``.
    """
    import pandas as pd

    out = []
    for value in series:
        if value is None:
            out.append(None)
            continue
        try:
            if pd.isna(value):
                out.append(None)
                continue
        except (TypeError, ValueError):
            pass  # array-likes: fall through to the float attempt
        try:
            number = float(value)
        except (TypeError, ValueError):
            out.append(str(value))
            continue
        out.append(number if math.isfinite(number) else None)
    return out


def jsonable_datetimes(series: Any, epoch: str) -> list:
    """Timestamp seconds → host-local ISO strings (NA → ``None``).

    ``epoch`` is :func:`geecs_data_utils.tiled_schema.timestamp_epoch`'s
    verdict: ``"labview"`` values are shifted by the wire offset first,
    ``"unix"`` values are used as-is.  Strings are naive local time
    (the service host's zone — Pacific in the lab), which Plotly's date
    axis renders verbatim — raw epoch seconds never reach the user.
    """
    from datetime import datetime

    from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

    offset = LABVIEW_EPOCH_OFFSET if epoch == "labview" else 0.0
    out = []
    for value in jsonable_values(series):
        if not isinstance(value, float):
            out.append(None)
            continue
        try:
            stamp = datetime.fromtimestamp(value - offset)
        except (OverflowError, OSError, ValueError):
            out.append(None)
            continue
        out.append(stamp.isoformat(sep=" ", timespec="milliseconds"))
    return out


def binned_series(
    frame: Any,  # pandas.DataFrame: the kept shots
    cfg: BinningConfig,
    columns: Sequence[str],
    x: Optional[str] = None,
) -> tuple[Any, dict, Optional[list]]:
    """Bin *columns* of *frame*: the binned views' one call into ``bin_frame``.

    Bins group the shots (``cfg.bin_col``); with an *x*, each bin is
    placed at that column's per-bin mean, reindexed onto the Y bins (X's
    own dropna may keep other bins, and a point must never plot at another
    bin's position; a bin X lacks is a null, which Plotly skips).

    Returns
    -------
    tuple
        ``(result, series, x_centers)``: ``bin_frame``'s result, column →
        ``{"center", "err_low", "err_high"}`` as JSON values (columns
        ``bin_frame`` dropped are left out), and the X positions or ``None``.

    Raises
    ------
    KeyError
        The bin column is not in *frame*.
    TypeError, ValueError
        A column ``bin_frame`` cannot aggregate, or degenerate percentiles.
    """
    result = bin_frame(frame, dataclasses.replace(cfg, value_cols=tuple(columns)))
    x_centers = None
    if x:
        x_result = bin_frame(
            frame, dataclasses.replace(cfg, value_cols=(x,), agg="mean")
        )
        if (x, "center") in x_result.frame.columns:
            x_centers = jsonable_values(
                x_result.frame[(x, "center")].reindex(result.frame.index)
            )
    series = {
        column: {
            part: jsonable_values(result.frame[(column, part)])
            for part in ("center", "err_low", "err_high")
        }
        for column in columns
        if (column, "center") in result.frame.columns
    }
    return result, series, x_centers


def jsonable_labels(index: Any) -> list:
    """Bin labels (a frame index) as JSON-safe scalars.

    Numeric labels stay numbers (finite only), interval/string labels
    become strings, missing labels (the NA bin) become ``None``.
    """
    return jsonable_values(list(index))


def jsonable_document(value: Any) -> Any:
    """A run document (or any nested value) made JSON-safe.

    The browsing API serves the start/stop documents verbatim.  Tiled
    hands them back as plain containers, but a key may still carry a
    NaN, a numpy scalar, a date or a set — none of which ``json.dumps``
    survives.  Dicts and lists recurse; ``str``/``bool``/``int``/``None``
    pass through; floats follow the NaN/inf → ``None`` rule; anything
    else goes through :func:`jsonable_values` (numeric → float, else its
    ``str``).  A document key must never 500 a run's detail.
    """
    if isinstance(value, dict):
        return {str(key): jsonable_document(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_document(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return jsonable_values([value])[0]


# --------------------------- show the code ----------------------------


def _snippet_prelude(uid: str, run_day: Optional[str]) -> str:
    day_arg = f"date.fromisoformat({run_day!r})" if run_day else "None"
    return (
        "from datetime import date\n"
        "from geecs_data_utils.tiled_catalog import TiledScanCatalog, "
        "resolve_scan_folder\n"
        "from geecs_data_utils.scan_frame import scan_frame\n"
        "\n"
        "catalog = TiledScanCatalog.from_config()\n"
        f"detail = catalog.load_run({uid!r})\n"
        f"folder = resolve_scan_folder(detail, {day_arg})\n"
        "pf = scan_frame(detail, folder)  # union frame + provenance\n"
    )


def _snippet_filters(filters: RowFilters) -> str:
    if not filters.active_groups():
        return "frame = pf.frame\n"
    blob = filters.model_dump_json()
    return (
        "from geecs_data_utils.data.row_filters import RowFilters, apply_filters\n"
        "\n"
        f"filters = RowFilters.model_validate_json({blob!r})\n"
        "frame = apply_filters(pf.frame, filters)\n"
    )


def snippet_display(display: Optional[dict]) -> str:
    """Render the snippet's ``display=`` argument text.

    Empty when no cosmetics are set; the ``layout`` key is client-side
    and never reaches figures.
    """
    cosmetics = {k: v for k, v in (display or {}).items() if k != "layout"}
    return f", display={cosmetics!r}" if cosmetics else ""


def frame_code(
    uid: str,
    run_day: Optional[str],
    columns: list[str],
    filters: RowFilters,
    datetime_columns: Optional[dict] = None,
    x: Optional[str] = None,
    display: Optional[dict] = None,
) -> str:
    """The notebook snippet reproducing a ``/api/.../frame`` response.

    ``datetime_columns`` maps column name → epoch verdict for the
    columns the endpoint served as local datetimes — the snippet must
    perform the same conversion or it would hand back raw seconds while
    claiming to reproduce the view.  The closing ``shots_figure`` call
    hands back the payload's figure (from the notebook frame
    it titles axes with raw column names, where a page shows pretty names).
    """
    converted = datetime_columns or {}
    conversion = ""
    if converted:
        conversion = (
            "# the view renders timestamps as local datetimes:\n"
            "from datetime import datetime\n"
        )
        if "labview" in converted.values():
            conversion += (
                "from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET\n"
            )
        for column, epoch in converted.items():
            shift = " - LABVIEW_EPOCH_OFFSET" if epoch == "labview" else ""
            conversion += (
                f"frame[{column!r}] = (frame[{column!r}]{shift})"
                ".map(datetime.fromtimestamp)\n"
            )
    x_arg = f", x={x!r}" if x else ""
    # The kinds map keeps date-axis semantics in the notebook figure
    # (log/range guards on datetime axes — the page's rule).
    kinds = {column: "datetime" for column in converted}
    kinds_arg = f", kinds={kinds!r}" if kinds else ""
    return (
        "# reproduces this view exactly — the endpoint calls the same functions\n"
        + _snippet_prelude(uid, run_day)
        + _snippet_filters(filters)
        + conversion
        + f"series = frame[{columns!r}]\n"
        + "from geecs_portal.figures import shots_figure"
        + "  # pip install geecs-data-portal\n"
        + f"shots_figure(frame, y={columns!r}{x_arg}{kinds_arg}"
        + f"{snippet_display(display)})"
        + "  # the payload's figure\n"
    )


def bin_images_code(
    uid: str,
    run_day: Optional[str],
    device: str,
    filters: RowFilters,
    cfg: BinningConfig,
) -> str:
    """The notebook snippet reproducing a ``/api/.../bin-images`` response.

    Reproduces the endpoint's NUMBERS exactly (bins, counts, member
    shots — the same ``bin_rows`` call with the filter as its mask) and
    the unprocessed pixels (the same ``average_rows``).
    """
    non_default = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(BinningConfig)
        if getattr(cfg, f.name) != f.default
    }
    kwargs = ", ".join(f"{k}={v!r}" for k, v in sorted(non_default.items()))
    if filters.active_groups():
        masking = (
            "from geecs_data_utils.data.row_filters import RowFilters, filter_mask\n"
            + "\n"
            + f"filters = RowFilters.model_validate_json({filters.model_dump_json()!r})\n"
            + "mask = filter_mask(pf.frame, filters)\n"
        )
    else:
        masking = "mask = None\n"
    return (
        "# reproduces this listing exactly — the endpoint calls the same functions\n"
        + _snippet_prelude(uid, run_day)
        + masking
        + "from geecs_data_utils.data.binning import BinningConfig, bin_rows\n"
        + "from geecs_data_utils.io.shot_images import average_rows\n"
        + "from geecs_data_utils.tiled_schema import shot_axis_for_frame\n"
        + "\n"
        + f"cfg = BinningConfig({kwargs})\n"
        + "shots = shot_axis_for_frame(pf.frame)  # union row -> shot number\n"
        + "members = [\n"
        + "    (label, sorted({int(s) for s in shots.iloc[[r - 1 for r in rows]].dropna()}))\n"
        + "    for label, rows in bin_rows(pf.frame, cfg, mask=mask)\n"
        + "]\n"
        + "members  # per bin: (label, shot list); counts = len(shot list)\n"
        + "\n"
        + "# one grid image (the first bin's), before display windowing:\n"
        + "pixels = (\n"
        + f"    average_rows(folder, detail.data, {device!r}, members[0][1]).array\n"
        + "    if members\n"
        + "    else None\n"
        + ")\n"
    )


def binned_code(
    uid: str,
    run_day: Optional[str],
    columns: list[str],
    filters: RowFilters,
    cfg: BinningConfig,
    x: Optional[str] = None,
    display: Optional[dict] = None,
) -> str:
    """The notebook snippet reproducing a ``/api/.../binned`` response.

    With an ``x``, the snippet also reproduces the per-bin mean X
    positions the figure plots against (bins group, X places).
    """
    non_default = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(BinningConfig)
        if getattr(cfg, f.name) != f.default
    }
    non_default["value_cols"] = columns
    kwargs = ", ".join(f"{k}={v!r}" for k, v in sorted(non_default.items()))
    x_lines = ""
    x_args = "result.frame.index.tolist(), series"
    if x:
        x_lines = (
            "from dataclasses import replace\n"
            "\n"
            "# bins GROUP the data; the X parameter PLACES it (per-bin mean),\n"
            "# reindexed onto the y bins so diverging dropna cannot shift points\n"
            f"x_result = bin_frame(frame, replace(cfg, value_cols=({x!r},), "
            "agg='mean'))\n"
            f"x_centers = x_result.frame[({x!r}, 'center')]"
            ".reindex(result.frame.index)\n"
        )
        x_args += f",\n              x_values=x_centers.tolist(), x_label={x!r}"
    return (
        "# reproduces this view exactly — the endpoint calls the same functions\n"
        + _snippet_prelude(uid, run_day)
        + _snippet_filters(filters)
        + "from geecs_data_utils.data.binning import BinningConfig, bin_frame\n"
        + "\n"
        + f"cfg = BinningConfig({kwargs})\n"
        + "result = bin_frame(frame, cfg)\n"
        + "result.frame  # (column, center/err_low/err_high); result.counts per bin\n"
        + x_lines
        + "from geecs_portal.figures import binned_figure"
        + "  # pip install geecs-data-portal\n"
        + "series = {c: {s: result.frame[(c, s)].tolist()\n"
        + '               for s in ("center", "err_low", "err_high")}\n'
        + f"          for c in {columns!r} if (c, 'center') in result.frame}}\n"
        + f"binned_figure({x_args}, y={columns!r}, "
        + f"bin_col={cfg.bin_col!r}{snippet_display(display)})"
        + "  # the payload's figure\n"
    )
