"""Analysis-tab endpoint helpers: param parsing, JSON shaping, snippets.

The middle of the three-layer contract
(``Planning/data_portal/03_analysis_tabs_design.md``): the ``/api``
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
from typing import Any, Optional

from geecs_data_utils.data.binning import BinningConfig
from geecs_data_utils.data.row_filters import RowFilters


class BadParam(ValueError):
    """A malformed query parameter — the app renders it as HTTP 400."""


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


def parse_bincfg(raw: str) -> BinningConfig:
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
_DISPLAY_NUMBERS = ("xmin", "xmax", "ymin", "ymax", "msize")
_DISPLAY_FIELDS = {*_DISPLAY_BOOLS, *_DISPLAY_NUMBERS, "colors", "layout"}


def parse_display(raw: str) -> dict:
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


def jsonable_labels(index: Any) -> list:
    """Bin labels (a frame index) as JSON-safe scalars.

    Numeric labels stay numbers (finite only), interval/string labels
    become strings, missing labels (the NA bin) become ``None``.
    """
    return jsonable_values(list(index))


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


def _snippet_display(display: Optional[dict]) -> str:
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
    hands back the figure the Plot tab renders (from the notebook frame
    it titles axes with raw column names — the page adds pretty names).
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
    return (
        "# reproduces this view exactly — the endpoint calls the same functions\n"
        + _snippet_prelude(uid, run_day)
        + _snippet_filters(filters)
        + conversion
        + f"series = frame[{columns!r}]\n"
        + "from geecs_portal.figures import shots_figure\n"
        + f"shots_figure(frame, y={columns!r}{x_arg}{_snippet_display(display)})"
        + "  # the figure the Plot tab renders\n"
    )


def binned_code(
    uid: str,
    run_day: Optional[str],
    columns: list[str],
    filters: RowFilters,
    cfg: BinningConfig,
    display: Optional[dict] = None,
) -> str:
    """The notebook snippet reproducing a ``/api/.../binned`` response."""
    non_default = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(BinningConfig)
        if getattr(cfg, f.name) != f.default
    }
    non_default["value_cols"] = columns
    kwargs = ", ".join(f"{k}={v!r}" for k, v in sorted(non_default.items()))
    return (
        "# reproduces this view exactly — the endpoint calls the same functions\n"
        + _snippet_prelude(uid, run_day)
        + _snippet_filters(filters)
        + "from geecs_data_utils.data.binning import BinningConfig, bin_frame\n"
        + "\n"
        + f"result = bin_frame(frame, BinningConfig({kwargs}))\n"
        + "result.frame  # (column, center/err_low/err_high); result.counts per bin\n"
        + "from geecs_portal.figures import binned_figure\n"
        + "series = {c: {s: result.frame[(c, s)].tolist()\n"
        + '               for s in ("center", "err_low", "err_high")}\n'
        + f"          for c in {columns!r} if (c, 'center') in result.frame}}\n"
        + f"binned_figure(result.frame.index.tolist(), series, y={columns!r}, "
        + f"bin_col={cfg.bin_col!r}{_snippet_display(display)})"
        + "  # the figure the Plot tab renders\n"
    )
