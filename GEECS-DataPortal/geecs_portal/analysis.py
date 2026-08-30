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
    if kind is int:
        if float(value) != int(value):
            raise BadParam(f"bad bincfg param: {key} must be an integer")
        return int(value)
    if not math.isfinite(float(value)):
        raise BadParam(f"bad bincfg param: {key} must be finite")
    return float(value)


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
    for key, allowed in _BIN_CHOICES.items():
        if key in payload and payload[key] not in allowed:
            raise BadParam(f"bad bincfg param: {key} must be one of {sorted(allowed)}")
    if "bin_col" in payload and not isinstance(payload["bin_col"], str):
        raise BadParam("bad bincfg param: bin_col must be a string")
    for key in _BIN_INTS:
        if payload.get(key) is not None:
            payload[key] = _bin_number(key, payload[key], int)
    for key in _BIN_FLOATS:
        if payload.get(key) is not None:
            payload[key] = _bin_number(key, payload[key], float)
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


def frame_code(
    uid: str, run_day: Optional[str], columns: list[str], filters: RowFilters
) -> str:
    """The notebook snippet reproducing a ``/api/.../frame`` response."""
    return (
        "# reproduces this view exactly — the endpoint calls the same functions\n"
        + _snippet_prelude(uid, run_day)
        + _snippet_filters(filters)
        + f"series = frame[{columns!r}]\n"
    )


def binned_code(
    uid: str,
    run_day: Optional[str],
    columns: list[str],
    filters: RowFilters,
    cfg: BinningConfig,
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
    )
