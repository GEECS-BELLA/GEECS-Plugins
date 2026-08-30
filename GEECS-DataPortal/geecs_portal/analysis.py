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


def parse_bincfg(raw: str) -> BinningConfig:
    """Deserialize the ``bincfg`` query param (empty → defaults).

    ``BinningConfig`` is a plain dataclass, so validation is explicit
    here: unknown keys and out-of-vocabulary choice fields are refused
    (a typo'd ``err`` must 400 at the boundary, not 500 inside
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
        On malformed JSON, unknown keys, or invalid choice values.
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
    for key in ("percentiles", "value_cols", "bin_edges"):
        if isinstance(payload.get(key), list):
            payload[key] = tuple(payload[key])
    try:
        return BinningConfig(**payload)
    except (TypeError, ValueError) as exc:
        raise BadParam(f"bad bincfg param: {exc}") from exc


def jsonable_values(series: Any) -> list:
    """A pandas Series as a JSON-safe list (NaN/±inf/NaT → ``None``).

    ``json.dumps`` emits invalid JSON for float NaN and browsers refuse
    it — every array leaving an ``/api`` endpoint passes through here.
    """
    out = []
    for value in series:
        if value is None:
            out.append(None)
            continue
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
