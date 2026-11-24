"""Named settings sets: one JSON file per set, in one folder.

The folder is the process's own state directory, never the data share and
never the checkout: ``--settings-dir`` (the service unit passes its
systemd ``StateDirectory``), else the user's state folder
(:func:`user_state_dir`) for a portal started by hand. It is chosen once,
at start.

A set is a ``geecs_schemas.plot_settings.PlotterSettings`` document,
``<name>.json``: readable, copyable, and versioned so an older file loads
with new fields at their defaults. The model is GEECS-Schemas' (the
``plot_settings`` kind); this module keeps the store and the page's form
rules over it.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from geecs_analysis.image_prep import HIT_FILTERS
from geecs_schemas.plot_settings import (
    MAX_CAMERAS,
    MAX_CHANNEL,
    MAX_FIT_ORDER,
    MAX_IMAGE_COLUMNS,
    MAX_SCALARS,
    MAX_Y_LISTS,
    TILE_MAX_RANGE,
    ImageCamera,
    PlotterSettings,
    ScanRef,
)
from pydantic import ValidationError


#: The most scans the overlay draws (the main one with its extras): one
#: colour each, ``--trace-1..8``. The merged view takes any number.
MAX_OVERLAY_SCANS = 8

#: The Images page's scan-scalar statistic, one for every picked scalar.
SCALAR_STATS: tuple[str, ...] = ("mean_std", "median_iqr", "mean_stderr")

#: The choices of the tab and image controls, as stored.
CHOICES: dict[str, tuple[str, ...]] = {
    "tab": ("xy", "map", "images"),
    "image_strip": ("bins", "shots", "pick"),
    "image_pick": ("first", "ymax"),
    "image_hits": HIT_FILTERS,
    "image_layout": ("grid", "row"),
    "image_zscale": ("tile", "shared", "fixed"),
    "image_scalar_stat": SCALAR_STATS,
    "xy_view": ("shot", "bin"),
    "scan_view": ("overlay", "merged"),
}

#: A fit name as stored: GEECSplotter's Fit menu (``figures.FITS``) or one
#: a newer copy offers, kept rather than refused (the model keeps unknowns).
_FIT_NAME = re.compile(r"[a-z0-9_]{1,40}")

#: The set a page opens when the URL names none.
DEFAULT_SET = "default"

#: A letter or digit first, then letters, digits, spaces, ``-``, ``_`` and
#: ``.``; never ending in a space or a dot (Windows drops both from names).
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9 ._-]{0,62}[a-z0-9_-])?")

#: A shared set's name, as a link and the page name it: ``shared/<stem>``,
#: the stem its file's in the configs repo (``plot_settings/<stem>.yaml``).
#: A viewer's own set and the sets kept on this host never carry it, so a
#: name means one store, never both.
SHARED_PREFIX = "shared/"

#: A shared set's stem: a set name without spaces (the configurator names
#: files from letters, digits, ``.``, ``_`` and ``-`` alone).
SHARED_STEM = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9_-])?")


class SettingsError(ValueError):
    """A set name that is not allowed, or a set file that cannot be read."""


class NameRefused(SettingsError):
    """A Save-as name that cannot be used: the name is the thing to fix."""


def set_name(name: str) -> str:
    """The name as stored (lower case), or :class:`SettingsError`.

    Names compare case-insensitively, as Windows and macOS paths do, and
    runs of spaces count as one.
    """
    folded = " ".join(name.split()).lower()
    if folded.startswith(SHARED_PREFIX):
        return SHARED_PREFIX + shared_stem(folded[len(SHARED_PREFIX) :])
    if not _NAME.fullmatch(folded):
        raise SettingsError(
            f"{name!r} is not a settings name: use letters, digits, spaces, "
            "'-', '_' and '.', starting with a letter or digit"
        )
    return folded


def is_shared(name: str) -> bool:
    """Whether *name* (as :func:`set_name` returns it) is a shared set's."""
    return name.startswith(SHARED_PREFIX)


def shared_stem(name: str) -> str:
    """The configs repo's file stem for a Save-as name, or :class:`NameRefused`.

    The name folds as :func:`set_name` folds it, ``shared/`` dropped if
    typed, and its spaces become ``-``: "1BL 2BL timing" is
    ``1bl-2bl-timing``.
    """
    folded = " ".join(name.split()).lower()
    folded = folded.removeprefix(SHARED_PREFIX).strip().replace(" ", "-")
    if not SHARED_STEM.fullmatch(folded):
        raise NameRefused(
            f"{name!r} is not a settings name: use letters, digits, spaces, "
            "'-', '_' and '.', starting with a letter or digit"
        )
    return folded


@dataclass(frozen=True)
class SettingsStore:
    """The folder of named sets.

    Parameters
    ----------
    folder : Path
        Where the ``<name>.json`` files live; created on the first save.
    """

    folder: Path

    def path(self, name: str) -> Path:
        """The file a set is stored in; a shared set's name is refused (not this store's)."""
        stored = set_name(name)
        if is_shared(stored):
            raise SettingsError(
                f"{stored!r} is a shared set: it is kept in the configs repo, "
                "and changes only by Save as"
            )
        return self.folder / f"{stored}.json"

    def names(self) -> list[str]:
        """Every saved set, with the default first whether saved or not."""
        saved = (
            {p.stem for p in self.folder.glob("*.json") if _NAME.fullmatch(p.stem)}
            if self.folder.is_dir()
            else set()
        )
        return [DEFAULT_SET, *sorted(saved - {DEFAULT_SET})]

    def exists(self, name: str) -> bool:
        """Whether *name* has been saved."""
        return self.path(name).is_file()

    def load(self, name: str) -> PlotterSettings:
        """The set, or the defaults for a name never saved."""
        path = self.path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return PlotterSettings()
        except OSError as exc:
            raise SettingsError(f"cannot read {path}: {exc}") from exc
        try:
            return PlotterSettings.model_validate(json.loads(text))
        except (ValueError, ValidationError) as exc:
            raise SettingsError(
                f"{path} is not a plotter settings file: {exc}"
            ) from exc

    def save(self, name: str, settings: PlotterSettings) -> Path:
        """Write the set whole (``_write_whole``); the first save makes the folder."""
        path = self.path(name)
        text = settings.model_dump_json(indent=2) + "\n"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            _write_whole(path, text)
        except OSError as exc:
            raise SettingsError(f"cannot save {path}: {exc}") from exc
        return path


def _write_whole(path: Path, text: str) -> None:
    """Write *path* through a uniquely named sibling and a rename.

    The shape of ``geecs_logbook._fs.replace_with``: a unique temp name per
    call, so two saves of one set never clobber each other's half file, and
    a reader sees the old set or the new one, never part of one.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def user_state_dir(*, windows: bool = os.name == "nt") -> Path:
    """The settings folder of a portal started by hand, with no ``--settings-dir``.

    The user's own state folder: ``$XDG_STATE_HOME`` (else
    ``~/.local/state``), on Windows ``%LOCALAPPDATA%``, then
    ``geecs-data-portal/settings``.
    """
    if windows:
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "geecs-data-portal" / "settings"


def resolve_store(settings_dir: Optional[Path] = None) -> SettingsStore:
    """The store for this process: ``--settings-dir``, else the user's state folder."""
    if settings_dir is not None:
        return SettingsStore(Path(settings_dir).expanduser())
    return SettingsStore(user_state_dir())


#: The page form's field names, mapped to the settings they write.
FORM_FIELDS = {
    "x": "x",
    "y": "y",
    "xf": "x_filter",
    "yf": "y_filter",
    "xalias": "x_alias",
    "yalias": "y_alias",
    "xauto": "x_auto",
    "y2": "y2",
    "y3": "y3",
    "y2f": "y2_filter",
    "y3f": "y3_filter",
    "y2alias": "y2_alias",
    "y3alias": "y3_alias",
    "ycount": "y_count",
    "yfit": "y_fit",
    "y2fit": "y2_fit",
    "y3fit": "y3_fit",
    "yorder": "y_fit_order",
    "y2order": "y2_fit_order",
    "y3order": "y3_fit_order",
    "day": "day",
    "scan": "scan",
    "tab": "tab",
    "device": "image_device",
    "devf": "image_device_filter",
    "strip": "image_strip",
    "pick": "image_pick",
    "hits": "image_hits",
    "centre": "image_centre",
    "layout": "image_layout",
    "cols": "image_columns",
    "cmap": "image_cmap",
    "zscale": "image_zscale",
    "zmin": "image_zmin",
    "zmax": "image_zmax",
    "cbar": "image_cbar",
    "axes": "image_axes",
    "bg": "image_bg",
    "bgsub": "image_bgsub",
    "channel": "image_channels",
    "xmin": "image_xmin",
    "xmax": "image_xmax",
    "scalstat": "image_scalar_stat",
    "tilemax": "image_tile_max",
    "mtol": "merge_tolerance",
    "scalf": "image_scalar_filter",
    "view": "xy_view",
    "filters": "shot_filters",
    "bincfg": "bin_config",
    "display": "display",
    "mapv": "map_value",
    "mapx": "map_x",
    "mapy": "map_y",
    "mapf": "map_filter",
    "mapcfg": "map_config",
    "scans": "extra_scans",
    "multi": "scan_view",
    "oy": "overlay_y",
}

#: The JSON-text fields and the portal parser each must pass.
_JSON_FIELDS = {
    "shot_filters": "parse_filters",
    "bin_config": "parse_bincfg",
    "display": "parse_display",
    "map_config": "parse_gridcfg",
}

_BOOLEANS = (
    "x_alias",
    "y_alias",
    "x_auto",
    "y2_alias",
    "y3_alias",
    "image_cbar",
    "image_axes",
    "image_bgsub",
)

#: The whole-number fields and their bounds (``None``: no upper bound).
_INTEGERS: dict[str, tuple[int, Optional[int]]] = {
    "scan": (1, None),
    "y_count": (1, MAX_Y_LISTS),
    "y_fit_order": (0, MAX_FIT_ORDER),
    "y2_fit_order": (0, MAX_FIT_ORDER),
    "y3_fit_order": (0, MAX_FIT_ORDER),
    "image_columns": (1, MAX_IMAGE_COLUMNS),
    "overlay_y": (1, MAX_Y_LISTS),
}

#: The fixed z-scale values: any finite number, or empty for none.
_LEVELS = ("image_zmin", "image_zmax", "image_xmin", "image_xmax")

#: A colour-map name as stored (matplotlib's names are of this shape).
_CMAP_NAME = re.compile(r"[A-Za-z0-9_]{1,40}")


def merged(settings: PlotterSettings, form: dict[str, str]) -> PlotterSettings:
    """*settings* with every page field *form* carries; the rest kept.

    A page with no scan open has no pick lists, so a field it does not
    send keeps its stored value rather than resetting.
    """
    changes = {
        key: _form_value(field, key, form[field])
        for field, key in FORM_FIELDS.items()
        if field in form and not (key in ("day", "scan") and not form[field])
    }
    rois = dict(settings.image_rois)
    if "roi" in form:
        _set_roi(
            rois, changes.get("image_device") or settings.image_device, form["roi"]
        )
    scalars = _scalars(settings.image_scalars, form)
    if scalars != settings.image_scalars:
        changes["image_scalars"] = scalars
    more = _more_cameras(settings.image_more, form, rois)
    if more != settings.image_more:
        changes["image_more"] = more
    if rois != settings.image_rois:
        changes["image_rois"] = rois
    return settings.model_copy(update=changes)


#: A stacked camera's field: the first camera's form name, ``@`` and its
#: place (2 for the second): ``cmap@2``, ``roi@3``.
_CAMERA_FIELD = re.compile(
    rf"(device|devf|cmap|zscale|zmin|zmax|hits|centre|cbar|axes|bgsub|channel|xmin|xmax|roi)@([2-{MAX_CAMERAS}])"
)


def camera_field(name: str, place: int) -> str:
    """The form name of camera *place*'s *name* (the first camera's own)."""
    return name if place == 1 else f"{name}@{place}"


def _set_roi(rois: dict[str, list[int]], device: Optional[str], text: str) -> None:
    """Save *text* (``roi_value``) as *device*'s region; ``""`` clears it."""
    if not device:
        return
    roi = roi_value(text)
    if roi is None:
        rois.pop(device, None)
    else:
        rois[device] = roi


def _more_cameras(
    cameras: list[ImageCamera], form: dict[str, str], rois: dict[str, list[int]]
) -> list[ImageCamera]:
    """The stacked cameras after *form*: its fields, then ``addcam``/``dropcam``.

    Each field is checked as the first camera's is (``cmap@2`` as
    ``cmap``); one naming a camera not there is ignored (a page from
    before a drop). ``addcam`` stacks a camera under the rest; ``dropcam``
    (its place) takes one away.
    """
    more = list(cameras)
    # Regions last: a region is its camera's, whose device may change here.
    for name, value in sorted(
        form.items(), key=lambda item: item[0].startswith("roi@")
    ):
        match = _CAMERA_FIELD.fullmatch(name)
        if not match or int(match.group(2)) - 2 >= len(more):
            continue
        field, index = match.group(1), int(match.group(2)) - 2
        if field == "roi":
            _set_roi(rois, more[index].device, value)
            continue
        key = FORM_FIELDS[field]
        update = {key.removeprefix("image_"): _form_value(name, key, value)}
        more[index] = more[index].model_copy(update=update)
    if form.get("addcam") and len(more) < MAX_CAMERAS - 1:
        more.append(ImageCamera())
    drop = form.get("dropcam", "")
    if drop.isdigit() and 0 <= int(drop) - 2 < len(more):
        del more[int(drop) - 2]
    return more


def _scalars(scalars: list[str], form: dict[str, str]) -> list[str]:
    """The scan scalars after *form*'s ``scaladd`` (one column) and ``scaldrop``."""
    out = list(scalars)
    drop = form.get("scaldrop", "")
    if drop in out:
        out.remove(drop)
    add = form.get("scaladd", "").strip()
    if add and add not in out:
        if len(out) >= MAX_SCALARS:
            raise SettingsError(f"at most {MAX_SCALARS} scan scalars")
        out.append(add)
    return out


def scan_refs(text: str) -> list[ScanRef]:
    """``"2026-10-26:6,2026-10-27:3"`` as scans, in order, each once.

    The form's (and the figure route's) spelling of ``extra_scans``;
    ``""`` is none.

    Raises
    ------
    SettingsError
        An entry that is not ``YYYY-MM-DD:N`` with N a scan number.
    """
    refs: list[ScanRef] = []
    for entry in filter(None, (e.strip() for e in text.split(","))):
        day, _, number = entry.partition(":")
        try:
            ref = ScanRef(day=date.fromisoformat(day), scan=int(number))
        except (ValueError, ValidationError):
            raise SettingsError(
                f"{entry!r} is not a day and a scan (YYYY-MM-DD:N)"
            ) from None
        if ref not in refs:
            refs.append(ref)
    return refs


def scan_refs_text(refs: list[ScanRef]) -> str:
    """The form's spelling of *refs* (:func:`scan_refs` reads it back)."""
    return ",".join(f"{r.day.isoformat()}:{r.scan}" for r in refs)


def roi_value(text: str) -> Optional[list[int]]:
    """``"x0,y0,x1,y1"`` as pixels, ``None`` for ``""`` (the whole image).

    Raises
    ------
    SettingsError
        Not four whole numbers, a negative one, or an empty region.
    """
    if not text.strip():
        return None
    try:
        x0, y0, x1, y1 = (int(v) for v in text.split(","))
    except ValueError:
        raise SettingsError(f"roi={text!r}: must be four whole numbers") from None
    if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
        raise SettingsError(f"roi={text!r}: must be x0,y0,x1,y1 with x0 < x1, y0 < y1")
    return [x0, y0, x1, y1]


def form_value(field: str, value: str) -> object:
    """A page form field (``FORM_FIELDS``) parsed; :class:`SettingsError` if bad."""
    return _form_value(field, FORM_FIELDS[field], value)


def _form_value(field: str, key: str, value: str) -> object:
    """One form field as its settings value, or :class:`SettingsError`."""
    try:
        if key in _BOOLEANS:
            if value not in ("true", "false"):
                raise ValueError("must be true or false")
            return value == "true"
        if key in _PARSERS:
            return _PARSERS[key](value)
        if key in _INTEGERS:
            low, high = _INTEGERS[key]
            number = int(value)
            if number < low or (high is not None and number > high):
                raise ValueError(
                    f"must be {low} or more"
                    if high is None
                    else f"must be {low} to {high}"
                )
            return number
        return _text_value(key, value)
    except ValueError as exc:
        raise SettingsError(f"{field}={value!r}: {exc}") from None


def _centre_value(value: str) -> Optional[float]:
    """The centring threshold: ``None`` (off) for ``""``, else a number above 0."""
    if not value:
        return None
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(
            "must be a number above 0 (counts, or below 1 a fraction of the peak)"
        )
    return number


def _tile_max_value(text: str) -> Optional[int]:
    """The max tile width: ``None`` (no limit) for ``""``, else whole pixels in range."""
    if not text.strip():
        return None
    number = int(text)
    low, high = TILE_MAX_RANGE
    if not low <= number <= high:
        raise ValueError(f"must be {low} to {high} pixels, or empty for no limit")
    return number


def _bg_value(text: str) -> Optional[ScanRef]:
    """The background scan, ``YYYY-MM-DD:N``; ``None`` (no background) for ``""``."""
    refs = scan_refs(text)
    if len(refs) > 1:
        raise ValueError("one background scan, not several")
    return refs[0] if refs else None


#: The fields with a parser of their own.
_PARSERS: dict[str, Callable[[str], object]] = {
    "day": date.fromisoformat,
    "extra_scans": scan_refs,
    "image_centre": _centre_value,
    "image_bg": _bg_value,
    "image_tile_max": lambda text: _tile_max_value(text),
    "merge_tolerance": lambda text: _tolerance_value(text),
    "image_channels": lambda text: channels_value(text),
}


def channels_value(text: str) -> list[int]:
    """A trace row's channels, ``"0,2,3"``: each once, in order, at least one.

    The form's spelling of ``image_channels`` / ``ImageCamera.channels``
    (the row's checkboxes), and the routes' ``channel=`` repeated.
    """
    picked: list[int] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        number = int(part)
        if not 0 <= number <= MAX_CHANNEL:
            raise ValueError(f"a channel is 0 to {MAX_CHANNEL}")
        if number not in picked:
            picked.append(number)
    if not picked:
        raise ValueError("pick at least one channel")
    return sorted(picked)


def _tolerance_value(text: str) -> Optional[float]:
    """The merge tolerance box: a number, 0 or more; empty is the default (half a step)."""
    if not text.strip():
        return None
    number = float(text)
    if not (math.isfinite(number) and number >= 0):
        raise ValueError("must be a number, 0 or more (empty: half the smallest step)")
    return number


def _text_value(key: str, value: str) -> object:
    """A fit, choice, percentile, colour-map or device field; ``ValueError`` if bad."""
    if key.endswith("_fit") and not _FIT_NAME.fullmatch(value):
        raise ValueError("is not a fit name")
    if key in CHOICES and value not in CHOICES[key]:
        raise ValueError(f"must be one of {', '.join(CHOICES[key])}")
    if key in _LEVELS:
        value = value.strip()  # a box holding only spaces is "no limit"
        if not value:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("must be a number")
        return number
    if key == "image_cmap" and not _CMAP_NAME.fullmatch(value):
        raise ValueError("is not a colour-map name")
    if key in ("image_device", "map_value", "map_x", "map_y") and not value:
        return None  # the default: the scan's first camera, Y1, the scan's axes
    if key in _JSON_FIELDS and value:
        from geecs_portal import analysis

        try:
            getattr(analysis, _JSON_FIELDS[key])(value)
        except analysis.BadParam as bad:
            raise ValueError(str(bad)) from None
    return value
