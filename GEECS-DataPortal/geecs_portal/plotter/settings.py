"""Named settings sets: one JSON file per set, in one folder.

GEECSplotter keeps its settings in the LabVIEW ``user data`` folder, and
so does this: ``{geecs_data}/DataPortal/settings/``, where
``geecs_data`` is ``config.ini [Paths] geecs_data`` (the same folder the
GEECS database credentials are read from, usually on the shared drive).
When that key is unset or its folder is not reachable, the sets live in
``<repo>/user data/DataPortal/settings/`` (gitignored) instead, where
``<repo>`` is the main checkout even when the plotter runs from a session
worktree under ``.claude/worktrees/``, so every copy sees the same sets.
``--settings-dir`` overrides both. The folder is chosen once, at start.

A set is a :class:`PlotterSettings` document, ``<name>.json``: readable,
copyable, and versioned so an older file loads with new fields at their
defaults.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

#: Y, Y2 and Y3: the most Y lists a set holds (``figures.MAX_Y`` draws them).
MAX_Y_LISTS = 3

#: The highest polynomial order the Fit's order box takes.
MAX_FIT_ORDER = 10

#: The most image tiles across.
MAX_IMAGE_COLUMNS = 20

#: The choices of the tab and image controls, as stored.
CHOICES: dict[str, tuple[str, ...]] = {
    "tab": ("xy", "images"),
    "image_strip": ("bins", "shots"),
    "image_layout": ("grid", "row"),
    "image_zscale": ("tile", "shared", "fixed"),
}

#: A fit name as stored: GEECSplotter's Fit menu (``figures.FITS``) or one
#: a newer copy offers, kept rather than refused (see ``extra`` below).
_FIT_NAME = re.compile(r"[a-z0-9_]{1,40}")

#: The set a page opens when the URL names none.
DEFAULT_SET = "default"

CONFIG_INI = Path("~/.config/geecs_python_api/config.ini").expanduser()


def checkout_root(package_file: Path) -> Path:
    """The checkout *package_file* is in, or the main one for a worktree.

    Session worktrees live at ``<repo>/.claude/worktrees/<name>/`` (root
    CLAUDE.md); their local sets belong with the main checkout's, or each
    worktree would keep its own and lose the others'.
    """
    repo = package_file.resolve().parents[3]
    if repo.parent.name == "worktrees" and repo.parents[1].name == ".claude":
        return repo.parents[2]
    return repo


#: ``<repo>/user data/``: the local fallback, beside the (main) checkout.
LOCAL_USER_DATA = checkout_root(Path(__file__)) / "user data"

_SUBFOLDER = Path("DataPortal") / "settings"

#: Where GeecsPlotter kept the sets, before it folded into the portal: its
#: sets are copied from here once (:func:`adopt_plotter_sets`).
_PLOTTER_SUBFOLDER = Path("GeecsPlotter") / "settings"

#: A letter or digit first, then letters, digits, spaces, ``-``, ``_`` and
#: ``.``; never ending in a space or a dot (Windows drops both from names).
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9 ._-]{0,62}[a-z0-9_-])?")

Origin = Literal["given", "shared", "local"]


class SettingsError(ValueError):
    """A set name that is not allowed, or a set file that cannot be read."""


class NameRefused(SettingsError):
    """A Save-as name that cannot be used: the name is the thing to fix."""


class PlotterSettings(BaseModel):
    """Every control on the page; each field has its default."""

    # Unknown fields are kept, not refused: the shared folder is read by
    # every copy of the plotter, and an older copy must still load (and
    # save, without dropping) a set a newer copy wrote with more fields.
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[1] = 1
    #: ``None`` is the scan variable; ``""`` is the shot number.
    x: Optional[str] = None
    #: ``None`` is the first measurement column.
    y: Optional[str] = None
    #: GEECSplotter's Y2 and Y3, each on its own axis; ``None`` is "not
    #: picked yet" (the page offers a default).
    y2: Optional[str] = None
    y3: Optional[str] = None
    #: How many of Y, Y2, Y3 are shown and drawn. The lists past it keep
    #: their picks for when the count comes back up.
    y_count: int = Field(default=1, ge=1, le=MAX_Y_LISTS)
    x_filter: str = ""
    y_filter: str = ""
    y2_filter: str = ""
    y3_filter: str = ""
    x_alias: bool = True
    y_alias: bool = True
    y2_alias: bool = True
    y3_alias: bool = True
    #: GEECSplotter's "autofill par1": X is the open scan's variable.
    x_auto: bool = True
    #: The Fit per Y list (``figures.FITS``; ``"none"`` is no fit), and the
    #: polynomial order each uses when its fit is ``"polynomial"``. A plain
    #: string, so a set naming a fit a newer copy added still loads here.
    y_fit: str = "none"
    y2_fit: str = "none"
    y3_fit: str = "none"
    y_fit_order: int = Field(default=2, ge=0, le=MAX_FIT_ORDER)
    y2_fit_order: int = Field(default=2, ge=0, le=MAX_FIT_ORDER)
    y3_fit_order: int = Field(default=2, ge=0, le=MAX_FIT_ORDER)
    #: The scan last opened under this set, so opening the set (or starting
    #: the plotter) goes back to it, as LabVIEW reopens its last file.
    day: Optional[date] = None
    scan: Optional[int] = None
    #: The tab shown: the XY plot or the image strip.
    tab: str = "xy"
    #: The image strip: its camera (``None``: the scan's first), the device
    #: list's filter, one tile per bin or per shot, a single row or a grid,
    #: tiles across, and the display (colour map and percentile window).
    image_device: Optional[str] = None
    image_device_filter: str = ""
    image_strip: str = "bins"
    image_layout: str = "grid"
    image_columns: int = Field(default=5, ge=1, le=MAX_IMAGE_COLUMNS)
    #: A matplotlib name; one this copy does not know draws greyscale.
    image_cmap: str = "gray"
    #: The z scale, LabVIEW's autoscale: each tile from its own darkest to
    #: its brightest pixel ("tile"), every tile on one scale from the
    #: darkest to the brightest pixel of all of them ("shared"), or the
    #: black and white values below ("fixed"). No percentile window: a
    #: laser mode is a small spot, and any top-percentile white saturates
    #: it. (Sets saved with ``image_plo``/``image_phi`` keep them as
    #: unused extras.)
    image_zscale: str = "tile"
    image_zmin: Optional[float] = None
    image_zmax: Optional[float] = None
    #: Each camera's region of interest, ``[x0, y0, x1, y1]`` in pixels
    #: (x right, y down, the end excluded); a camera not here is whole.
    image_rois: dict[str, list[int]] = Field(default_factory=dict)


def set_name(name: str) -> str:
    """The name as stored (lower case), or :class:`SettingsError`.

    Names compare case-insensitively, as Windows and macOS paths do, and
    runs of spaces count as one.
    """
    folded = " ".join(name.split()).lower()
    if not _NAME.fullmatch(folded):
        raise SettingsError(
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
    origin : {"given", "shared", "local"}
        Why this folder: ``--settings-dir``, ``config.ini``'s
        ``geecs_data``, or the local fallback. The page shows it.
    root : Path, optional
        A folder that must already exist for the store to be read or
        written: the ``geecs_data`` folder for the shared store, the
        checkout for the local one. A share that drops after start is then
        an error, never "no sets saved", and nothing is built on a bare
        mount point. ``None`` (a ``--settings-dir``) checks nothing, and
        the first save creates the folder.
    """

    folder: Path
    origin: Origin = "given"
    root: Optional[Path] = None

    def path(self, name: str) -> Path:
        """The file a set is stored in."""
        return self.folder / f"{set_name(name)}.json"

    def reachable(self) -> bool:
        """Whether the store's root is there (the share is mounted)."""
        return self.root is None or self.root.is_dir()

    def _require_reachable(self) -> None:
        if not self.reachable():
            raise SettingsError(
                f"{self.root} is not reachable; "
                "settings are neither read nor saved until it is back"
            )

    def names(self) -> list[str]:
        """Every saved set, with the default first whether saved or not."""
        saved = (
            {p.stem for p in self.folder.glob("*.json") if _NAME.fullmatch(p.stem)}
            if self.reachable() and self.folder.is_dir()
            else set()
        )
        return [DEFAULT_SET, *sorted(saved - {DEFAULT_SET})]

    def exists(self, name: str) -> bool:
        """Whether *name* has been saved."""
        return self.path(name).is_file()

    def load(self, name: str) -> PlotterSettings:
        """The set, or the defaults for a name never saved."""
        path = self.path(name)
        self._require_reachable()
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
        """Write the set (atomically, except on Windows: see ``_write_whole``)."""
        path = self.path(name)
        self._require_reachable()
        text = settings.model_dump_json(indent=2) + "\n"
        try:
            # Only below the root, which exists: never a tree on a bare mount.
            self.folder.mkdir(parents=True, exist_ok=True)
            if self.origin == "local" and self.root is not None:
                _ignore_in_git(self.root / LOCAL_USER_DATA.name)
            _write_whole(path, text)
        except OSError as exc:
            raise SettingsError(f"cannot save {path}: {exc}") from exc
        return path


def _ignore_in_git(user_data: Path) -> None:
    """Make the local ``user data`` folder ignore itself.

    The root ``.gitignore`` lists it too, but a checkout on an older
    branch does not, and sets must never show up as files to commit.
    """
    marker = user_data / ".gitignore"
    if not marker.exists():
        marker.write_text("# Local settings sets: never committed.\n*\n")


def _write_whole(path: Path, text: str) -> None:
    """Write *path* through a uniquely named sibling and a rename.

    The shape of ``geecs_logbook._fs.replace_with`` (a unique temp name per
    call, so two hosts saving one set never clobber each other's half
    file), with ``scan_analysis.task_queue._write_atomic``'s Windows rule:
    on an SMB share the rename needs DELETE permission on the target,
    usually only its creator's, so Windows writes in place (a reader may
    then catch half a file: a passing "Settings problem"). The repo's
    private atomic writers are to be folded into one under #839.
    """
    if os.name == "nt":
        path.write_text(text, encoding="utf-8")
        return
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        # mkstemp's 0600 would hide the set from colleagues; best effort,
        # since a CIFS mount without unix extensions refuses chmod.
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def geecs_user_data(config_path: Path = CONFIG_INI) -> Optional[Path]:
    """``config.ini [Paths] geecs_data``, when it is set and reachable."""
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        value = config.get("Paths", "geecs_data", fallback="").strip()
    except configparser.Error:
        return None
    if not value:
        return None
    folder = Path(os.path.expandvars(value)).expanduser()
    return folder if folder.is_dir() else None


def resolve_store(
    settings_dir: Optional[Path] = None, *, config_path: Path = CONFIG_INI
) -> SettingsStore:
    """The store for this process: given, else shared, else local."""
    if settings_dir is not None:
        return SettingsStore(Path(settings_dir).expanduser(), "given")
    shared = geecs_user_data(config_path)
    if shared is not None:
        store = SettingsStore(shared / _SUBFOLDER, "shared", root=shared)
        adopt_plotter_sets(store, shared / _PLOTTER_SUBFOLDER)
        return store
    store = SettingsStore(
        LOCAL_USER_DATA / _SUBFOLDER, "local", root=LOCAL_USER_DATA.parent
    )
    adopt_plotter_sets(store, LOCAL_USER_DATA / _PLOTTER_SUBFOLDER)
    return store


def adopt_plotter_sets(store: SettingsStore, plotter_folder: Path) -> int:
    """Copy GeecsPlotter's sets into *store* once; how many were copied.

    Only while *store*'s folder does not exist yet, so the copy happens on
    the portal's first start beside a plotter folder and never again. It
    never overwrites and never deletes: the plotter keeps its own files.
    A share that refuses is logged and skipped; the portal starts anyway.
    """
    if store.folder.exists() or not plotter_folder.is_dir():
        return 0
    copied = 0
    try:
        store.folder.mkdir(parents=True)
        if store.origin == "local" and store.root is not None:
            _ignore_in_git(store.root / LOCAL_USER_DATA.name)
        for source in sorted(plotter_folder.glob("*.json")):
            target = store.folder / source.name
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
    except OSError as exc:
        logger.warning(
            "could not copy GeecsPlotter's sets from %s: %s", plotter_folder, exc
        )
    else:
        logger.info("copied %d GeecsPlotter sets into %s", copied, store.folder)
    return copied


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
    "layout": "image_layout",
    "cols": "image_columns",
    "cmap": "image_cmap",
    "zscale": "image_zscale",
    "zmin": "image_zmin",
    "zmax": "image_zmax",
}

_BOOLEANS = ("x_alias", "y_alias", "x_auto", "y2_alias", "y3_alias")

#: The whole-number fields and their bounds (``None``: no upper bound).
_INTEGERS: dict[str, tuple[int, Optional[int]]] = {
    "scan": (1, None),
    "y_count": (1, MAX_Y_LISTS),
    "y_fit_order": (0, MAX_FIT_ORDER),
    "y2_fit_order": (0, MAX_FIT_ORDER),
    "y3_fit_order": (0, MAX_FIT_ORDER),
    "image_columns": (1, MAX_IMAGE_COLUMNS),
}

#: The fixed z-scale values: any finite number, or empty for none.
_LEVELS = ("image_zmin", "image_zmax")

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
    if "roi" in form:
        device = changes.get("image_device") or settings.image_device
        if device:
            rois = dict(settings.image_rois)
            roi = roi_value(form["roi"])
            if roi is None:
                rois.pop(device, None)
            else:
                rois[device] = roi
            changes["image_rois"] = rois
    return settings.model_copy(update=changes)


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
        if key == "day":
            return date.fromisoformat(value)
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


def _text_value(key: str, value: str) -> object:
    """A fit, choice, percentile, colour-map or device field; ``ValueError`` if bad."""
    if key.endswith("_fit") and not _FIT_NAME.fullmatch(value):
        raise ValueError("is not a fit name")
    if key in CHOICES and value not in CHOICES[key]:
        raise ValueError(f"must be one of {', '.join(CHOICES[key])}")
    if key in _LEVELS:
        if not value:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("must be a number")
        return number
    if key == "image_cmap" and not _CMAP_NAME.fullmatch(value):
        raise ValueError("is not a colour-map name")
    if key == "image_device" and not value:
        return None
    return value
