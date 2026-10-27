"""Named settings sets: one JSON file per set, in one folder.

GEECSplotter keeps its settings in the LabVIEW ``user data`` folder, and
so does this: ``{geecs_data}/GeecsPlotter/settings/``, where
``geecs_data`` is ``config.ini [Paths] geecs_data`` (the same folder the
GEECS database credentials are read from, usually on the shared drive).
When that key is unset or its folder is not reachable, the sets live in
``<repo>/user data/GeecsPlotter/settings/`` (gitignored) instead, where
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
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

#: The set a page opens when the URL names none.
DEFAULT_SET = "default"

CONFIG_INI = Path("~/.config/geecs_python_api/config.ini").expanduser()


def checkout_root(package_file: Path) -> Path:
    """The checkout *package_file* is in, or the main one for a worktree.

    Session worktrees live at ``<repo>/.claude/worktrees/<name>/`` (root
    CLAUDE.md); their local sets belong with the main checkout's, or each
    worktree would keep its own and lose the others'.
    """
    repo = package_file.resolve().parents[2]
    if repo.parent.name == "worktrees" and repo.parents[1].name == ".claude":
        return repo.parents[2]
    return repo


#: ``<repo>/user data/``: the local fallback, beside the (main) checkout.
LOCAL_USER_DATA = checkout_root(Path(__file__)) / "user data"

_SUBFOLDER = Path("GeecsPlotter") / "settings"

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
    #: GEECSplotter's Y2 and Y3, each on its own axis; ``None`` or ``""``
    #: is off.
    y2: Optional[str] = None
    y3: Optional[str] = None
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
    #: The scan last opened under this set, so opening the set (or starting
    #: the plotter) goes back to it, as LabVIEW reopens its last file.
    day: Optional[date] = None
    scan: Optional[int] = None


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
        marker.write_text("# GeecsPlotter's local settings: never committed.\n*\n")


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
        return SettingsStore(shared / _SUBFOLDER, "shared", root=shared)
    return SettingsStore(
        LOCAL_USER_DATA / _SUBFOLDER, "local", root=LOCAL_USER_DATA.parent
    )


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
    "day": "day",
    "scan": "scan",
}

_BOOLEANS = ("x_alias", "y_alias", "x_auto", "y2_alias", "y3_alias")


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
    return settings.model_copy(update=changes)


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
        if key == "scan":
            number = int(value)
            if number < 1:
                raise ValueError("must be 1 or more")
            return number
    except ValueError as exc:
        raise SettingsError(f"{field}={value!r}: {exc}") from None
    return value
