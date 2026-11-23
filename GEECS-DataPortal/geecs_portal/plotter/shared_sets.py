"""Shared settings sets: files in the configs repo, saved through the configurator.

A viewer's own set (their name) is their working copy: saved on every
pick, into this host's settings folder (:class:`~geecs_portal.plotter.settings.SettingsStore`),
never committed. **Save as** shares a set instead: the portal asks the
configurator (``GeecsConfigurator``, ``--configurator-url``, a site value)
over HTTP to save ``plot_settings/<stem>.yaml`` in the experiment's
configs folder, and the configurator validates it, commits it to the
configs repo's main as the viewer's (the operator tier) and fast-forwards
the checkouts. Every portal reads the shared sets from its configs root
(``--configs-root``), so a set saved on one host opens on every other.

A shared set is named ``shared/<stem>`` in a link and on the page
(:data:`~geecs_portal.plotter.settings.SHARED_PREFIX`). A viewer's change to
a shared set copies it into their own set, as with anyone else's set; the
shared set changes only by another Save as onto its name.

The portal never writes the configs tree. Without a configurator, or
without a configs root, Save as is refused and says which flag is missing.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import httpx
import yaml
from geecs_schemas.plot_settings import PLOT_SETTINGS_FOLDER, PlotterSettings
from geecs_schemas.storage import STORAGE_ROOT_PATHS
from pydantic import ValidationError

from geecs_portal.plotter.settings import (
    SHARED_STEM,
    SHARED_PREFIX,
    NameRefused,
    SettingsError,
    SettingsStore,
    is_shared,
    set_name,
    shared_stem,
)

log = logging.getLogger(__name__)

#: The configurator's kind name for a plot settings set (GEECS-Schemas' registry).
KIND = "plot_settings"

#: One round trip's bound (seconds): a save fetches, validates, commits and pushes.
TIMEOUT_S = 60.0

#: How long a set just saved is answered from memory while the configs root
#: has not caught up with its commit (the configurator moves the share's
#: clone on a thread after the commit; seconds, not minutes).
PENDING_S = 120.0

#: The suffixes the configurator reads for this kind.
_SUFFIXES = (".yaml", ".yml")


class SetSaver(Protocol):
    """Saves one shared set (the configurator; a double in the tests)."""

    def save(
        self, stem: str, document: dict[str, Any], *, operator: str, overwrite: bool
    ) -> str:
        """Save *document* as ``<stem>.yaml``; the sentence to show. Refusals are :class:`SettingsError`."""
        ...


class ConfiguratorSets:
    """The configurator's document API, for the one kind the portal saves.

    Parameters
    ----------
    base_url : str
        The configurator as this process reaches it (``http://127.0.0.1:8800``).
    experiment : str
        The experiment whose ``plot_settings/`` folder the save writes.
    client : httpx.Client, optional
        The HTTP client; the tests pass one over the configurator's app.
    """

    def __init__(
        self,
        base_url: str,
        experiment: str,
        *,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.experiment = experiment
        self._client = client

    def _call(
        self, method: str, path: str, body: Optional[dict[str, Any]] = None
    ) -> tuple[int, Any]:  # Any: a JSON answer of any shape
        """One request; ``(status, json)``. A configurator that does not answer is a :class:`SettingsError`."""
        url = f"{self.base_url}/api/{path}"
        params = {"experiment": self.experiment}
        try:
            if self._client is not None:
                reply = self._client.request(method, url, params=params, json=body)
            else:
                with httpx.Client(timeout=TIMEOUT_S) as client:
                    reply = client.request(method, url, params=params, json=body)
        except httpx.HTTPError as exc:
            raise SettingsError(
                f"the configurator did not answer ({self.base_url}): {exc}"
            ) from exc
        try:
            return reply.status_code, reply.json()
        except ValueError:
            return reply.status_code, reply.text[:200]

    def _existing(self, stem: str) -> Optional[str]:
        """The configurator's id of ``<stem>.yaml`` (or its ``.yml`` twin), or ``None``."""
        status, listing = self._call("GET", f"documents/{KIND}")
        if status != 200 or not isinstance(listing, list):
            raise SettingsError(
                f"the configurator could not list the shared sets ({status}): "
                f"{_detail(listing)}"
            )
        ids = {str(e.get("id")) for e in listing if isinstance(e, dict)}
        return next((f"{stem}{s}" for s in _SUFFIXES if f"{stem}{s}" in ids), None)

    def save(
        self, stem: str, document: dict[str, Any], *, operator: str, overwrite: bool
    ) -> str:
        """A create (refused over a file that is there), or with *overwrite* the whole document over its etag."""
        doc_id = self._existing(stem)
        if doc_id is not None and not overwrite:
            raise taken(stem)
        etag = ""
        if doc_id is not None:
            quoted = urllib.parse.quote(doc_id)
            status, current = self._call("GET", f"documents/{KIND}/{quoted}")
            if status != 200 or not isinstance(current, dict):
                raise SettingsError(
                    f"the configurator could not read the shared set {stem!r} "
                    f"({status}): {_detail(current)}"
                )
            etag = str(current.get("etag") or "")
        target = urllib.parse.quote(doc_id or f"{stem}.yaml")
        body = {
            "document": document,
            "etag": etag,
            "operator": operator,
            "create": doc_id is None,
        }
        status, saved = self._call("PUT", f"documents/{KIND}/{target}", body)
        if status == 200 and isinstance(saved, dict):
            outcome = saved.get("outcome")
            if outcome == "committed":
                commit = str(saved.get("commit") or "")[:10]
                return f"shared as {SHARED_PREFIX}{stem}, committed to the configs repo ({commit}) as {operator}'s"  # fmt: skip
            if outcome == "nothing to do":
                return f"{SHARED_PREFIX}{stem} is unchanged: the configs repo already holds exactly this set"  # fmt: skip
            return f"{SHARED_PREFIX}{stem}: {outcome}"
        reason = _detail(saved)
        if status == 409 and doc_id is None:  # created meanwhile by someone else
            raise taken(stem)
        if status == 409:
            raise SettingsError(
                f"{SHARED_PREFIX}{stem} changed while it was saved; save again ({reason})"
            )
        raise SettingsError(
            f"the configurator did not save {SHARED_PREFIX}{stem} ({status}): {reason}"
        )


def taken(stem: str) -> NameRefused:
    """The refusal of a Save as onto a shared set that is there, without overwrite."""
    return NameRefused(f"a shared set named {stem!r} is already saved")


def _detail(payload: Any) -> str:  # Any: a JSON answer of any shape
    """The configurator's ``detail`` as one line (FastAPI's error shape), else the payload."""
    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, dict) and "errors" in detail:
        return "; ".join(
            f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}"
            for e in detail["errors"]
        )
    return str(detail)


def shared_folder(configs_root: Path, experiment: str) -> Path:
    """The experiment's shared-set folder under the configs repo's root."""
    experiments = STORAGE_ROOT_PATHS["experiment"].replace("<experiment>", experiment)
    return configs_root / experiments / PLOT_SETTINGS_FOLDER


@dataclass
class SharedSets:
    """The shared sets: read from the configs root, saved through *saver*.

    Parameters
    ----------
    folder : Path, optional
        The experiment's ``plot_settings`` folder in the configs root;
        ``None``: no configs root, no shared sets.
    saver : SetSaver, optional
        The configurator; ``None``: Save as is refused.
    clock : callable, optional
        The time source, for the tests.
    """

    folder: Optional[Path] = None
    saver: Optional[SetSaver] = None
    clock: Callable[[], float] = time.monotonic
    #: Sets saved here whose commit the configs root may not show yet:
    #: stem -> (when, the set as saved).
    _pending: dict[str, tuple[float, PlotterSettings]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _file(self, stem: str) -> Optional[Path]:
        if self.folder is None:
            return None
        return next(
            (p for s in _SUFFIXES if (p := self.folder / f"{stem}{s}").is_file()), None
        )

    def _read(self, path: Path) -> PlotterSettings:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SettingsError(f"cannot read {path}: {exc}") from exc
        try:
            return PlotterSettings.model_validate({} if data is None else data)
        except ValidationError as exc:
            raise SettingsError(
                f"{path} is not a plotter settings file: {exc}"
            ) from exc

    def _fresh_pending(self, stem: str) -> Optional[PlotterSettings]:
        with self._lock:
            held = self._pending.get(stem)
            if held is None:
                return None
            if self.clock() - held[0] > PENDING_S:
                del self._pending[stem]
                return None
            return held[1]

    def stems(self) -> list[str]:
        """Every shared set's stem, the ones just saved included."""
        found: set[str] = set()
        if self.folder is not None and self.folder.is_dir():
            # Only a file named as Save as names it (a stray name is no set).
            found |= {
                p.stem
                for p in self.folder.iterdir()
                if p.suffix in _SUFFIXES and SHARED_STEM.fullmatch(p.stem)
            }
        with self._lock:
            found |= set(self._pending)
        return sorted(found)

    def exists(self, stem: str) -> bool:
        """Whether ``<stem>.yaml`` is a shared set (or was just saved as one)."""
        return self._file(stem) is not None or self._fresh_pending(stem) is not None

    def load(self, stem: str) -> PlotterSettings:
        """The shared set; the defaults for one never saved.

        A set this portal just saved is answered as saved until the
        configs root holds the same document (or :data:`PENDING_S` passes).
        """
        pending = self._fresh_pending(stem)
        path = self._file(stem)
        on_disk = self._read(path) if path is not None else None
        if pending is not None:
            if on_disk is not None and on_disk == pending:
                with self._lock:
                    self._pending.pop(stem, None)
            return pending
        return on_disk if on_disk is not None else PlotterSettings()

    def save(
        self, stem: str, settings: PlotterSettings, *, operator: str, overwrite: bool
    ) -> None:
        """Share *settings* as ``<stem>``; the configurator's answer goes to the log."""
        if self.saver is None:
            raise SettingsError(
                "Save as shares the set through the configurator, and this "
                "portal has none: start it with --configurator-url"
            )
        if self.folder is None:
            raise SettingsError(
                "Save as shares the set in the configs repo, and this portal "
                "reads none: start it with --configs-root"
            )
        document = settings.model_dump(mode="json", exclude_defaults=True)
        document["schema_version"] = settings.schema_version
        log.info(
            self.saver.save(stem, document, operator=operator, overwrite=overwrite)
        )
        with self._lock:
            self._pending[stem] = (self.clock(), settings)


@dataclass(frozen=True)
class SetBook:
    """Every set the page can open: this host's (working copies) and the shared ones.

    A name with :data:`~geecs_portal.plotter.settings.SHARED_PREFIX` is a
    shared set's; any other is this host's store.
    """

    local: SettingsStore
    shared: SharedSets = field(default_factory=SharedSets)

    @property
    def folder(self) -> Path:
        """The settings folder of this host (the working copies)."""
        return self.local.folder

    def local_names(self) -> list[str]:
        """The sets kept on this host, the default first."""
        return self.local.names()

    def shared_names(self) -> list[str]:
        """The shared sets, as ``shared/<stem>``."""
        return [SHARED_PREFIX + s for s in self.shared.stems()]

    def names(self) -> list[str]:
        """Every set: this host's, then the shared ones."""
        return [*self.local_names(), *self.shared_names()]

    def load(self, name: str) -> PlotterSettings:
        """The set *name* names; the defaults for one never saved."""
        stored = set_name(name)
        if is_shared(stored):
            return self.shared.load(stored[len(SHARED_PREFIX) :])
        return self.local.load(stored)

    def save(self, name: str, settings: PlotterSettings) -> Path:
        """Save a working copy on this host; a shared set's name is refused."""
        return self.local.save(name, settings)

    def save_as(
        self,
        name: str,
        settings: PlotterSettings,
        *,
        operator: str,
        overwrite: bool,
    ) -> str:
        """Share *settings* under the Save-as *name*; the shared set's name (``shared/<stem>``)."""
        if not operator:
            raise SettingsError(
                "say who you are before sharing a set: it is committed under "
                "your name (the page asks your name)"
            )
        stem = shared_stem(name)
        if not overwrite and self.shared.exists(stem):
            raise taken(stem)
        self.shared.save(stem, settings, operator=operator, overwrite=overwrite)
        return SHARED_PREFIX + stem
