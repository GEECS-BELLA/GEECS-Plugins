"""Saving a preset: a commit to the configs repo's main, made by the configurator.

The scanner never writes the configs tree. "Save as preset" asks the
configurator (``GeecsConfigurator``, a site value: ``--configurator-url``)
over HTTP to save ``presets/<name>.yaml``: it reads the document's etag
(none for a new preset), then sends the whole document (a ``PUT``) with the
operator, and the configurator validates it, commits it to main as the
operator's (its operator tier) and fast-forwards the checkout the scanner
reads. The scanner imports nothing of it (a peer service, the GEECS-MCP
posture) and uses the standard library's HTTP client, so it adds no
dependency.

Without a configurator nothing is saved, and the answer says how to
configure one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from geecs_scanner.service.errors import ScannerError

#: ``(method, url, body) -> (status, body)``: one HTTP exchange.
Send = Callable[[str, str, Optional[bytes]], tuple[int, bytes]]

#: The configurator's kind name for a scan preset (GEECS-Schemas' registry).
KIND = "preset"
#: One round trip's bound (seconds): a save fetches, validates, commits and pushes.
TIMEOUT_S = 60.0


@dataclass(frozen=True)
class SavedPreset:
    """What a save did: the file below the configs root, the commit (``None`` when nothing changed), a sentence."""

    path: str
    commit: Optional[str]
    message: str


class PresetSaver(Protocol):
    """Saves one preset document (the configurator; an in-memory double in the demo)."""

    def save(self, name: str, document: dict[str, Any], *, operator: str, overwrite: bool) -> SavedPreset:  # fmt: skip
        """Save *document* as preset *name*: created, or replaced only with *overwrite*; a :class:`ScannerError` when refused.

        One already there without *overwrite* is ``policy_refusal`` with
        ``exists=True`` (the page's "replace?" dialog), decided by the
        configurator, whatever this scanner's own checkout lists.
        """
        ...


class NoConfigurator:
    """No configurator configured: every save is refused, saying how to configure one."""

    def save(self, name: str, document: dict[str, Any], *, operator: str, overwrite: bool) -> SavedPreset:  # fmt: skip
        """Refuse: the scanner does not write the configs tree itself."""
        raise ScannerError(
            "policy_refusal",
            "presets are saved through the configurator, and none is configured: "
            "start the scanner with --configurator-url (the site's "
            "GEECS_SCANNER_EXTRA_ARGS)",
        )


class ConfiguratorPresets:
    """The configurator's document API, for the one kind the scanner saves.

    Parameters
    ----------
    base_url : str
        The configurator as this process reaches it (``http://127.0.0.1:8800``).
    experiment : str
        The experiment whose ``presets/`` folder the save writes.
    """

    def __init__(self, base_url: str, experiment: str, *, timeout: float = TIMEOUT_S, send: Optional[Send] = None):  # fmt: skip
        self.base_url = base_url.rstrip("/")
        self.experiment = experiment
        self.timeout = timeout
        #: The transport: the standard library's by default; contract tests pass the configurator's app.
        self._send: Send = send or self._urllib

    def _urllib(self, method: str, url: str, data: Optional[bytes]) -> tuple[int, bytes]:  # fmt: skip
        request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})  # fmt: skip
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return reply.status, reply.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _call(self, method: str, path: str, body: Optional[dict[str, Any]] = None) -> tuple[int, Any]:  # fmt: skip
        """One request; ``(status, json)``. A configurator that does not answer is ``manager_unreachable``."""
        query = urllib.parse.urlencode({"experiment": self.experiment})
        url = f"{self.base_url}/api/{path}?{query}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        try:
            status, raw = self._send(method, url, data)
        except (urllib.error.URLError, OSError) as exc:
            raise ScannerError(
                "manager_unreachable", f"the configurator did not answer ({url}): {exc}"
            ) from exc
        try:
            return status, json.loads(raw or b"null")
        except ValueError:
            return status, raw.decode("utf-8", errors="replace")[:200]

    def _existing(self, name: str) -> Optional[str]:
        """The configurator's id of preset *name* (``name.yaml`` or its ``.yml`` twin), or ``None``."""
        status, listing = self._call("GET", f"documents/{KIND}")
        if status == 404:
            raise ScannerError("invalid_request", f"the configurator has no preset folder for experiment {self.experiment!r}: {_detail(listing)}")  # fmt: skip
        if status != 200 or not isinstance(listing, list):
            raise ScannerError("manager_unreachable", f"the configurator's preset list answered {status}: {_detail(listing)}")  # fmt: skip
        ids = [str(e.get("id")) for e in listing if isinstance(e, dict)]
        for suffix in (".yaml", ".yml"):
            if f"{name}{suffix}" in ids:
                return f"{name}{suffix}"
        return None

    def save(self, name: str, document: dict[str, Any], *, operator: str, overwrite: bool) -> SavedPreset:  # fmt: skip
        """A create (refused over a file that is there), or, with *overwrite*, the whole document over its current etag."""
        doc_id = self._existing(name)
        if doc_id is not None and not overwrite:
            raise exists_refusal(name)
        etag = ""
        if doc_id is not None:
            status, current = self._call("GET", f"documents/{KIND}/{urllib.parse.quote(doc_id)}")  # fmt: skip
            if status != 200 or not isinstance(current, dict):
                raise ScannerError("manager_unreachable", f"the configurator could not read preset {name!r} ({status}): {_detail(current)}")  # fmt: skip
            etag = str(current.get("etag") or "")
        target = doc_id or f"{name}.yaml"
        body = {"document": document, "etag": etag, "operator": operator, "create": doc_id is None}  # fmt: skip
        status, saved = self._call("PUT", f"documents/{KIND}/{urllib.parse.quote(target)}", body)  # fmt: skip
        if status == 200 and isinstance(saved, dict):
            commit = saved.get("commit")
            path = str(saved.get("path") or target)
            if saved.get("outcome") == "committed":
                message = f"preset {name!r} committed to the configs repo's main ({str(commit)[:10]}) as {operator}'s"  # fmt: skip
            elif saved.get("outcome") == "nothing to do":
                message = f"preset {name!r} is unchanged: main already holds exactly this document"  # fmt: skip
            else:
                message = f"preset {name!r}: {saved.get('outcome')} ({saved.get('pull_request') or 'no link'})"  # fmt: skip
            return SavedPreset(path=path, commit=commit, message=message)
        reason = _detail(saved)
        if status == 409 and doc_id is None:  # created meanwhile by someone else
            raise exists_refusal(name)
        if status == 409:
            raise ScannerError("conflict", f"preset {name!r} changed while it was saved; reload it and save again ({reason})")  # fmt: skip
        if status in (400, 403, 404, 405, 422, 428):
            raise ScannerError("invalid_request", f"the configurator refused preset {name!r}: {reason}")  # fmt: skip
        raise ScannerError("manager_unreachable", f"the configurator did not save preset {name!r} ({status}): {reason}")  # fmt: skip


def exists_refusal(name: str) -> ScannerError:
    """The refusal of a save over a preset that is there without overwrite (the page's "replace?" dialog keys on ``exists``)."""
    return ScannerError("policy_refusal", f"preset {name!r} already exists; replace it or pick another name", exists=True)  # fmt: skip


def _detail(payload: Any) -> str:  # Any: a JSON answer of any shape
    """The configurator's ``detail`` as one line (FastAPI's error shape), else the payload."""
    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, dict) and "errors" in detail:
        return "; ".join(f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in detail["errors"])  # fmt: skip
    return str(detail)
