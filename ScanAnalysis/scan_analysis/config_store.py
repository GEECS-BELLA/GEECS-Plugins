"""The analysis-config store: list, read, validate and write diagnostics and groups.

Plain Python over a ``scan_analysis_configs`` tree — no web framework,
no Qt — so the config editor's web router, a notebook, MCP or a script
all edit configs the same way.  Replaces the Qt editor's
Qt ``ConfigFileGUI`` (deleted in 1.21.0).

Rules the store enforces:

- Documents are the GEECS-Schemas models (``AnalysisDiagnostic`` v2,
  ``AnalysisGroup``); a document that does not validate is never written.
- Writes are atomic (temp file + rename in the same directory) and
  optimistic: a save carries the etag the file had when it was read, and a
  changed file refuses the save (:class:`ConflictError`) — two editors
  cannot silently clobber each other.  Creating requires the file not to
  exist; diagnostic IDs are unique across every namespace, as the loader
  requires.
- The on-disk form is canonical (:func:`geecs_schemas.analysis.canonical_document`
  as YAML, field order preserved), so a save never reformats more than it
  changes.
- Only the configs tree is touched.  Nothing here knows about scan
  folders (the repo's scan-folder invariant is irrelevant by construction,
  but the store still creates at most the namespace folder, never parents
  above the tree).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import yaml
from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisGroup,
    canonical_document,
)
from pydantic import BaseModel, ValidationError

__all__ = [
    "ConfigStore",
    "ConflictError",
    "DocumentInvalid",
    "Entry",
    "Loaded",
    "NotFound",
    "Report",
    "Saved",
]

DocumentKind = Literal["analyzer", "group"]

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class ConfigStoreError(Exception):
    """Base of the store's errors (each maps to one HTTP status in the router)."""


class NotFound(ConfigStoreError):
    """No document with that ID (or namespace) exists."""


class ConflictError(ConfigStoreError):
    """The file changed since it was read, or a create would overwrite."""


class DocumentInvalid(ConfigStoreError):
    """The document does not validate; ``errors`` carries the pydantic locations."""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__(
            "; ".join(f"{e['loc']}: {e['msg']}" for e in errors) or "invalid"
        )
        self.errors = errors


@dataclass(frozen=True)
class Entry:
    """One document in a listing."""

    id: str
    namespace: str
    kind: DocumentKind
    valid: bool
    error: Optional[str] = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """The listing row as the API serialises it."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "kind": self.kind,
            "valid": self.valid,
            "error": self.error,
            **self.summary,
        }


@dataclass(frozen=True)
class Loaded:
    """A document read from disk, with the etag a save must present."""

    id: str
    namespace: str
    kind: DocumentKind
    document: dict[str, Any]
    etag: str
    yaml: str
    valid: bool
    errors: list[dict[str, str]]

    def to_json(self) -> dict[str, Any]:
        """The loaded document as the API serialises it."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "kind": self.kind,
            "document": self.document,
            "etag": self.etag,
            "yaml": self.yaml,
            "valid": self.valid,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class Report:
    """The result of validating a document without writing it."""

    ok: bool
    errors: list[dict[str, str]]
    canonical: Optional[dict[str, Any]]
    yaml: Optional[str]

    def to_json(self) -> dict[str, Any]:
        """The report as the API serialises it."""
        return {
            "ok": self.ok,
            "errors": self.errors,
            "canonical": self.canonical,
            "yaml": self.yaml,
        }


@dataclass(frozen=True)
class Saved:
    """The outcome of a write."""

    id: str
    namespace: str
    kind: DocumentKind
    etag: str
    yaml: str
    created: bool

    def to_json(self) -> dict[str, Any]:
        """The save outcome as the API serialises it."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "kind": self.kind,
            "etag": self.etag,
            "yaml": self.yaml,
            "created": self.created,
        }


_MODELS: dict[str, type[BaseModel]] = {
    "analyzer": AnalysisDiagnostic,
    "group": AnalysisGroup,
}
_FOLDERS: dict[str, str] = {"analyzer": "analyzers", "group": "groups"}


def _one_line(exc: BaseException) -> str:
    """A parse error on one line (PyYAML's span several: problem, mark, context)."""
    text = " ".join(str(exc).split())
    return text[:240] if text else type(exc).__name__


def _errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {"loc": ".".join(str(part) for part in err["loc"]), "msg": err["msg"]}
        for err in exc.errors()
    ]


def dump_yaml(document: Mapping[str, Any]) -> str:
    """The one YAML serialisation every writer uses (field order kept)."""
    return yaml.safe_dump(dict(document), sort_keys=False, default_flow_style=False)


class ConfigStore:
    """List, read, validate and write the documents of one configs tree.

    Parameters
    ----------
    root : Path
        The ``scan_analysis_configs`` root (the parent of ``analyzers/``
        and ``groups/``).
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ paths

    def folder(self, kind: DocumentKind) -> Path:
        """The ``analyzers/`` or ``groups/`` folder of the tree."""
        return self.root / _FOLDERS[kind]

    def namespaces(self, kind: DocumentKind) -> list[str]:
        """The namespaces a document can be saved into: the top-level folders.

        Files deeper down are listed and readable (``_files`` walks the whole
        tree like the runtime loaders) but their nested folder is not offered
        as a save target — ``save`` takes one path segment.
        """
        folder = self.folder(kind)
        if not folder.is_dir():
            return []
        return sorted(p.name for p in folder.iterdir() if p.is_dir())

    def _files(self, kind: DocumentKind) -> list[Path]:
        """Every YAML file under the kind's folder, at any depth.

        The same walk the runtime loaders do (``rglob`` in
        ``image_analysis.config.loader`` and ScanAnalysis'
        ``analysis_group_loader``), so what the store lists, checks for
        duplicate ids and cross-references is exactly what a run will load.
        """
        folder = self.folder(kind)
        if not folder.is_dir():
            return []
        return sorted(
            p
            for p in folder.rglob("*")
            if p.suffix in (".yaml", ".yml") and p.is_file()
        )

    def _ns(self, kind: DocumentKind, path: Path) -> str:
        """The namespace of a file: its folder relative to the kind's root."""
        rel = path.parent.relative_to(self.folder(kind)).as_posix()
        return "" if rel == "." else rel

    def path_for(self, kind: DocumentKind, doc_id: str) -> Path:
        """The existing file for ``doc_id`` (unique stem across namespaces)."""
        matches = [p for p in self._files(kind) if p.stem == doc_id]
        if not matches:
            raise NotFound(f"no {kind} {doc_id!r} under {self.folder(kind)}")
        if len(matches) > 1:
            raise ConflictError(
                f"{kind} id {doc_id!r} exists in several namespaces: "
                f"{sorted(self._ns(kind, p) for p in matches)}"
            )
        return matches[0]

    @staticmethod
    def _etag(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    @staticmethod
    def _check_name(value: str, what: str) -> None:
        if not _ID.match(value or ""):
            raise DocumentInvalid(
                [{"loc": what, "msg": f"{what} must match {_ID.pattern}"}]
            )

    # ---------------------------------------------------------------- listing

    def list(self, kind: DocumentKind) -> list[Entry]:
        """Every document of ``kind``, valid or not (invalid ones carry the error)."""
        entries: list[Entry] = []
        model = _MODELS[kind]
        for path in self._files(kind):
            ns = self._ns(kind, path)
            try:
                doc = model.model_validate(self._read_raw(path))
            except ValidationError as exc:
                entries.append(
                    Entry(path.stem, ns, kind, False, _errors(exc)[0]["msg"])
                )
                continue
            except Exception as exc:  # noqa: BLE001 — a broken YAML is an entry, not a crash
                entries.append(Entry(path.stem, ns, kind, False, _one_line(exc)))
                continue
            entries.append(Entry(path.stem, ns, kind, True, None, self._summary(doc)))
        return entries

    @staticmethod
    def _summary(doc: BaseModel) -> dict[str, Any]:
        if isinstance(doc, AnalysisDiagnostic):
            return {
                "name": doc.name,
                "analyzer_kind": doc.analyzer.kind,
                "image_type": doc.image_kind,
                "device": doc.scan.device or doc.name,
                "output_name": doc.effective_output_name,
            }
        return {"name": doc.name, "count": len(doc.analyzers)}

    def known_ids(self) -> list[str]:
        """Diagnostic IDs a group may reference."""
        return sorted(p.stem for p in self._files("analyzer"))

    # ---------------------------------------------------------------- reading

    @staticmethod
    def _read_raw(path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise DocumentInvalid([{"loc": "", "msg": "document is not a mapping"}])
        return data

    def read(self, kind: DocumentKind, doc_id: str) -> Loaded:
        """Read one document as it is on disk, with its etag.

        A file that is not even a YAML mapping still loads — as an invalid
        document with an empty ``document`` and the parse error in
        ``errors`` — so the editor can show the file as it is instead of
        failing the request.
        """
        path = self.path_for(kind, doc_id)
        try:
            raw = self._read_raw(path)
        except (yaml.YAMLError, DocumentInvalid, UnicodeDecodeError) as exc:
            raw, valid, errors = {}, False, [{"loc": "", "msg": _one_line(exc)}]
        else:
            report = self.validate(kind, raw)
            valid, errors = report.ok, report.errors
        return Loaded(
            id=path.stem,
            namespace=self._ns(kind, path),
            kind=kind,
            document=raw,
            etag=self._etag(path),
            yaml=path.read_text(encoding="utf-8", errors="replace"),
            valid=valid,
            errors=errors,
        )

    # ------------------------------------------------------------- validating

    def validate(self, kind: DocumentKind, document: Mapping[str, Any]) -> Report:
        """Validate without writing; the report carries the canonical form."""
        try:
            model = _MODELS[kind].model_validate(dict(document))
        except ValidationError as exc:
            return Report(False, _errors(exc), None, None)
        errors = self._cross_checks(kind, model)
        if errors:
            return Report(False, errors, None, None)
        canonical = canonical_document(model)
        return Report(True, [], canonical, dump_yaml(canonical))

    def _cross_checks(
        self, kind: DocumentKind, model: BaseModel
    ) -> list[dict[str, str]]:
        """Checks that need the tree, not just the document."""
        if kind == "group":
            known = set(self.known_ids())
            return [
                {"loc": f"analyzers.{i}.ref", "msg": f"unknown diagnostic {ref.ref!r}"}
                for i, ref in enumerate(model.analyzers)
                if ref.ref not in known
            ]
        return []

    # ---------------------------------------------------------------- writing

    def save(
        self,
        kind: DocumentKind,
        namespace: str,
        doc_id: str,
        document: Mapping[str, Any],
        *,
        etag: Optional[str],
    ) -> Saved:
        """Validate and write one document atomically.

        ``etag`` is the value returned by :meth:`read` (or an earlier save);
        ``None`` means "create" and refuses to overwrite an existing file.
        A stale etag raises :class:`ConflictError`.
        """
        self._check_name(namespace, "namespace")
        self._check_name(doc_id, "id")
        report = self.validate(kind, document)
        if not report.ok:
            raise DocumentInvalid(report.errors)
        assert report.yaml is not None
        target = self.folder(kind) / namespace / f"{doc_id}.yaml"
        # the same stem anywhere else in the tree is a different document
        elsewhere = [
            p
            for p in self._files(kind)
            if p.stem == doc_id and self._ns(kind, p) != namespace
        ]
        if elsewhere:
            raise ConflictError(
                f"{kind} id {doc_id!r} already exists in namespace "
                f"{self._ns(kind, elsewhere[0])!r}; ids are unique across the tree"
            )
        exists = target.exists() or target.with_suffix(".yml").exists()
        if exists:
            current = target if target.exists() else target.with_suffix(".yml")
            if etag is None:
                raise ConflictError(f"{kind} {doc_id!r} already exists; load it first")
            if self._etag(current) != etag:
                raise ConflictError(
                    f"{kind} {doc_id!r} changed on disk since it was loaded; reload"
                )
            target = current
        elif etag is not None:
            raise ConflictError(f"{kind} {doc_id!r} no longer exists on disk")
        target.parent.mkdir(exist_ok=True)  # the namespace folder only, never parents
        self._atomic_write(target, report.yaml)
        return Saved(
            doc_id, namespace, kind, self._etag(target), report.yaml, not exists
        )

    def delete(self, kind: DocumentKind, doc_id: str, *, etag: str) -> None:
        """Delete one document; the etag must match the file on disk."""
        path = self.path_for(kind, doc_id)
        if self._etag(path) != etag:
            raise ConflictError(
                f"{kind} {doc_id!r} changed on disk since it was loaded"
            )
        path.unlink()

    @staticmethod
    def _atomic_write(target: Path, text: str) -> None:
        fd, tmp = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------ git status

    def pending_changes(self) -> Optional[list[str]]:
        """``git status --porcelain`` of the tree, or ``None`` when it is not in a git checkout."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.root), "status", "--porcelain", "--", "."],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return [line for line in result.stdout.splitlines() if line.strip()]

    # ---------------------------------------------------------------- schemas

    @staticmethod
    def schema(kind: DocumentKind) -> dict[str, Any]:
        """The JSON Schema the editor renders its form from."""
        return _MODELS[kind].model_json_schema()
