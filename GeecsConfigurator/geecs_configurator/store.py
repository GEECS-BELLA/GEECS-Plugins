"""The document store: every file kind of the configs repository, read and checked, never written.

Generalized from ScanAnalysis's analysis-config store (copied, not shared:
that one is deleted when the analysis kinds move here) to every kind
``geecs_schemas.storage.KIND_STORAGE`` keeps as a file: the path from
``FileStorage.path_pattern()``, every suffix its readers accept, and a
shared folder (the analyzers tree) split by the kind its ``dispatch`` loader
reads a file as. Family files are package data, shown read-only as ``code``.

Rules:

- **Read only.** Nothing here writes, renames or deletes, and nothing
  creates a directory: a missing folder is no documents. (The commit path,
  with its own rules, is the next slice's.)
- A document's id is its path below the kind's folder (``laser_off.yaml``,
  ``Undulator/UC_TopView.yaml``); only an id the listing holds is ever read,
  so a request cannot name a path outside the tree.
- The etag (mtime and size) is what a later save will present; validation
  errors carry pydantic's locations, which the form places at its fields.
- :func:`diff` is the service's statement of what a draft changes and the
  strictest tier among the changes, the rule ``form.js`` draws live in the
  page; the two are pinned to agree by ``tests/test_form_parity.py``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from geecs_schemas import (
    KIND_STORAGE,
    SCHEMA_REGISTRY,
    TIERS,
    FileStorage,
    Setup,
)
from geecs_schemas.epics.family import FAMILIES_FOLDER, FAMILIES_PACKAGE
from geecs_schemas.storage import NAMED, STORAGE_ROOT_PATHS
from pydantic import BaseModel, ValidationError

FAMILY = "family_file"
#: A JSON value or a JSON Schema fragment, as the API carries it.
Json = Any


class NotFound(LookupError):
    """No such kind, experiment or document."""


class Error(BaseModel):
    """One validation error at a document location (pydantic's ``loc``)."""

    loc: list[str | int]
    msg: str


class Entry(BaseModel):
    """One document in a listing."""

    id: str
    valid: bool
    error: str | None = None


class Loaded(BaseModel):
    """A document as it is on disk, with the etag a save will present."""

    kind: str
    experiment: str | None
    id: str
    path: str
    document: Json
    etag: str
    yaml: str
    valid: bool
    errors: list[Error]
    tier: str | None
    read_only: bool


class Report(BaseModel):
    """A draft validated without writing it."""

    ok: bool
    errors: list[Error]


class Change(BaseModel):
    """One difference between the stored document and a draft, at a field's path."""

    path: list[str | int]
    op: str
    before: Json = None
    after: Json = None
    tier: str | None


class Diff(BaseModel):
    """What a draft changes, and the strictest tier among the changes (what saving it would do)."""

    changes: list[Change]
    tier: str | None
    etag: str


def dump_yaml(document: Json) -> str:
    """The one YAML rendering (field order kept)."""
    return yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)  # fmt: skip


def _errors(exc: ValidationError) -> list[Error]:
    return [Error(loc=list(e["loc"]), msg=e["msg"]) for e in exc.errors()]


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:240] or type(exc).__name__


class DocumentStore:
    """List, read, validate and diff the documents of one configs repository checkout.

    Parameters
    ----------
    root : Path
        The configs repository's root (the parent of ``scanner_configs/``).
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def experiments(self) -> list[str]:
        """The experiments with a folder under the experiment root."""
        base = self.root / STORAGE_ROOT_PATHS["experiment"].split("/<experiment>")[0]
        return (
            sorted(p.name for p in base.iterdir() if p.is_dir())
            if base.is_dir()
            else []
        )

    # ---------------------------------------------------------------- paths

    def _storage(self, kind: str) -> FileStorage:
        storage = KIND_STORAGE.get(kind)
        if not isinstance(storage, FileStorage):
            raise NotFound(f"{kind!r} is not a file kind")
        return storage

    def folder(self, kind: str, experiment: str | None) -> Path:
        """The kind's folder: the part of its path pattern before the file name."""
        pattern = self._storage(kind).path_pattern()
        if "<experiment>" in pattern:
            if experiment not in self.experiments():
                raise NotFound(f"no experiment {experiment!r}")
            pattern = pattern.replace("<experiment>", experiment)
        return self.root / pattern.rsplit("/", 1)[0].removesuffix("/**")

    def _files(self, kind: str, experiment: str | None) -> dict[str, Path]:
        storage, base = self._storage(kind), self.folder(kind, experiment)
        if not base.is_dir():
            return {}
        if storage.pattern != NAMED:
            path = base / storage.pattern
            return {path.name: path} if path.is_file() else {}
        walk = base.rglob if storage.nested else base.glob
        paths = sorted(
            p for s in storage.suffixes for p in walk(f"*{s}") if p.is_file()
        )
        if storage.dispatch:
            paths = [
                p for p in paths if self._declared_kind(storage.dispatch, p) == kind
            ]
        return {p.relative_to(base).as_posix(): p for p in paths}

    @staticmethod
    def _declared_kind(dispatch: str, path: Path) -> str | None:
        """The kind a shared folder's loader reads a file as (its model, valid or not)."""
        module, _, name = dispatch.rpartition(".")
        loader: Callable[[Mapping[str, Any]], BaseModel] = getattr(importlib.import_module(module), name)  # fmt: skip
        by_model = {m.__name__: k for k, m in SCHEMA_REGISTRY.items()}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return by_model.get(
                type(loader(raw if isinstance(raw, dict) else {})).__name__
            )
        except ValidationError as exc:
            return by_model.get(exc.title)
        except (yaml.YAMLError, UnicodeDecodeError):
            return None

    # -------------------------------------------------------------- reading

    def list(self, kind: str, experiment: str | None) -> list[Entry]:
        """Every document of a kind, valid or not (an invalid one carries its first error)."""
        if kind == FAMILY:
            return [Entry(id=name, valid=True) for name in sorted(self._families())]
        out = []
        for doc_id, path in self._files(kind, experiment).items():
            try:
                errors = self.validate(kind, self._raw(path), stem=path.stem).errors
            except Exception as exc:  # noqa: BLE001 - a broken file is a row, not a crash
                errors = [Error(loc=[], msg=_one_line(exc))]
            out.append(Entry(id=doc_id, valid=not errors, error=errors[0].msg if errors else None))  # fmt: skip
        return out

    def read(self, kind: str, experiment: str | None, doc_id: str) -> Loaded:
        """One document as it is on disk; a file that does not parse loads empty with the error."""
        if kind == FAMILY:
            families = self._families()
            if doc_id not in families:
                raise NotFound(f"no family file {doc_id!r}")
            text = families[doc_id].read_text(encoding="utf-8")
            raw, path, etag = json.loads(text), f"GEECS-Schemas package data: {FAMILIES_FOLDER}/{doc_id}", "package"  # fmt: skip
        else:
            files = self._files(kind, experiment)
            if doc_id not in files:
                raise NotFound(f"no {kind} {doc_id!r}")
            file = files[doc_id]
            text, path = file.read_text(encoding="utf-8", errors="replace"), file.relative_to(self.root).as_posix()  # fmt: skip
            stat = file.stat()
            etag = f"{stat.st_mtime_ns}-{stat.st_size}"
        try:
            raw = raw if kind == FAMILY else self._raw(files[doc_id])
            errors = self.validate(kind, raw, stem=Path(doc_id).stem).errors
        except (yaml.YAMLError, ValueError, UnicodeDecodeError) as exc:
            raw, errors = {}, [Error(loc=[], msg=_one_line(exc))]
        return Loaded(
            kind=kind, experiment=experiment, id=doc_id, path=path, document=raw, etag=etag,
            yaml=text if kind != FAMILY else dump_yaml(raw), valid=not errors, errors=errors,
            tier="code" if kind == FAMILY else SCHEMA_REGISTRY[kind].tier, read_only=kind == FAMILY,
        )  # fmt: skip

    @staticmethod
    def _raw(path: Path) -> dict[str, Json]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("the document is not a mapping")
        return data

    @staticmethod
    def _families() -> dict[str, Any]:  # Any: an importlib Traversable
        folder = resources.files(FAMILIES_PACKAGE).joinpath(FAMILIES_FOLDER)
        return {e.name.removesuffix(".json"): e for e in folder.iterdir() if e.name.endswith(".json")}  # fmt: skip

    # ------------------------------------------------------------ checking

    @staticmethod
    def validate(kind: str, document: Json, *, stem: str | None = None) -> Report:
        """Validate a draft as its kind; a setup's ``name`` must also be its file stem."""
        if kind not in SCHEMA_REGISTRY:
            raise NotFound(f"no kind {kind!r}")
        try:
            model = SCHEMA_REGISTRY[kind].model_validate(document)
        except ValidationError as exc:
            return Report(ok=False, errors=_errors(exc))
        if isinstance(model, Setup) and stem is not None and model.name != stem:
            msg = f"a setup's name is its file stem: {stem!r}"
            return Report(ok=False, errors=[Error(loc=["name"], msg=msg)])
        return Report(ok=True, errors=[])

    def diff(self, kind: str, experiment: str | None, doc_id: str, draft: Json) -> Diff:
        """What ``draft`` changes in the stored document, with each change's tier."""
        loaded = self.read(kind, experiment, doc_id)
        schema = SCHEMA_REGISTRY[kind].model_json_schema()
        changes = diff_documents(loaded.document, draft, schema)
        return Diff(changes=changes, tier=strictest(c.tier for c in changes), etag=loaded.etag)  # fmt: skip


# ------------------------------------------------------------------ the diff
# The same rules as form.js's diff (tests/test_form_parity.py): $ref and an
# optional resolve, a discriminated union follows the variant both sides
# hold, a field's tier is the strictest declared on its path, and a list
# compares index by index, so a reorder changes every index it moved.


def strictest(tiers: Any) -> str | None:  # Any: an iterable of tier names or None
    """The most restricted tier named, or ``None``."""
    named = [t for t in tiers if t]
    return max(named, key=TIERS.index) if named else None


def _deref(s: Json, root: Json) -> Json:
    if not isinstance(s, dict):
        return {}
    if "$ref" in s:
        target = _deref(root.get("$defs", {}).get(s["$ref"].split("/")[-1], {}), root)
        s = {**target, **{k: v for k, v in s.items() if k != "$ref"}}
    return s


def _unwrap(s: Json, root: Json) -> Json:
    s = _deref(s, root)
    union = s.get("anyOf") or s.get("oneOf")
    if union:
        rest = [m for m in union if _deref(m, root).get("type") != "null"]
        if len(rest) == 1 and len(rest) < len(union):
            outer = {
                k: s[k] for k in ("title", "description", "default", "tier") if k in s
            }
            return {**_deref(rest[0], root), **outer}
    return s


def _variant(s: Json, root: Json, before: Json, after: Json) -> Json | None:
    """The union's variant both sides hold; ``None`` when they differ (one change)."""
    members = [_deref(m, root) for m in s.get("oneOf") or s.get("anyOf") or []]
    if not members or not all("properties" in m for m in members):
        return s
    prop = (s.get("discriminator") or {}).get("propertyName") or next(
        (
            k
            for k in members[0]["properties"]
            if all(m["properties"].get(k, {}).get("const") is not None for m in members)
        ),  # fmt: skip
        None,
    )
    if prop is None or not isinstance(before, dict) or not isinstance(after, dict) or before.get(prop) != after.get(prop):  # fmt: skip
        return None
    return next((m for m in members if m["properties"][prop].get("const") == after.get(prop)), members[0])  # fmt: skip


def diff_documents(before: Json, after: Json, schema: Json) -> list[Change]:
    """Every difference at a field's path, each with the strictest tier on that path."""
    changes: list[Change] = []
    missing = object()

    def walk(
        s: Json, b: Json, a: Json, path: list[str | int], tier: str | None
    ) -> None:
        s = _unwrap(s, schema)
        tier = strictest([tier, s.get("tier")])
        if b == a and isinstance(b, bool) == isinstance(a, bool):
            return
        variant = _variant(s, schema, b, a)
        op = "add" if b is missing else "remove" if a is missing else "change"
        if variant is None:
            changes.append(Change(path=path, op=op, before=None if b is missing else b, after=None if a is missing else a, tier=tier))  # fmt: skip
            return
        props, extra = (
            variant.get("properties", {}),
            variant.get("additionalProperties"),
        )
        if isinstance(b, dict) and isinstance(a, dict):
            for k in [*b, *(k for k in a if k not in b)]:
                walk(props.get(k) or (extra if isinstance(extra, dict) else {}), b.get(k, missing), a.get(k, missing), [*path, k], tier)  # fmt: skip
        elif isinstance(b, list) and isinstance(a, list):
            for i in range(max(len(b), len(a))):
                walk(variant.get("items", {}), b[i] if i < len(b) else missing, a[i] if i < len(a) else missing, [*path, i], tier)  # fmt: skip
        else:
            changes.append(Change(path=path, op=op, before=None if b is missing else b, after=None if a is missing else a, tier=tier))  # fmt: skip

    walk(schema, before, after, [], None)
    return changes
