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
from geecs_schemas.convert import SchemaConversionError, convert_shot_control
from geecs_schemas.epics.family import FAMILIES_FOLDER, FAMILIES_PACKAGE
from geecs_schemas.storage import NAMED, STORAGE_ROOT_PATHS
from pydantic import BaseModel, ValidationError

FAMILY = "family_file"
#: How a trigger profile in the legacy dialect (no ``schema_version``) is shown: the reader converts it.
LEGACY = "legacy dialect, converted by the reader"
#: A shared folder's file whose kind cannot be read: listed under every kind of that folder.
_UNKNOWN = "?"
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
    note: str | None = None


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
    note: str | None = None


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


def _odd_keys(value: Json) -> bool:
    """Whether a parsed YAML value holds a mapping key that is not a string (a YAML 1.1 boolean or a number)."""
    if isinstance(value, dict):
        return any(not isinstance(k, str) or _odd_keys(v) for k, v in value.items())
    return isinstance(value, list) and any(_odd_keys(v) for v in value)


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
            paths = [base / storage.pattern]
        else:
            walk = base.rglob if storage.nested else base.glob
            paths = sorted(p for s in storage.suffixes for p in walk(f"*{s}"))
        # a symlink out of the configs root is never listed, so never read
        inside = self.root.resolve()
        paths = [p for p in paths if p.is_file() and p.resolve().is_relative_to(inside)]
        if storage.dispatch:
            paths = [p for p in paths if self._declared_kind(storage.dispatch, p) in (kind, _UNKNOWN)]  # fmt: skip
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
            return by_model.get(exc.title, _UNKNOWN)
        except (
            yaml.YAMLError,
            OSError,
            ValueError,
        ):  # unreadable or unparseable: shown under every kind of the folder
            return _UNKNOWN

    # -------------------------------------------------------------- reading

    def list(self, kind: str, experiment: str | None) -> list[Entry]:
        """Every document of a kind, valid or not (an invalid or unreadable one carries its first error)."""
        if kind == FAMILY:
            return [Entry(id=name, valid=True) for name in sorted(self._families())]
        out = []
        for doc_id, path in self._files(kind, experiment).items():
            _, errors, note = self._load(kind, path)
            out.append(Entry(id=doc_id, valid=not errors, error=errors[0].msg if errors else None, note=note))  # fmt: skip
        return out

    def read(self, kind: str, experiment: str | None, doc_id: str) -> Loaded:
        """One document as it is on disk; a file that does not read or parse loads empty with the error."""
        note = None
        if kind == FAMILY:
            families = self._families()
            if doc_id not in families:
                raise NotFound(f"no family file {doc_id!r}")
            raw = json.loads(families[doc_id].read_text(encoding="utf-8"))
            path, etag, text = f"GEECS-Schemas package data: {FAMILIES_FOLDER}/{doc_id}", "package", dump_yaml(raw)  # fmt: skip
            errors = self.validate(kind, raw).errors
        else:
            files = self._files(kind, experiment)
            if doc_id not in files:
                raise NotFound(f"no {kind} {doc_id!r}")
            file = files[doc_id]
            path = file.relative_to(self.root).as_posix()
            try:
                text, stat = file.read_text(encoding="utf-8", errors="replace"), file.stat()  # fmt: skip
                etag = f"{stat.st_mtime_ns}-{stat.st_size}"
            except OSError as exc:
                text, etag = "", ""
                raw, errors = {}, [Error(loc=[], msg=f"unreadable: {_one_line(exc)}")]
            else:
                raw, errors, note = self._load(kind, file)
        return Loaded(
            kind=kind, experiment=experiment, id=doc_id, path=path, document=raw, etag=etag,
            yaml=text, valid=not errors, errors=errors, note=note,
            tier="code" if kind == FAMILY else SCHEMA_REGISTRY[kind].tier, read_only=kind == FAMILY,
        )  # fmt: skip

    def _load(self, kind: str, path: Path) -> tuple[dict[str, Json], list[Error], str | None]:  # fmt: skip
        """A file as the document its readers see, its errors, and a note (a legacy dialect, converted)."""
        try:
            raw = self._raw(path)
        except (yaml.YAMLError, OSError, ValueError) as exc:
            return {}, [Error(loc=[], msg=_one_line(exc))], None
        note = None
        if kind == "trigger_profile" and "schema_version" not in raw:
            # the reader's rule (the scanner's resolver): no schema_version is the legacy dialect, converted
            try:
                profile = convert_shot_control(raw, name=path.stem)
            except (SchemaConversionError, ValueError) as exc:
                return (
                    raw,
                    [Error(loc=[], msg=f"{LEGACY} failed: {_one_line(exc)}")],
                    LEGACY,
                )
            if profile is None:
                return (
                    raw,
                    [Error(loc=[], msg="the legacy profile names no device")],
                    LEGACY,
                )
            raw, note = profile.model_dump(mode="json"), LEGACY
        if _odd_keys(raw):
            # YAML 1.1 reads a bare OFF: / ON: key as a boolean, which JSON would serve as "false": the document is
            # served as the model reads it (its own key normalization), so the page, validate and diff agree on an
            # unedited file
            try:
                raw = SCHEMA_REGISTRY[kind].model_validate(raw).model_dump(mode="json", exclude_unset=True)  # fmt: skip
            except ValidationError as exc:
                return raw, _errors(exc), note
        return raw, self.validate(kind, raw, stem=path.stem).errors, note

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
# optional resolve; a union of objects follows the variant both sides hold
# (by its discriminator, else structurally); equality is type-strict (true is
# not 1); a change's tier is the strictest declared on its path, and a whole
# subtree added, removed or switched takes the strictest declared anywhere in
# its schema; an unknown tier ranks strictest; a list compares index by index,
# so a reorder changes every index it moved.

_SCALARS = ("string", "number", "integer", "boolean", "null")


def _rank(tier: str | None) -> int:
    return -1 if not tier else TIERS.index(tier) if tier in TIERS else len(TIERS)


def strictest(tiers: Any) -> str | None:  # Any: an iterable of tier names or None
    """The most restricted tier named (an unknown one ranks strictest), or ``None``."""
    named = [t for t in tiers if t]
    return max(named, key=_rank) if named else None


def same(a: Json, b: Json) -> bool:
    """JSON equality, type-strict all the way down: ``true`` is not ``1``; key order is no difference."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False
    return type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float))) if a == b else False  # fmt: skip


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
        if len(rest) < len(union):
            outer = {k: s[k] for k in ("title", "description", "default", "tier") if k in s}  # fmt: skip
            if len(rest) == 1:
                return {**_deref(rest[0], root), **outer}
            # Optional[A | B]: the union without its null, as form.js unwraps it
            return {**{k: v for k, v in s.items() if k != "oneOf"}, "anyOf": rest}
    return s


def _extra(s: Json) -> Json:
    """The schema of an object's undeclared keys (additionalProperties, one patternProperties, or {})."""
    if isinstance(s.get("additionalProperties"), dict):
        return s["additionalProperties"]
    patterns = s.get("patternProperties") or {}
    return next(iter(patterns.values())) if len(patterns) == 1 else {}


def _variants(s: Json, root: Json) -> tuple[str | None, list[tuple[Json, Json]]] | None:
    """A union of objects: (discriminator or None, [(tag, variant)]); ``None`` when a member is not an object."""
    members = [_deref(m, root) for m in s.get("oneOf") or s.get("anyOf") or []]
    if not members or not all(m.get("type") == "object" or "properties" in m for m in members):  # fmt: skip
        return None
    prop = (s.get("discriminator") or {}).get("propertyName") or next(
        (
            k
            for k in members[0].get("properties", {})
            if all(
                _deref(m.get("properties", {}).get(k), root).get("const") is not None
                for m in members
            )
        ),  # fmt: skip
        None,
    )

    def tag(i: int, m: Json) -> Json:
        if prop:
            return _deref(m["properties"][prop], root).get("const")
        return m.get("title") or f"option {i + 1}"

    return prop, [(tag(i, m), m) for i, m in enumerate(members)]


def _fits(v: Json, value: Json, root: Json) -> bool:
    props = v.get("properties", {})
    if not isinstance(value, dict) or any(
        k not in value for k in v.get("required", [])
    ):
        return False
    if v.get("additionalProperties") is False and any(k not in props for k in value):
        return False
    for k, p in props.items():
        p = _deref(p, root)
        if k in value and ("const" in p and not same(value[k], p["const"]) or "enum" in p and value[k] not in p["enum"]):  # fmt: skip
            return False
    return True


def _pick(union: tuple[str | None, list[tuple[Json, Json]]], value: Json, root: Json) -> tuple[Json, Json]:  # fmt: skip
    prop, members = union
    if prop:
        found = [
            m
            for m in members
            if isinstance(value, dict) and same(m[0], value.get(prop))
        ]
    else:
        found = [m for m in members if _fits(m[1], value, root)]
    return (found or members)[0]


def _tier_in(s: Json, root: Json, seen: tuple[str, ...] = ()) -> str | None:
    """The strictest tier declared anywhere in a schema (every variant, item and entry)."""
    if not isinstance(s, dict):
        return None
    if "$ref" in s:
        if s["$ref"] in seen:
            return s.get("tier")
        seen = (*seen, s["$ref"])
    d = _deref(s, root)
    kids = [*d.get("anyOf", []), *d.get("oneOf", []), *([d["items"]] if isinstance(d.get("items"), dict) else []),
            *([d["additionalProperties"]] if isinstance(d.get("additionalProperties"), dict) else []),
            *d.get("properties", {}).values(), *d.get("patternProperties", {}).values()]  # fmt: skip
    return strictest([d.get("tier"), *(_tier_in(k, root, seen) for k in kids)])


def _shared_variant(s: Json, b: Json, a: Json, root: Json) -> Json | None:
    """The schema both sides are under: for a union of objects the variant both hold, ``None`` when they differ."""
    union = _variants(s, root)
    if union is None:
        return s
    vb = _pick(union, b, root) if isinstance(b, dict) else None
    va = _pick(union, a, root) if isinstance(a, dict) else None
    return va[1] if vb is not None and va is not None and same(vb[0], va[0]) else None


def diff_documents(before: Json, after: Json, schema: Json) -> list[Change]:
    """Every difference at a field's path, each with its tier (see the rules above)."""
    changes: list[Change] = []
    missing = object()

    def push(
        s: Json, b: Json, a: Json, path: list[str | int], tier: str | None
    ) -> None:
        if any(isinstance(x, (dict, list)) for x in (b, a)):
            tier = strictest([tier, _tier_in(s, schema)])
        op = "add" if b is missing else "remove" if a is missing else "change"
        changes.append(Change(path=path, op=op, before=None if b is missing else b, after=None if a is missing else a, tier=tier))  # fmt: skip

    def walk(
        s: Json, b: Json, a: Json, path: list[str | int], tier: str | None
    ) -> None:
        s = _unwrap(s, schema)
        tier = strictest([tier, s.get("tier")])
        if b is not missing and a is not missing and same(b, a):
            return
        under = _shared_variant(s, b, a, schema)
        if under is None:
            push(s, b, a, path, tier)
            return
        s = under
        if isinstance(b, dict) and isinstance(a, dict):
            props, extra = s.get("properties", {}), _extra(s)
            for k in [*b, *(k for k in a if k not in b)]:
                walk(props.get(k) or extra, b.get(k, missing), a.get(k, missing), [*path, k], tier)  # fmt: skip
        elif isinstance(b, list) and isinstance(a, list):
            for i in range(max(len(b), len(a))):
                walk(s.get("items", {}), b[i] if i < len(b) else missing, a[i] if i < len(a) else missing, [*path, i], tier)  # fmt: skip
        else:
            push(s, b, a, path, tier)

    walk(schema, before, after, [], None)
    return changes
