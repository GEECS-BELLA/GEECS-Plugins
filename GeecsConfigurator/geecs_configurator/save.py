"""Saving a document: the base check, the tier's action, and what is written.

A save is a whole document over the version the editor was served (its
etag). Under the repository's lock it fetches and fast-forwards the live
checkout, then:

- refuses a blank operator (422), a read-only document (405), and a
  document with no etag (422: a file that did not read or parse, so the
  editor never showed it and there is nothing to save over);
- answers **409** with the current document and the draft's diff against
  it when the etag is not the current file's (the base moved), and again
  when a concurrent push changed the file while the save was pushed: the
  operator re-applies their edit, never an automatic merge;
- refuses (**428**) a trigger profile in the legacy dialect unless the save
  confirms the conversion (``convert_legacy``): it rewrites the whole file
  in the current schema, listed as its own change (path ``[]``, op
  ``convert``) at the strictest tier declared anywhere in the kind's
  schema, so it is a change even when no field moved;
- validates the draft (422 with the errors) and diffs it against the
  document **as served**;
- **follows the served form.** Where the file as stored and its served
  form validate differently (an unquoted YAML date in a text field is a
  date in the file, invalid, and text as served, valid), the save takes the
  served answer: the served text is what the editor showed and what is
  validated, and it is written quoted, so the file then reads back as what
  was validated (checked before any commit). The answer says so in
  ``notes``;
- writes a non-finite float as ``.inf``/``.nan``, never ``null`` (served as
  ``"Infinity"``/``"NaN"``, turned back where the stored document held one);
- acts by the strictest tier among the changes: ``operator`` commits to
  main (the operator the git author, "Name via configurator"; the service
  the committer); ``maintainer`` pushes a branch and opens a pull request
  through :class:`PullRequests` (the live checkout never sees the branch);
  any other tier is not saved from the editor (403).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

import yaml
from geecs_schemas import SCHEMA_REGISTRY
from pydantic import BaseModel

from geecs_configurator.repository import BaseMoved, ConfigsRepository, GitError, Identity, author_name  # fmt: skip
from geecs_configurator.store import LEGACY, SERVE_CONFIG, Change, DocumentStore, Json, Loaded, diff_documents, dump_yaml, restore_nonfinite, same, strictest, tier_in  # fmt: skip

#: The served form's answer is followed where the file as stored validates differently.
FOLLOWS_SERVED = "the file as stored and its served form validate differently (a YAML value read as a date or a number where the schema wants text, say); the save follows the served form and writes it quoted"  # fmt: skip
CONVERTED = "the legacy-dialect file is rewritten whole in the current schema"


class SaveRequest(BaseModel):
    """A whole document over the version the editor was served."""

    document: dict[str, Any]  # Any: validated by the kind's model, not here
    etag: str
    operator: str
    #: The operator confirms rewriting a legacy-dialect file in the current schema.
    convert_legacy: bool = False


class Saved(BaseModel):
    """What a save did."""

    model_config = SERVE_CONFIG

    outcome: Literal["committed", "proposed", "nothing to do"]
    tier: str | None
    changes: list[Change]
    commit: str | None = None
    branch: str | None = None
    pull_request: str | None = None
    etag: str
    notes: list[str] = []


class Refused(Exception):
    """A save the service will not make: an HTTP status and a JSON-able detail."""

    def __init__(self, status: int, detail: Json):
        super().__init__(detail)
        self.status, self.detail = status, detail


class PullRequest(BaseModel):
    """A proposed change: where to review it, and whether a pull request was opened."""

    opened: bool
    url: str | None


class PullRequests(Protocol):
    """Opens a pull request for a pushed branch (the GitHub App's; a fake in tests)."""

    def open(self, branch: str, title: str, body: str) -> PullRequest:
        """Open a pull request from ``branch`` into main."""
        ...


class CompareLinkOnly:
    """No pull requests (a deploy key pushes but cannot open one): the compare link, when the repo is known."""

    def __init__(self, github_repo: str = "", base: str = "main"):
        self.github_repo, self.base = github_repo, base

    def open(self, branch: str, title: str, body: str) -> PullRequest:
        """The branch's compare link (the maintainer opens the pull request from it)."""
        url = f"https://github.com/{self.github_repo}/compare/{self.base}...{branch}?expand=1" if self.github_repo else None  # fmt: skip
        return PullRequest(opened=False, url=url)


def _branch_name(kind: str, doc_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", PurePosixPath(doc_id).stem).strip("-.") or "document"  # fmt: skip
    return f"configurator/{kind}/{stem}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S%f}"


def _summary(changes: list[Change]) -> str:
    lines = [f"- {c.op} {'/'.join(map(str, c.path)) or '(the whole file)'} [{c.tier or 'no tier'}]" for c in changes[:40]]  # fmt: skip
    return "\n".join(lines + ([f"- … {len(changes) - 40} more"] if len(changes) > 40 else []))  # fmt: skip


def _written(draft: Json) -> bytes:
    """The YAML a draft is written as; refuses one that would not read back as itself."""
    text = dump_yaml(draft)
    if not same(yaml.safe_load(text), draft):
        raise Refused(500, "the document does not read back as itself from YAML; nothing was saved")  # fmt: skip
    return text.encode("utf-8")


class _Save:
    """One save's state; :meth:`run` is :func:`save`."""

    def __init__(self, store: DocumentStore, repo: ConfigsRepository, kind: str, experiment: str | None, doc_id: str, request: SaveRequest):  # fmt: skip
        self.store, self.repo, self.request = store, repo, request
        self.kind, self.experiment, self.doc_id = kind, experiment, doc_id
        self.schema = SCHEMA_REGISTRY[kind].model_json_schema()

    def conflict(self, reason: str, draft: Json) -> Refused:
        """The 409: the document as it now is, and what the draft changes in it."""
        current = self.store.read(self.kind, self.experiment, self.doc_id)
        served = self.store.served(current)
        changes = diff_documents(served, restore_nonfinite(draft, served), self.schema)
        return Refused(409, {
            "reason": reason,
            "current": json.loads(current.model_dump_json()),
            "diff": [json.loads(c.model_dump_json()) for c in changes],
        })  # fmt: skip

    def run(self, prs: PullRequests, email: str) -> Saved:
        """The whole save, under the repository's lock."""
        try:
            author = Identity(author_name(self.request.operator), email)
        except ValueError as exc:
            raise Refused(422, str(exc)) from exc
        with self.repo.lock:
            try:
                base = self.repo.sync()
            except GitError as exc:
                raise Refused(502, f"the configs repo could not be brought current, so the base cannot be checked: {exc}") from exc  # fmt: skip
            loaded = self._checked()
            draft, changes, notes = self._draft(loaded)
            if not changes:
                return Saved(outcome="nothing to do", tier=None, changes=[], etag=loaded.etag, notes=notes)  # fmt: skip
            tier = strictest(c.tier for c in changes)
            data = _written(draft)
            message = f"{self.kind} {loaded.path}: saved by {self.request.operator.strip()} via configurator\n\n{_summary(changes)}\n"  # fmt: skip
            if tier == "operator":
                sha = self._commit(loaded.path, data, author, message, base, draft)
                etag = self.store.read(self.kind, self.experiment, self.doc_id).etag
                return Saved(outcome="committed", tier=tier, changes=changes, commit=sha, etag=etag, notes=notes)  # fmt: skip
            if tier == "maintainer":
                return self._propose(prs, loaded, data, author, message, base, changes, notes)  # fmt: skip
            raise Refused(403, f"a {tier}-tier change is not saved from the editor")

    def _checked(self) -> Loaded:
        """The stored document, once the save may go over it: writable, readable, the served base, confirmed."""
        loaded = self.store.read(self.kind, self.experiment, self.doc_id)
        if loaded.read_only:
            raise Refused(405, f"{self.kind} is read-only here ({loaded.tier})")
        if not loaded.etag:
            raise Refused(422, "the stored document did not read or parse, so there is nothing to save over; fix the file by hand")  # fmt: skip
        if self.request.etag != loaded.etag:
            raise self.conflict("the document changed since it was served", self.request.document)  # fmt: skip
        if loaded.note == LEGACY and not self.request.convert_legacy:
            raise Refused(428, "the file is the legacy dialect: saving rewrites it whole in the current schema; confirm with convert_legacy")  # fmt: skip
        return loaded

    def _draft(self, loaded: Loaded) -> tuple[Json, list[Change], list[str]]:
        """The draft as it will be written, validated; its changes against the served form; the notes."""
        served = self.store.served(loaded)
        draft = restore_nonfinite(self.request.document, served)
        stem = PurePosixPath(self.doc_id).stem
        report = self.store.validate(self.kind, draft, stem=stem)
        if not report.ok:
            raise Refused(422, {"errors": [e.model_dump() for e in report.errors]})
        changes = diff_documents(served, draft, self.schema)
        notes = []
        if loaded.note == LEGACY:
            changes.insert(0, Change(path=[], op="convert", tier=tier_in(self.schema, self.schema)))  # fmt: skip
            notes.append(CONVERTED)
        elif loaded.valid != self.store.validate(self.kind, served, stem=stem).ok:
            notes.append(FOLLOWS_SERVED)
        return draft, changes, notes

    def _propose(self, prs: PullRequests, loaded: Loaded, data: bytes, author: Identity, message: str, base: str, changes: list[Change], notes: list[str]) -> Saved:  # fmt: skip
        """The maintainer tier: a branch on origin and a pull request; the live checkout never sees it."""
        branch = _branch_name(self.kind, self.doc_id)
        try:
            sha = self.repo.push_branch(loaded.path, data, author=author, message=message, base=base, branch=branch)  # fmt: skip
        except GitError as exc:
            raise Refused(502, f"the branch was not pushed; nothing changed: {exc}") from exc  # fmt: skip
        body = f"Proposed by {self.request.operator.strip()} from the configurator (maintainer tier).\n\n{_summary(changes)}\n"  # fmt: skip
        pr = prs.open(branch, f"{self.kind}: {loaded.path}", body)
        if not pr.opened:
            notes.append("no pull request was opened (no credential for it): the maintainer opens one from the branch")  # fmt: skip
        return Saved(outcome="proposed", tier="maintainer", changes=changes, commit=sha, branch=branch, pull_request=pr.url, etag=loaded.etag, notes=notes)  # fmt: skip

    def _commit(self, rel: str, data: bytes, author: Identity, message: str, base: str, draft: Json) -> str | None:  # fmt: skip
        """The operator-tier commit; a concurrent push that changed this file is the 409."""

        def moved(_old: str, _new: str) -> None:
            raise BaseMoved("the document changed on main while the save was pushed")

        try:
            return self.repo.commit_file(rel, data, author=author, message=message, base=base, on_moved=moved)  # fmt: skip
        except BaseMoved as exc:
            raise self.conflict(str(exc), draft) from exc
        except GitError as exc:
            raise Refused(502, f"the push failed; nothing changed: {exc}") from exc


def save(
    store: DocumentStore,
    repo: ConfigsRepository,
    prs: PullRequests,
    kind: str,
    experiment: str | None,
    doc_id: str,
    request: SaveRequest,
    *,
    email: str,
) -> Saved:
    """Save one document as its tier says (the module docstring); :class:`Refused` when it will not."""
    return _Save(store, repo, kind, experiment, doc_id, request).run(prs, email)
