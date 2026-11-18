"""Which changed documents do not validate: the gate on main, the same check the configs repo's CI runs.

One rule, two callers. The service's timer (and any fast-forward) asks it
of every commit it would apply that it did not make; the configs repo's
validate-on-push workflow runs it (``python -m geecs_configurator.validation``,
with GEECS-Schemas and PyYAML alone installed: this module and ``store.py``
import nothing else, pinned by ``tests/test_propagate.py``) on the paths a
push changed. Only **changed** documents are checked: a document
already invalid on main and untouched does not block (the deployed catalogue
holds some), and a change that leaves one invalid does.

A path is a document when the store lists it under a file kind (the same
listing the editor shows: every suffix, the shared analyzers tree split by
its loader); any other path (a README, a script) is not checked. A document
is checked as its readers read it: a legacy trigger profile through the
converter, a setup's name against its stem.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from geecs_schemas import (
    KIND_STORAGE,
    REFERS_TO,
    SCHEMA_REGISTRY,
    FileStorage,
    references,
)
from geecs_schemas.analysis.document import ANALYZERS_FOLDER
from pydantic import ValidationError

from geecs_configurator.store import DocumentStore, NotFound


def documents(store: DocumentStore) -> dict[str, tuple[str, str | None, str]]:
    """Every document in the tree: its path below the root -> (kind, experiment, id)."""
    out: dict[str, tuple[str, str | None, str]] = {}
    for kind, storage in KIND_STORAGE.items():
        if not isinstance(storage, FileStorage):
            continue
        for experiment in store.experiments() if "<experiment>" in storage.path_pattern() else [None]:  # fmt: skip
            try:
                files = store.files(kind, experiment)
            except NotFound:
                continue
            for doc_id, path in files.items():
                out[path.relative_to(store.root).as_posix()] = (
                    kind,
                    experiment,
                    doc_id,
                )
    return out


def _run_by_stem(known: dict[str, tuple[str, str | None, str]]) -> dict[str, set[str]]:
    """The analyzers tree's documents by file stem: a run loads one by its stem, across every facility folder."""
    out: dict[str, set[str]] = {}
    for path, (kind, _, _) in known.items():
        storage = KIND_STORAGE[kind]
        if isinstance(storage, FileStorage) and storage.root == "analysis" and storage.folder == ANALYZERS_FOLDER:  # fmt: skip
            out.setdefault(PurePosixPath(path).stem, set()).add(path)
    return out


def stem_twins(store: DocumentStore, rel: str) -> list[str]:
    """The other analyzers-tree files with ``rel``'s stem, when ``rel`` (below the root) is one the tree runs by stem."""
    storage = next((s for s in KIND_STORAGE.values() if isinstance(s, FileStorage) and s.root == "analysis" and s.folder == ANALYZERS_FOLDER), None)  # fmt: skip
    folder = (
        storage.path_pattern().rsplit("/", 1)[0].removesuffix("/**")
        if storage
        else None
    )
    if folder is None or not rel.startswith(folder + "/"):
        return []
    return sorted(_run_by_stem(documents(store)).get(PurePosixPath(rel).stem, set()) - {rel})  # fmt: skip


def deletable(kind: str) -> bool:
    """Whether a kind's documents are named files of a folder (deleted from the editor, named by others); never a kind's one file."""
    storage = KIND_STORAGE.get(kind)
    return isinstance(storage, FileStorage) and "<name>" in storage.path_pattern()


def _pattern(kind: str) -> re.Pattern[str] | None:
    """A named kind's path pattern as a regex (``exp`` and ``name`` groups, every suffix); ``None`` for a single-file kind."""
    if not deletable(kind):
        return None
    storage = KIND_STORAGE[kind]
    assert isinstance(storage, FileStorage)
    text = re.escape(storage.path_pattern()).replace(r"\*\*/", "(?:.+/)?").replace("<experiment>", "(?P<exp>[^/]+)")  # fmt: skip
    suffixes = "|".join(re.escape(s) for s in storage.suffixes)
    return re.compile(text.replace(r"<name>\.yaml", f"(?P<name>[^/]+)(?:{suffixes})") + "$")  # fmt: skip


def _same_namespace(a: str, b: str) -> bool:
    """Whether a name of kind ``a`` and one of kind ``b`` are looked up in one folder (recipes and ``.himg`` jobs share one)."""
    return KIND_STORAGE[a].path_pattern() == KIND_STORAGE[b].path_pattern()  # type: ignore[union-attr]


def referrers(store: DocumentStore, kind: str, experiment: str | None, name: str, *, known: dict[str, tuple[str, str | None, str]] | None = None, besides: str = "") -> list[str]:  # fmt: skip
    """The documents that name ``name`` of ``kind`` (``geecs_schemas.references``), their paths below the root.

    A name is looked up in the same experiment (an analysis group's across
    the analyzers tree). A referrer that does not validate names nothing
    this can read, so it is not listed. ``besides``: a path never listed
    (the document itself).
    """
    out = []
    for path, (k, exp, doc_id) in (known if known is not None else documents(store)).items():  # fmt: skip
        if path == besides or not any(_same_namespace(t, kind) for t in REFERS_TO.get(k, ())):  # fmt: skip
            continue
        if "<experiment>" in KIND_STORAGE[kind].path_pattern() and exp != experiment:  # type: ignore[union-attr]  # fmt: skip
            continue
        loaded = store.read(k, exp, doc_id)
        try:
            model = SCHEMA_REGISTRY[k].model_validate(loaded.document)
        except ValidationError:
            continue
        if loaded.valid and any(_same_namespace(rk, kind) and rn == name for rk, rn in references(model)):  # fmt: skip
            out.append(path)
    return sorted(out)


def still_named(path: str, names: list[str]) -> str:
    """Why a delete is refused, in words (the names are listed beside it)."""
    one = len(names) == 1
    return f"{path} is still named by {len(names)} document{'' if one else 's'}; change {'it' if one else 'them'} first"  # fmt: skip


def _deleted_problems(store: DocumentStore, known: dict[str, tuple[str, str | None, str]], deleted: Iterable[str]) -> list[str]:  # fmt: skip
    """A deleted document another still names: the reader of that name would break."""
    out = []
    for rel in deleted:
        for kind in KIND_STORAGE:
            pattern = _pattern(kind)
            hit = pattern.match(rel) if pattern else None
            if hit is None or not any(_same_namespace(t, kind) for ts in REFERS_TO.values() for t in ts):  # fmt: skip
                continue
            experiment, name = hit.groupdict().get("exp"), hit.group("name")
            # the name may still answer (another suffix, another folder of the analyzers tree)
            if any(_same_namespace(k, kind) and exp == experiment and PurePosixPath(p).stem == name for p, (k, exp, _) in known.items()):  # fmt: skip
                continue
            names = referrers(store, kind, experiment, name, known=known)
            if names:
                out.append(f"{rel}: deleted, but {', '.join(names)} still name{'s' if len(names) == 1 else ''} it")  # fmt: skip
            break
    return out


def problems(
    root: Path, paths: Iterable[str], deleted: Iterable[str] = ()
) -> list[str]:
    """``"<path>: <first error>"`` for each changed path that is a document and does not validate.

    A changed document of the analyzers tree whose file stem another file
    there also has is a problem too: the loader refuses a tree with two
    (every run by name fails), whichever box or pull request brought it.
    And a ``deleted`` path that held a document another still names (a
    group its recipe, a preset its trigger profile): the reader of that
    name would break, however the delete was made.
    """
    store = DocumentStore(root)
    known = documents(store)
    stems = _run_by_stem(known)
    out = _deleted_problems(store, known, deleted)
    for path in paths:
        if path in known:
            kind, experiment, doc_id = known[path]
            loaded = store.read(kind, experiment, doc_id)
            twins = sorted(stems.get(PurePosixPath(path).stem, set()) - {path})
            if not loaded.valid:
                out.append(
                    f"{path}: {loaded.errors[0].msg if loaded.errors else 'invalid'}"
                )
            elif twins and path in stems.get(PurePosixPath(path).stem, set()):
                out.append(f"{path}: its stem is also {twins[0]}'s; diagnostic stems are unique across the analyzers tree")  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    """``geecs-configurator-validate --root <configs checkout> <path>...``: exit 1 if a changed document is invalid."""
    parser = argparse.ArgumentParser(
        prog="geecs-configurator-validate", description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="the configs repo's checkout"
    )
    parser.add_argument(
        "paths", nargs="*", help="the paths the push changed, relative to the root"
    )
    parser.add_argument(
        "--deleted",
        nargs="*",
        default=[],
        help="the paths the push deleted: refused while another document still names one",
    )
    args = parser.parse_args(argv)
    found = problems(args.root, args.paths, args.deleted)
    for line in found:
        print(f"invalid: {line}")
    print(f"{len(args.paths)} changed and {len(args.deleted)} deleted path(s), {len(found)} problem(s)")  # fmt: skip
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
