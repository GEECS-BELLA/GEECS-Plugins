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
import sys
from collections.abc import Iterable
from pathlib import Path

from geecs_schemas import KIND_STORAGE, FileStorage

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


def problems(root: Path, paths: Iterable[str]) -> list[str]:
    """``"<path>: <first error>"`` for each changed path that is a document and does not validate."""
    store = DocumentStore(root)
    known = documents(store)
    out = []
    for path in paths:
        if path in known:
            kind, experiment, doc_id = known[path]
            loaded = store.read(kind, experiment, doc_id)
            if not loaded.valid:
                out.append(
                    f"{path}: {loaded.errors[0].msg if loaded.errors else 'invalid'}"
                )
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
    args = parser.parse_args(argv)
    found = problems(args.root, args.paths)
    for line in found:
        print(f"invalid: {line}")
    print(f"{len(args.paths)} changed path(s), {len(found)} invalid document(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
