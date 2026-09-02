"""Publish schema contracts as JSON Schema artifacts (#727, Phase 2b-i).

Generic clients (OSPREY's plan panel, agent approval prompts, future web
UIs) render forms from JSON Schema, never from Python.  This module keeps a
**registry** of the models published that way — :data:`EXPORTED_SCHEMAS`,
artifact name → model — and renders one committed artifact per entry under
``docs/geecs_schemas/<name>.schema.json`` (on the published mkdocs site, so
clients outside this repo fetch them by URL).  ``scan_request`` is the first
entry: the queueserver funnel's one ``request`` parameter.  The named plans
of Phase 2b add theirs as one line each.

Same discipline as :mod:`geecs_schemas.docgen` (the Markdown twin): a
generator, committed artifacts, and a no-drift test
(``tests/test_schema_export.py``) that iterates the registry and fails CI
when any artifact falls out of step with its model.  Regenerate after an
intentional schema change with::

    python -m geecs_schemas.schema_export

(or ``GEECS-Schemas/tests/generate_schema_artifacts.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from geecs_schemas.scan_request import ScanRequest

#: The published JSON Schema contracts: artifact name → model.  Each entry
#: renders to ``docs/geecs_schemas/<name>.schema.json``.  Add a line here
#: and regenerate; the no-drift guard covers it from then on.
EXPORTED_SCHEMAS: dict[str, type[BaseModel]] = {
    "scan_request": ScanRequest,
}

#: Where the artifacts live, relative to the repo root.
ARTIFACT_DIR = Path("docs/geecs_schemas")

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def artifact_path(name: str) -> Path:
    """Return the committed artifact path for registry entry *name*.

    Parameters
    ----------
    name : str
        A key of :data:`EXPORTED_SCHEMAS`.

    Returns
    -------
    pathlib.Path
        ``docs/geecs_schemas/<name>.schema.json``, relative to the repo root.
    """
    return ARTIFACT_DIR / f"{name}.schema.json"


#: The ScanRequest artifact path — kept as a named constant because the
#: worker's plan annotation and OSPREY's profile point at it.
SCHEMA_ARTIFACT = artifact_path("scan_request")


def json_schema(model: type[BaseModel]) -> dict:
    """Return *model*'s contract as a JSON Schema dict.

    Parameters
    ----------
    model : type[BaseModel]
        The pydantic model to export.

    Returns
    -------
    dict
        ``model.model_json_schema()`` — the full nested vocabulary with
        every field description, ready for form generation or validation —
        plus an explicit ``$schema`` dialect marker.
    """
    # Pydantic emits 2020-12 vocabulary (prefixItems, $ref siblings) but
    # no dialect marker; external validators default to older drafts and
    # would silently ignore the tuple shapes without it (#730 review).
    return {"$schema": JSON_SCHEMA_DIALECT, **model.model_json_schema()}


def scan_request_json_schema() -> dict:
    """The ScanRequest contract as a JSON Schema dict (registry entry ``scan_request``).

    Returns
    -------
    dict
        As :func:`json_schema` for :class:`~geecs_schemas.ScanRequest`.
    """
    return json_schema(EXPORTED_SCHEMAS["scan_request"])


def render_artifact(name: str = "scan_request") -> str:
    """Render the committed artifact text for registry entry *name*.

    Parameters
    ----------
    name : str
        A key of :data:`EXPORTED_SCHEMAS` (default: the ScanRequest entry).

    Returns
    -------
    str
        Pretty-printed JSON, newline-terminated — byte-for-byte what the
        no-drift test expects at :func:`artifact_path`.
    """
    return json.dumps(json_schema(EXPORTED_SCHEMAS[name]), indent=2) + "\n"


def _repo_root() -> Path:
    """Return the repository root (three levels above this module).

    Returns
    -------
    pathlib.Path
        ``<repo>/`` — the parent of ``GEECS-Schemas/``.
    """
    # schema_export.py → geecs_schemas/ → GEECS-Schemas/ → <repo root>
    return Path(__file__).resolve().parents[2]


def write_artifact(path: Optional[Path] = None, *, name: str = "scan_request") -> Path:
    """Write one registry entry's artifact to disk and return where.

    Parameters
    ----------
    path : pathlib.Path, optional
        Destination file; defaults to ``<repo>/`` + :func:`artifact_path`.
    name : str
        The registry entry to render (default: the ScanRequest entry).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    destination = path if path is not None else _repo_root() / artifact_path(name)
    destination.write_text(render_artifact(name), encoding="utf-8")
    return destination


def write_artifacts(root: Optional[Path] = None) -> list[Path]:
    """Write every registry entry's artifact under *root*.

    Parameters
    ----------
    root : pathlib.Path, optional
        Repository root to write under (default: this checkout).

    Returns
    -------
    list of pathlib.Path
        The paths written, in registry order.
    """
    base = root if root is not None else _repo_root()
    return [
        write_artifact(base / artifact_path(name), name=name)
        for name in EXPORTED_SCHEMAS
    ]


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point: regenerate the committed schema artifacts.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root to write under (default: this checkout).",
    )
    args = parser.parse_args(argv)
    for written in write_artifacts(args.root):
        print(f"wrote {written}")


if __name__ == "__main__":
    main()
