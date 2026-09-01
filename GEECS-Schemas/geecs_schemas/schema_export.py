"""Publish the ScanRequest contract as a JSON Schema artifact (#727).

The queueserver funnel (``geecs_scan_request_plan``) takes one ``request``
parameter — the JSON form of :class:`~geecs_schemas.ScanRequest`. This
module exports that contract in the lingua franca every generic client
understands: ``ScanRequest.model_json_schema()`` serialized to a committed
artifact, ``docs/geecs_schemas/scan_request.schema.json``, so any
JSON-Schema-aware consumer (OSPREY's plan panel, agent approval prompts,
future web UIs) can render a full nested ScanRequest form — optimization
sub-model included — without importing Python.

Same discipline as :mod:`geecs_schemas.docgen` (the Markdown twin): a
generator, a committed artifact, and a no-drift test
(``tests/test_schema_export.py``) that fails CI when the artifact falls
out of step with the schemas. Regenerate after an intentional schema
change with::

    python -m geecs_schemas.schema_export

(or ``GEECS-Schemas/tests/generate_scan_request_schema.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from geecs_schemas.scan_request import ScanRequest

# Path to the committed artifact, relative to the repo root. Kept here so
# the generator, the regenerator script, and the no-drift test all agree
# on it. Living under docs/ puts the file on the published mkdocs site,
# so clients outside this repo can fetch it by URL.
SCHEMA_ARTIFACT = Path("docs/geecs_schemas/scan_request.schema.json")


def scan_request_json_schema() -> dict:
    """The ScanRequest contract as a JSON Schema dict.

    Returns
    -------
    dict
        ``ScanRequest.model_json_schema()`` — the full nested vocabulary
        (axes, acquisition, actions, trigger profile, optimization) with
        every field description, ready for form generation or validation.
    """
    return ScanRequest.model_json_schema()


def render_artifact() -> str:
    """Render the committed artifact text (stable, newline-terminated).

    Returns
    -------
    str
        Pretty-printed JSON — byte-for-byte what the no-drift test
        expects at :data:`SCHEMA_ARTIFACT`.
    """
    return json.dumps(scan_request_json_schema(), indent=2) + "\n"


def _repo_root() -> Path:
    """Return the repository root (three levels above this module).

    Returns
    -------
    pathlib.Path
        ``<repo>/`` — the parent of ``GEECS-Schemas/``.
    """
    # schema_export.py → geecs_schemas/ → GEECS-Schemas/ → <repo root>
    return Path(__file__).resolve().parents[2]


def write_artifact(path: Optional[Path] = None) -> Path:
    """Write the schema artifact to disk and return where it was written.

    Parameters
    ----------
    path : pathlib.Path, optional
        Destination file; defaults to ``<repo>/`` + :data:`SCHEMA_ARTIFACT`.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    destination = path if path is not None else _repo_root() / SCHEMA_ARTIFACT
    destination.write_text(render_artifact(), encoding="utf-8")
    return destination


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point: regenerate the committed schema artifact.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination file "
            "(default: <repo>/docs/geecs_schemas/scan_request.schema.json)."
        ),
    )
    args = parser.parse_args(argv)
    written = write_artifact(args.output)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
