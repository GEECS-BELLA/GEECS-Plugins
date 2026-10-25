#!/usr/bin/env python3
"""Export each web service's OpenAPI document and its SSE event schemas.

Why this exists
---------------
The browser apps (``geecs-web``, ``Planning/epics_estate.md`` §2.1 and §5)
generate their typed clients from these committed documents, pinned to one
GEECS-Plugins commit. A route or a response model that changes without its
document is a client that silently lies, so ``--check`` fails CI when the
committed document differs from a fresh export.

What is written, per service, under ``<Package>/api/`` (beside the code
that defines it, so a change to the document selects that package's CI leg
like any other file in it):

* ``openapi.json`` — ``app.openapi()`` of the app the service's own entry
  point builds, with every optional route family switched on (the logbook's
  write routes, the portal's config editor, every panel kind): the
  document is the union a client may meet; a deployment without a flag
  answers 404 on that family's routes.
* ``events.schema.json`` — where the service streams Server-Sent Events: a
  JSON Schema (draft 2020-12) whose top-level ``properties`` map each SSE
  ``event:`` name to the schema of that frame's ``data:``. A map, not a
  payload: generated as one TypeScript interface, it types a handler per
  event.

Both are ``json.dumps(..., indent=2, sort_keys=True)`` plus a newline, and
every operationId comes from the method and the path
(``geecs_web_theme.web.operation_id``), never the route function's name.

Environments
------------
Each service is imported in **its own** Poetry environment — the env its
CI leg installs (``.github/workflows/unit-tests.yml``), which is the only
one guaranteed to import it with the right extras. So the script has two
modes:

* with ``--service <Package>`` it renders that one service in the current
  interpreter (CI runs it this way, from inside the leg's env);
* without, it is the driver: for each service it runs itself under
  ``poetry run`` in that package's directory, the way ``scripts/check.sh``
  runs each package's suite. The driver needs only the standard library.

``--check`` writes nothing: it renders the service twice, once here and
once in a child process under a different ``PYTHONHASHSEED`` (a set's
iteration order is the classic leak), fails on any difference between the
two (nondeterministic) and then on any difference from the committed files
(stale), naming the regenerate command.

Usage
-----
    python scripts/export_openapi.py                        # regenerate every service
    python scripts/export_openapi.py --check                # what CI checks, every service
    python scripts/export_openapi.py GeecsScanner           # regenerate one (driver)
    poetry run python ../scripts/export_openapi.py --service GeecsScanner --check
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The documents' directory inside each service's package directory.
API_DIR = "api"
OPENAPI = "openapi.json"
EVENTS = "events.schema.json"
#: Every file name this script owns in ``<Package>/api/``; one the service
#: no longer produces is stale too.
MANAGED = (OPENAPI, EVENTS)

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
REGENERATE = "python scripts/export_openapi.py"
#: Exit codes: a check that ran and found the documents stale (or the export
#: nondeterministic), and a service this interpreter cannot import (any
#: other failure is a traceback, exit 1).
DRIFT = 1
UNIMPORTABLE = 2

# Any: a FastAPI app, and the JSON documents (free-form by definition).
AppBuilder = Callable[[Path], Any]
EventsBuilder = Callable[[], Optional[dict[str, Any]]]


@dataclass(frozen=True)
class Service:
    """One web service: how to build its app, and its SSE events document.

    Attributes
    ----------
    build : callable
        ``build(scratch) -> FastAPI``. ``scratch`` is an empty temporary
        directory for anything the factory insists on touching (the
        logbook's SQLite file); nothing in it reaches the document.
    events : callable
        ``events() -> dict | None`` — the SSE document, or ``None`` when the
        service streams no typed events (yet).
    """

    build: AppBuilder
    events: EventsBuilder


# --------------------------------------------------------------- the services
#
# Each builder uses the service's own entry point with empty inputs: the
# document needs the route table, never behaviour, so nothing here is a
# fake of anything (the MCP contract tests' fixtures are the behavioural
# copies; they live in GEECS-MCP/tests/conftest.py, importable from no
# other tree).


def _scanner_app(scratch: Path) -> Any:
    """The scanner exactly as ``geecs-scanner --demo`` builds it."""
    from geecs_scanner.__main__ import build_service
    from geecs_scanner.web import create_app

    service = build_service(
        experiment="", identity="openapi-export", demo=True, demo_period=0.0
    )
    return create_app(service)


def _scanner_events() -> dict[str, Any]:
    from geecs_scanner.web.events import EVENT_MODELS

    return events_document(EVENT_MODELS, "GEECS Scanner events")


def _portal_app(scratch: Path) -> Any:
    """The portal with its config editor mounted (as the deployed unit runs it).

    The editor needs a configs tree to exist; an empty one is enough for its
    routes (the factory warns that it holds no diagnostics — expected here).
    The portal mounts the editor only when ``scan_analysis`` imports, and
    otherwise says so in a log line and carries on: a portal env installed
    without the ``analysis`` extra would render a document silently missing
    ``/configs``, so that is refused here as the env problem it is.
    """
    from geecs_portal.app import create_app

    app = create_app(_NoCatalog(), processing_config_dir=scratch, config_editor=True)
    if not app.state.portal.config_editor_enabled:
        raise ImportError(
            "the portal mounted no config editor: this env lacks scan_analysis "
            "(install the portal with `poetry install --with dev --extras analysis`)"
        )
    return app


class _NoCatalog:
    """A ``ScanCatalog`` that is never asked anything: the factory only stores it."""


def _logbook_app(scratch: Path) -> Any:
    """The logbook with a store, so its write routes exist."""
    from geecs_logbook.app import create_app

    return create_app(
        "openapi-export", base_directory=scratch, notes_db=scratch / "notes.db"
    )


def _panels_app(scratch: Path) -> Any:
    """The panels with every kind's live state given, so every kind's routes exist.

    ``create_app`` only stores the live-state objects, so a bare ``object()``
    stands in for each (the panels' own tests do the same for cameras); the
    PV bridge's ``/api`` routes are always mounted. The factory passes
    ``operation_id`` itself.
    """
    from geecs_panels.app import create_app

    return create_app(
        "openapi-export",
        [],
        object(),
        scopes=object(),
        motors=object(),
        supplies=object(),
    )


def _panels_events() -> Optional[dict[str, Any]]:
    """The panels' event schemas, from ``python -m geecs_panels.contracts``.

    The hook for the contracts lane: that module (the device-stream event
    schema, frozen in the estate's lane S) prints one JSON Schema document
    on stdout. Until it exists the panels publish no events document, and
    the day it lands ``--check`` fails until the document is regenerated
    and committed — which is the point.
    """
    import importlib.util

    if importlib.util.find_spec("geecs_panels.contracts") is None:
        return None
    out = subprocess.run(
        [sys.executable, "-m", "geecs_panels.contracts"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def _no_events() -> None:
    return None


#: The exported services, by package directory (= CI leg name).
SERVICES: dict[str, Service] = {
    "GeecsScanner": Service(_scanner_app, _scanner_events),
    "GEECS-DataPortal": Service(_portal_app, _no_events),
    "GeecsLogbook": Service(_logbook_app, _no_events),
    "GeecsPanels": Service(_panels_app, _panels_events),
}


# ------------------------------------------------------------------ rendering


def dump(document: Any) -> str:
    """The one serialization: sorted keys, two-space indent, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def events_document(models: dict[str, Any], title: str) -> dict[str, Any]:
    """Build the events document from an ``event name → Pydantic model`` table.

    Parameters
    ----------
    models : dict
        Each SSE event name and the model its frame's ``data`` serializes.
    title : str
        The document's title.

    Returns
    -------
    dict
        A draft 2020-12 JSON Schema: ``properties`` maps every event name to
        its frame's schema (a ``$ref`` into ``$defs``), all required.
    """
    from pydantic.json_schema import models_json_schema

    keys, schema = models_json_schema(
        [(model, "serialization") for model in models.values()],
        ref_template="#/$defs/{model}",
    )
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "title": title,
        "description": (
            "Each property is one Server-Sent-Events `event:` name; its schema "
            "is the JSON object that frame's `data:` carries. A map of the "
            "stream's frames, not a payload."
        ),
        "type": "object",
        "properties": {
            name: keys[(model, "serialization")] for name, model in models.items()
        },
        "required": sorted(models),
        "additionalProperties": False,
        "$defs": schema.get("$defs", {}),
    }


def operation_ids(document: dict[str, Any]) -> list[str]:
    """Every operationId in an OpenAPI document, in path order."""
    return [
        op["operationId"]
        for _, item in sorted(document.get("paths", {}).items())
        for _, op in sorted(item.items())
        if isinstance(op, dict) and "operationId" in op
    ]


def render(name: str) -> dict[str, str]:
    """Render one service in this interpreter: ``file name → text``.

    Raises
    ------
    RuntimeError
        When two operations share an operationId (a client generator would
        merge or drop one).
    """
    service = SERVICES[name]
    with tempfile.TemporaryDirectory(prefix="openapi-export-") as scratch:
        document = service.build(Path(scratch)).openapi()
    ids = operation_ids(document)
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise RuntimeError(f"{name}: duplicate operationIds {duplicates}")
    files = {OPENAPI: dump(document)}
    events = service.events()
    if events is not None:
        files[EVENTS] = dump(events)
    return files


def render_in_child(name: str, seed: str = "12345") -> dict[str, str]:
    """Render *name* again in a fresh interpreter under another hash seed."""
    env = {**os.environ, "PYTHONHASHSEED": seed}
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--service", name, "--print"],
        capture_output=True,
        text=True,
        env=env,
    )
    if child.returncode:
        raise RuntimeError(f"{name}: the second render failed:\n{child.stderr}")
    return json.loads(child.stdout)


def api_dir(name: str, root: Path = REPO_ROOT) -> Path:
    """Where *name*'s documents are committed."""
    return root / name / API_DIR


def write(files: dict[str, str], directory: Path) -> list[Path]:
    """Write *files* into *directory*; remove a managed file no longer produced."""
    directory.mkdir(exist_ok=True)
    written = []
    for file_name in MANAGED:
        path = directory / file_name
        if file_name in files:
            path.write_text(files[file_name], encoding="utf-8")
            written.append(path)
        elif path.exists():
            path.unlink()
    return written


def problems(files: dict[str, str], directory: Path) -> list[str]:
    """How the committed files in *directory* differ from *files* (empty: fresh)."""
    found = []
    for file_name in MANAGED:
        path = directory / file_name
        shown = f"{directory.parent.name}/{directory.name}/{file_name}"
        want = files.get(file_name)
        have = path.read_text(encoding="utf-8") if path.exists() else None
        if want == have:
            continue
        if want is None:
            found.append(f"{shown}: committed, but the service no longer produces it")
        elif have is None:
            found.append(f"{shown}: missing")
        else:
            diff = difflib.unified_diff(
                have.splitlines(), want.splitlines(), "committed", "fresh", lineterm=""
            )
            head = "\n".join(list(diff)[:40])
            found.append(f"{shown}: stale\n{head}")
    return found


def run_service(name: str, *, check: bool, root: Path = REPO_ROOT) -> int:
    """Export or check one service in this interpreter; the process exit code."""
    files = render(name)
    directory = api_dir(name, root)
    if not check:
        for path in write(files, directory):
            print(f"wrote {path.relative_to(root)}")
        return 0
    again = render_in_child(name)
    if again != files:
        differing = sorted(
            set(files) ^ set(again)
            | {f for f in set(files) & set(again) if files[f] != again[f]}
        )
        print(
            f"{name}: the export is nondeterministic — two renders differ in "
            f"{differing}; fix the source of the difference, not the document",
            file=sys.stderr,
        )
        return DRIFT
    found = problems(files, directory)
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(
            f"{name}: the committed API documents are out of date — run "
            f"`{REGENERATE}` (or `{REGENERATE} {name}`) and commit the result",
            file=sys.stderr,
        )
        return DRIFT
    print(f"{name}: API documents up to date")
    return 0


# --------------------------------------------------------------------- driver


def drive(names: list[str], *, check: bool) -> int:
    """Run every named service in its own package env; the worst exit code."""
    status = 0
    script = str(Path(__file__).resolve())
    # Never the caller's venv: `poetry run` must pick the package's own.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    for name in names:
        cmd = ["poetry", "run", "python", script, "--service", name]
        if check:
            cmd.append("--check")
        print(f"== {name}", flush=True)
        code = subprocess.run(cmd, cwd=REPO_ROOT / name, env=env).returncode
        if code == UNIMPORTABLE:
            print(
                f"   ({name}'s env cannot import it — missing or stale: "
                "`poetry install --with dev` there, see /env-doctor)",
                file=sys.stderr,
            )
        status = max(status, code)
    return status


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "packages", nargs="*", help="services to drive (default: every service)"
    )
    parser.add_argument(
        "--service",
        choices=sorted(SERVICES),
        help="render this one service in the current interpreter",
    )
    parser.add_argument(
        "--check", action="store_true", help="compare, write nothing; exit 1 on drift"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the exported services and exit"
    )
    parser.add_argument("--print", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(SERVICES))
        return 0

    if args.service:
        try:
            if args.print:
                print(json.dumps(render(args.service)))
                return 0
            return run_service(args.service, check=args.check)
        except ImportError:
            traceback.print_exc()
            return UNIMPORTABLE
    unknown = [p for p in args.packages if p not in SERVICES]
    if unknown:
        parser.error(f"not an exported service: {unknown} (know: {sorted(SERVICES)})")
    return drive(args.packages or list(SERVICES), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
