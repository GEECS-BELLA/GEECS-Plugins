"""Tests for ``scripts/export_openapi.py`` — the services' committed API documents.

The real services cannot be imported here (each lives in its own env; the
root env has none of them), so the mechanics are tested on a toy app, and
the committed documents are tested as files. The real apps' determinism is
checked where they import: every CI leg's ``--check`` renders its service
twice, the second time in a fresh interpreter under another hash seed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "export_openapi.py"


def _load() -> ModuleType:
    """Import the script by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("export_openapi", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_openapi"] = module
    spec.loader.exec_module(module)
    return module


exporter = _load()

#: A small app in the services' idiom: typed routes, a path parameter, a
#: model nested in a model, the shared operationId generator.
TOY_APP = textwrap.dedent(
    """
    from fastapi import FastAPI
    from pydantic import BaseModel
    from geecs_web_theme.web import operation_id


    class Axis(BaseModel):
        name: str
        unit: str = "mm"


    class StatusOut(BaseModel):
        state: str
        axes: list[Axis]
        flags: dict[str, bool]


    def build(scratch):
        app = FastAPI(title="Toy", version="1.2.3",
                      generate_unique_id_function=operation_id)

        @app.get("/api/status")
        def status() -> StatusOut:
            return StatusOut(state="ok", axes=[], flags={})

        @app.post("/api/run/{uid}/analysis")
        def analyse(uid: str, body: Axis) -> StatusOut:
            return StatusOut(state="ok", axes=[body], flags={})

        return app


    EVENTS = {"status": StatusOut, "axis": Axis}
    """
)


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Register the toy as the only service; return a scratch repo root."""
    source = tmp_path / "toy_service.py"
    source.write_text(TOY_APP)
    spec = importlib.util.spec_from_file_location("toy_service", source)
    assert spec is not None and spec.loader is not None
    toy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy_module)
    monkeypatch.setattr(
        exporter,
        "SERVICES",
        {
            "Toy": exporter.Service(
                toy_module.build,
                lambda: exporter.events_document(toy_module.EVENTS, "Toy events"),
            )
        },
    )
    # The real second render runs a child interpreter that would not see
    # the monkeypatched table; the nondeterminism test replaces this.
    monkeypatch.setattr(exporter, "render_in_child", exporter.render)
    root = tmp_path / "repo"
    (root / "Toy").mkdir(parents=True)
    return root


def _export_and_dump(seed: str, tmp_path: Path) -> str:
    """Render the toy in a fresh interpreter under *seed*; the bundle as JSON."""
    (tmp_path / "toy_service.py").write_text(TOY_APP)
    program = textwrap.dedent(
        f"""
        import importlib.util, json, sys
        sys.path.insert(0, {str(tmp_path)!r})
        spec = importlib.util.spec_from_file_location("x", {str(SCRIPT)!r})
        x = importlib.util.module_from_spec(spec); sys.modules["x"] = x
        spec.loader.exec_module(x)
        import toy_service
        x.SERVICES = {{"Toy": x.Service(toy_service.build,
            lambda: x.events_document(toy_service.EVENTS, "Toy events"))}}
        print(json.dumps(x.render("Toy")))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    ).stdout


# --------------------------------------------------------------- the mechanics


def test_the_export_is_deterministic_across_processes(tmp_path: Path) -> None:
    """Two interpreters under different hash seeds render byte-identical files."""
    assert _export_and_dump("1", tmp_path) == _export_and_dump("2", tmp_path)


def test_a_fresh_export_passes_the_check(toy: Path) -> None:
    assert exporter.run_service("Toy", check=False, root=toy) == 0
    written = sorted(p.name for p in (toy / "Toy" / "api").iterdir())
    assert written == ["events.schema.json", "openapi.json"]
    assert exporter.run_service("Toy", check=True, root=toy) == 0


def test_a_stale_document_fails_the_check(
    toy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A route or model change without a regenerated document fails, with a diff."""
    exporter.run_service("Toy", check=False, root=toy)
    document = toy / "Toy" / "api" / "openapi.json"
    doc = json.loads(document.read_text())
    doc["paths"]["/api/status"]["get"]["summary"] = "an older wording"
    document.write_text(exporter.dump(doc))
    assert exporter.run_service("Toy", check=True, root=toy) == exporter.DRIFT
    err = capsys.readouterr().err
    assert "Toy/api/openapi.json: stale" in err
    assert "an older wording" in err
    assert "python scripts/export_openapi.py" in err


def test_a_missing_document_fails_the_check(toy: Path) -> None:
    exporter.run_service("Toy", check=False, root=toy)
    (toy / "Toy" / "api" / "events.schema.json").unlink()
    assert exporter.run_service("Toy", check=True, root=toy) == exporter.DRIFT


def test_a_document_the_service_no_longer_produces_fails_the_check(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An events document left behind after the stream lost its models is stale."""
    exporter.run_service("Toy", check=False, root=toy)
    service = exporter.SERVICES["Toy"]
    monkeypatch.setitem(
        exporter.SERVICES, "Toy", exporter.Service(service.build, lambda: None)
    )
    assert exporter.run_service("Toy", check=True, root=toy) == exporter.DRIFT
    # Regenerating removes it.
    exporter.run_service("Toy", check=False, root=toy)
    assert not (toy / "Toy" / "api" / "events.schema.json").exists()


def test_a_nondeterministic_export_fails_the_check(
    toy: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two renders that differ fail before the committed files are even read."""
    exporter.run_service("Toy", check=False, root=toy)
    monkeypatch.setattr(
        exporter,
        "render_in_child",
        lambda name: {**exporter.render(name), "openapi.json": "{}\n"},
    )
    assert exporter.run_service("Toy", check=True, root=toy) == exporter.DRIFT
    assert "nondeterministic" in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore:Duplicate Operation ID")
def test_duplicate_operation_ids_refuse_the_export(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two operations under one id would be merged by a client generator."""
    from fastapi import FastAPI

    def build(scratch: Path) -> FastAPI:
        app = FastAPI(generate_unique_id_function=lambda route: "same")

        @app.get("/a")
        def a() -> dict:
            return {}

        @app.get("/b")
        def b() -> dict:
            return {}

        return app

    monkeypatch.setitem(exporter.SERVICES, "Toy", exporter.Service(build, lambda: None))
    with pytest.raises(RuntimeError, match="duplicate operationIds"):
        exporter.render("Toy")


def test_the_events_document_maps_each_event_to_its_frame_schema(toy: Path) -> None:
    from jsonschema import Draft202012Validator

    events = json.loads(exporter.render("Toy")["events.schema.json"])
    Draft202012Validator.check_schema(events)
    assert events["properties"] == {
        "axis": {"$ref": "#/$defs/Axis"},
        "status": {"$ref": "#/$defs/StatusOut"},
    }
    validator = Draft202012Validator(events)
    frames = {"axis": {"name": "x", "unit": "mm"}, "status": {
        "state": "ok", "axes": [], "flags": {}}}  # fmt: skip
    validator.validate(frames)
    assert not validator.is_valid({**frames, "axis": {"unit": "mm"}})


def test_list_prints_the_services_without_importing_them() -> None:
    """``--list`` is what CI and check.sh select the legs with: stdlib only."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--list"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert out == list(exporter.SERVICES)


# ------------------------------------------------------- the committed documents


@pytest.mark.parametrize("name", sorted(exporter.SERVICES))
def test_every_service_has_a_committed_openapi_3_document(name: str) -> None:
    """Each committed document exists, parses as OpenAPI 3.x, and is canonical.

    Existence is asserted, not skipped on: the documents are what the web
    clients are generated from, so a move that drops one must fail CI.
    """
    from fastapi.openapi.models import OpenAPI

    path = exporter.api_dir(name) / exporter.OPENAPI
    assert path.exists(), f"{path} missing — run `python scripts/export_openapi.py`"
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["openapi"].startswith("3.")
    OpenAPI.model_validate(document)
    assert text == exporter.dump(document), "hand-edited: regenerate, never edit"
    ids = exporter.operation_ids(document)
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "path",
    sorted(REPO_ROOT.glob(f"*/{exporter.API_DIR}/{exporter.EVENTS}")),
    ids=lambda p: p.parent.parent.name,
)
def test_every_committed_events_document_is_a_json_schema(path: Path) -> None:
    from jsonschema import Draft202012Validator

    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    Draft202012Validator.check_schema(document)
    assert text == exporter.dump(document), "hand-edited: regenerate, never edit"
    assert set(document["required"]) == set(document["properties"])


def test_no_orphan_documents() -> None:
    """Every committed ``<Package>/api/`` document belongs to an exported service."""
    committed = {
        p.parent.parent.name
        for name in exporter.MANAGED
        for p in REPO_ROOT.glob(f"*/{exporter.API_DIR}/{name}")
    }
    assert committed == set(exporter.SERVICES)
