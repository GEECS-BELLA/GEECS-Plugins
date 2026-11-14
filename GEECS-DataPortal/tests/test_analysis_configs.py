"""The analysis documents as this portal checks them: schema, validate, and the previews of an unsaved recipe.

The configurator's dry run of a recipe, a ``.himg`` job or a group is
``/api/analysis/validate`` here (its contract test mounts this app); the
run page's embedded editor previews through ``/api/analysis/preview``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from geecs_portal.app import create_app
from test_app import FakeCatalog, _detail

pytest.importorskip("scan_analysis")


@pytest.fixture()
def configs_tree(tmp_path) -> Path:
    tree = tmp_path / "proc_configs"
    (tree / "analyzers" / "HTU").mkdir(parents=True)
    (tree / "analyzers" / "HTU" / "UC_Crop.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "device": "cam",
                "output_name": "UC_Crop",
                "input": {"kind": "camera"},
                "scan": {"priority": 100},
            }
        )
    )
    (tree / "groups" / "HTU").mkdir(parents=True)
    (tree / "groups" / "HTU" / "g.yaml").write_text(
        yaml.safe_dump({"name": "g", "analyzers": ["UC_Crop"]})
    )
    return tree


@pytest.fixture()
def scan_folder(tmp_path) -> Path:
    folder = (
        tmp_path / "Undulator" / "Y2026" / "07-Jul" / "26_0713" / "scans" / "Scan002"
    )
    (folder / "cam").mkdir(parents=True)
    return folder


def _client(scan_folder, configs_tree, **kwargs) -> TestClient:
    catalog = FakeCatalog()
    detail = _detail(2)
    detail.start_doc["scan_folder"] = str(scan_folder)
    catalog.details["uid-002"] = detail
    return TestClient(create_app(catalog, processing_config_dir=configs_tree, **kwargs))


def _stored(configs_tree: Path) -> dict:
    return yaml.safe_load((configs_tree / "analyzers" / "HTU" / "UC_Crop.yaml").read_text())  # fmt: skip


def _validate(client: TestClient, kind: str, document: dict, doc_id: str | None = None):  # fmt: skip
    body = {"document": document, **({"id": doc_id} if doc_id else {})}
    return client.post(f"/api/analysis/validate/{kind}", json=body)


class TestSchema:
    def test_a_recipes_steps_and_measure_are_the_registrys(self, scan_folder, configs_tree):  # fmt: skip
        schema = _client(scan_folder, configs_tree).get("/api/analysis/schema/analysis_recipe").json()  # fmt: skip
        from geecs_analysis.recipe import recipe_schema

        assert schema == recipe_schema() and schema["tier"] == "operator"
        assert (
            "StepRef" not in schema["$defs"]
        )  # bound: the registry's union, not an open reference

    def test_the_other_kinds_are_their_models(self, scan_folder, configs_tree):
        from geecs_schemas.analysis import AnalysisGroup, HimgJob

        client = _client(scan_folder, configs_tree)
        assert client.get("/api/analysis/schema/analysis_group").json() == AnalysisGroup.model_json_schema()  # fmt: skip
        assert client.get("/api/analysis/schema/himg_job").json() == HimgJob.model_json_schema()  # fmt: skip
        assert client.get("/api/analysis/schema/preset").status_code == 404

    def test_off_without_the_configs_tree(self, scan_folder, configs_tree):
        client = TestClient(create_app(FakeCatalog()))
        assert client.get("/api/analysis/schema/analysis_recipe").status_code == 404
        assert _validate(client, "analysis_recipe", {}).status_code == 404


class TestValidate:
    def test_a_recipe_that_binds_is_ok(self, scan_folder, configs_tree):
        doc = {**_stored(configs_tree), "measure": {"kind": "beam"}}
        r = _validate(_client(scan_folder, configs_tree), "analysis_recipe", doc, "HTU/UC_Crop.yaml")  # fmt: skip
        assert r.json() == {"ok": True, "errors": []}

    def test_a_schema_error_is_at_its_field(self, scan_folder, configs_tree):
        doc = {**_stored(configs_tree), "input": {"kind": "telescope"}}
        out = _validate(
            _client(scan_folder, configs_tree), "analysis_recipe", doc
        ).json()
        assert not out["ok"] and out["errors"][0]["loc"][0] == "input"

    def test_a_step_the_registry_refuses_is_at_its_step(
        self, scan_folder, configs_tree
    ):
        doc = {**_stored(configs_tree), "measure": {"kind": "beam"}, "steps": [{"step": "roi", "bounds": [[0, 4], [0, 4]]}, {"step": "no_such_step"}]}  # fmt: skip
        out = _validate(
            _client(scan_folder, configs_tree), "analysis_recipe", doc
        ).json()
        assert not out["ok"] and out["errors"][0]["loc"][:2] == ["steps", 1]

    def test_a_stem_another_file_holds_is_refused(self, scan_folder, configs_tree):
        """Runs load by stem across the whole tree: a second UC_Crop would break every run of it."""
        doc = {**_stored(configs_tree), "measure": {"kind": "beam"}}
        client = _client(scan_folder, configs_tree)
        out = _validate(client, "analysis_recipe", doc, "HTT/UC_Crop.yaml").json()
        assert not out["ok"] and "HTU/UC_Crop.yaml" in out["errors"][0]["msg"]
        # the file itself, and a new stem, are fine
        assert _validate(client, "analysis_recipe", doc, "HTU/UC_Crop.yaml").json()[
            "ok"
        ]
        assert _validate(client, "analysis_recipe", doc, "HTT/UC_Other.yaml").json()[
            "ok"
        ]

    def test_a_tree_already_holding_a_duplicate_says_so(
        self, scan_folder, configs_tree
    ):
        (configs_tree / "analyzers" / "HTT").mkdir()
        (configs_tree / "analyzers" / "HTT" / "UC_Crop.yml").write_text("{}")
        doc = {**_stored(configs_tree), "measure": {"kind": "beam"}}
        out = _validate(_client(scan_folder, configs_tree), "himg_job", {"kind": "himg_to_stack", "device": "cam"}, "HTU/new.yaml").json()  # fmt: skip
        assert (
            not out["ok"]
            and "Duplicate diagnostic ID 'UC_Crop'" in out["errors"][0]["msg"]
        )
        assert not _validate(_client(scan_folder, configs_tree), "analysis_recipe", doc, "HTU/UC_Crop.yaml").json()["ok"]  # fmt: skip

    def test_an_id_out_of_the_folder_is_refused(self, scan_folder, configs_tree):
        r = _validate(_client(scan_folder, configs_tree), "analysis_recipe", _stored(configs_tree), "../x.yaml")  # fmt: skip
        assert r.status_code == 422

    def test_a_groups_references_name_diagnostics(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree)
        assert _validate(client, "analysis_group", {"name": "g", "analyzers": ["UC_Crop"]}).json()["ok"]  # fmt: skip
        out = _validate(client, "analysis_group", {"name": "g", "analyzers": ["UC_Crop", "nope"]}).json()  # fmt: skip
        assert out["ok"] is False and [(e["loc"], "'nope'" in e["msg"]) for e in out["errors"]] == [(["analyzers", 1, "ref"], True)]  # fmt: skip
        twice = _validate(client, "analysis_group", {"name": "g", "analyzers": ["UC_Crop", "UC_Crop"]}).json()  # fmt: skip
        assert twice["ok"] is False and twice["errors"][0]["loc"] == ["analyzers", 1, "ref"] and "duplicate" in twice["errors"][0]["msg"]  # fmt: skip
        named_like_a_ref = _validate(client, "analysis_group", {"name": "UC_Crop", "analyzers": ["nope", "UC_Crop"]}).json()  # fmt: skip
        assert named_like_a_ref["errors"][0]["loc"] == [
            "analyzers",
            0,
            "ref",
        ]  # the group's own quoted name is not a reference

    def test_validate_writes_nothing(self, scan_folder, configs_tree):
        before = {p: p.read_bytes() for p in configs_tree.rglob("*") if p.is_file()}
        _validate(_client(scan_folder, configs_tree), "analysis_recipe", _stored(configs_tree), "HTU/UC_New.yaml")  # fmt: skip
        assert {p: p.read_bytes() for p in configs_tree.rglob("*") if p.is_file()} == before  # fmt: skip


class TestRunPageMount:
    def test_edit_opens_the_configurators_embedded_editor(self, scan_folder, configs_tree):  # fmt: skip
        client = _client(
            scan_folder, configs_tree, configurator_url="http://cfg.test:8800/"
        )
        page = client.get("/run/uid-002").text
        assert 'const CONFIGURATOR = "http://cfg.test:8800";' in page
        assert "/static/embed.js" in page and "/theme/kit.css" in page and "openInConfigurator" in page  # fmt: skip
        assert 'href="http://cfg.test:8800/?kind=analysis_recipe"' in page
        assert (
            client.get("/api/run/uid-002").json()["configurator"]
            == "http://cfg.test:8800"
        )
        listing = client.get("/api/run/uid-002/analysis").json()["analyzers"]
        assert [(a["id"], a["kind"]) for a in listing] == [("UC_Crop", "recipe")]

    def test_absent_without_the_configurator(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree)
        assert 'const CONFIGURATOR = "";' in client.get("/run/uid-002").text
        assert client.get("/api/run/uid-002").json()["configurator"] is None

    def test_the_run_pages_scripts_parse(self, scan_folder, configs_tree):
        from geecs_web_theme.testing import javascript_syntax_error, node_available

        if not node_available():  # pragma: no cover - CI and dev machines have it
            pytest.skip("node not available")
        page = _client(scan_folder, configs_tree, configurator_url="http://cfg.test").get("/run/uid-002").text  # fmt: skip
        import re

        scripts = re.findall(r"<script>(.*?)</script>", page, flags=re.S)
        assert scripts and all(javascript_syntax_error(s) is None for s in scripts)


class TestPreview:
    def test_missing_shot_is_404_and_bad_params_400(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree)
        doc = _stored(configs_tree)
        # no device folder file → the resolver reports the shot missing
        r = client.post(
            "/api/analysis/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 404
        r = client.post(
            "/api/analysis/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", "shot": "x"},
            },
        )
        assert r.status_code == 400
        r = client.post(
            "/api/analysis/preview",
            json={
                "document": doc,
                "params": {"uid": "nope", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 404
        r = client.post("/api/analysis/preview", json={"document": doc, "params": {}})
        assert r.status_code == 404

    def test_renders_the_unsaved_document_on_a_real_frame(
        self, scan_folder, configs_tree, tmp_path
    ):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image

        frame = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 500).astype(np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        client = _client(scan_folder, configs_tree)
        doc = _stored(configs_tree)
        doc["steps"] = [{"step": "roi", "bounds": [[1, 3], [1, 5]]}]
        r = client.post(
            "/api/analysis/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.content[:4] == b"\x89PNG"
        # the file on disk is untouched: the preview is the UNSAVED document
        assert not yaml.safe_load(
            (configs_tree / "analyzers" / "HTU" / "UC_Crop.yaml").read_text()
        ).get("steps")

    def test_the_preview_reads_the_recipes_own_device(self, scan_folder, configs_tree):
        """No host-picked device: the frame is the recipe's folder, whatever the params say.

        The drawer once had a "preview device" picker that drew one camera's
        frame through another camera's recipe, labelled as the recipe's.
        """
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image

        (scan_folder / "other").mkdir()
        rng = np.random.default_rng(3)
        for device in ("cam", "other"):
            frame = rng.integers(0, 4000, size=(8, 8), dtype=np.uint16)
            Image.fromarray(frame).save(
                scan_folder / device / f"Scan002_{device}_001.png"
            )
        client = _client(scan_folder, configs_tree)
        recipe = {
            "schema_version": 3,
            "device": "cam",
            "input": {"kind": "camera"},
            "measure": {"kind": "beam"},
        }

        def preview(**params):
            return client.post(
                "/api/analysis/preview",
                json={
                    "document": recipe,
                    "params": {"uid": "uid-002", "shot": 1, **params},
                },
            )

        own = preview()
        assert own.status_code == 200, own.text
        assert preview(device="other").content == own.content
        recipe["device"] = "other"
        assert preview().content != own.content
        # and the drawer offers no device to pick
        assert 'id="cedev"' not in client.get("/run/uid-002").text

    def test_the_preview_is_the_runs_own_draw(self, scan_folder, configs_tree):
        """A recipe's preview: ScanAnalysis' ``preview_frame``, tight crop.

        Byte-equal to ``scan_analysis.core_preview.preview_frame`` (the
        sink's per-frame call with the document's figure block; that module
        pins itself against the sink's product PNG) saved with
        ``bbox_inches="tight"`` — so the pane shows the product image the
        run would write. No portal palette or window reaches it.
        """
        import io

        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from geecs_analysis.recipe import figure_of
        from geecs_analysis.render import single
        from geecs_schemas.analysis import load_analysis_document
        from PIL import Image
        from scan_analysis.core_preview import (
            measure_frame,
            prepare_document,
            preview_frame,
        )

        rng = np.random.default_rng(1)
        frame = rng.integers(0, 4000, size=(12, 16), dtype=np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        client = _client(scan_folder, configs_tree)
        recipe = {
            "schema_version": 3,
            "device": "cam",
            "input": {"kind": "camera"},
            "steps": [{"step": "roi", "bounds": [[2, 10], [1, 14]]}],
            "measure": {"kind": "beam"},
            "figure": {
                "imshow": {"cmap": "viridis", "vmin": 0},
                "axes": {"title": "as the run draws it"},
                "colorbar": {"label": "counts"},
            },
        }
        r = client.post(
            "/api/analysis/preview",
            json={
                "document": recipe,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 200, r.text
        document = load_analysis_document(recipe)
        buffer = io.BytesIO()
        preview_frame(document, frame, scan_folder=scan_folder).savefig(
            buffer, format="png", bbox_inches="tight"
        )
        assert r.content == buffer.getvalue()
        # and NOT the same frame under a different figure block
        other = single(
            measure_frame(prepare_document(document), frame),
            figure_of(load_analysis_document(dict(recipe, figure={}))),
        )
        buffer = io.BytesIO()
        other.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content != buffer.getvalue()

    def test_the_preview_loads_the_recipes_frame_inputs_like_the_run(
        self, scan_folder, configs_tree
    ):
        """A background under ``{scan_dir}`` resolves to the device folder, as in a run.

        Without the scan folder the placeholder stays literal, the read
        fails, and a recipe without a fallback level makes the preview an
        error — so this test fails (400) if the folder is not passed, and
        the bytes pin that the REAL background frame was subtracted.
        """
        import io

        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from geecs_analysis.engine.compiled import analyze_compiled
        from geecs_analysis.recipe import figure_of
        from geecs_analysis.render import single
        from geecs_schemas.analysis import load_analysis_document
        from PIL import Image
        from scan_analysis.core_inputs import prepare_recipe

        rng = np.random.default_rng(2)
        frame = rng.integers(500, 4000, size=(12, 16), dtype=np.uint16)
        background = rng.integers(0, 400, size=(12, 16), dtype=np.uint16)
        Image.fromarray(frame).save(scan_folder / "cam" / "Scan002_cam_001.png")
        Image.fromarray(background).save(scan_folder / "cam" / "bg.png")
        client = _client(scan_folder, configs_tree)
        recipe = {
            "schema_version": 3,
            "device": "cam",
            "input": {"kind": "camera"},
            "inputs": {"bg": {"path": "{scan_dir}/bg.png"}},  # no fallback level
            "steps": [{"step": "background_frame", "source": "bg"}],
            "measure": {"kind": "beam"},
            "figure": {"imshow": {"cmap": "magma"}},
        }
        r = client.post(
            "/api/analysis/preview",
            json={
                "document": recipe,
                "params": {"uid": "uid-002", "device": "cam", "shot": 1},
            },
        )
        assert r.status_code == 200, r.text
        document = load_analysis_document(recipe)
        prepared = prepare_recipe(document, data_dir=scan_folder / "cam")
        expected = single(
            analyze_compiled(frame, prepared.recipe, inputs=prepared.inputs),
            figure_of(document),
        )
        buffer = io.BytesIO()
        expected.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content == buffer.getvalue()
        # the subtraction happened: the drawn frame is not the raw one
        assert not np.array_equal(
            analyze_compiled(frame, prepared.recipe, inputs=prepared.inputs).frame.data,
            frame.astype(float),
        )


class TestSummaryPreview:
    """The document's summaries over the scan's first shots, as the sink draws them."""

    RECIPE = {
        "schema_version": 3,
        "device": "cam",
        "input": {"kind": "camera"},
        "steps": [{"step": "roi", "bounds": [[1, 11], [2, 14]]}],
        "measure": {"kind": "beam"},
        "figure": {"imshow": {"cmap": "cividis"}},
        "summaries": [
            {"kind": "image_grid", "columns": 2, "panel_size": [3.0, 2.5]},
            {"kind": "average"},
        ],
    }

    @staticmethod
    def _post(client, doc, index, **params):
        return client.post(
            "/api/analysis/preview/summary",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": "cam", **params},
                "index": index,
            },
        )

    def test_summaries_over_the_first_shots_are_the_sinks_own_draw(
        self, scan_folder, configs_tree
    ):
        import io

        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from geecs_schemas.analysis import load_analysis_document
        from PIL import Image
        from scan_analysis.core_preview import preview_summary
        from scan_analysis.core_products import NOSCAN_POSITION_LABEL

        rng = np.random.default_rng(4)
        frames = [
            rng.integers(100, 4000, size=(12, 16), dtype=np.uint16) for _ in range(2)
        ]
        # the fake run recorded 3 events: shots 1 and 3 have a file, shot 2 the
        # device missed (no file), shot 4 is beyond the run — both are skipped
        for shot, frame in zip((1, 3), frames):
            Image.fromarray(frame).save(
                scan_folder / "cam" / f"Scan002_cam_{shot:03d}.png"
            )
        client = _client(scan_folder, configs_tree)
        r = self._post(client, self.RECIPE, 0, shots=4)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        document = load_analysis_document(self.RECIPE)
        # the panels sit at their shot numbers, the skipped shots absent
        expected = preview_summary(
            document,
            frames,
            [1.0, 3.0],
            NOSCAN_POSITION_LABEL,
            0,
            scan_folder=scan_folder,
        )
        buffer = io.BytesIO()
        expected.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content == buffer.getvalue()
        # the average kind, over the same shots
        r = self._post(client, self.RECIPE, 1, shots=4)
        assert r.status_code == 200, r.text
        expected = preview_summary(
            document,
            frames,
            [1.0, 3.0],
            NOSCAN_POSITION_LABEL,
            1,
            scan_folder=scan_folder,
        )
        buffer = io.BytesIO()
        expected.savefig(buffer, format="png", bbox_inches="tight")
        assert r.content == buffer.getvalue()
        # fewer shots asked: fewer panels (shot 1 only)
        r2 = self._post(client, self.RECIPE, 0, shots=1)
        grid = self._post(client, self.RECIPE, 0, shots=4)
        assert r2.status_code == 200 and r2.content != grid.content
        # no third summary in the document
        assert self._post(client, self.RECIPE, 2, shots=4).status_code == 404

    def test_the_cap_bounds_the_reads(self, scan_folder, configs_tree, monkeypatch):
        """At most 8 shots are read however many are asked: the shared host's memory."""
        from dataclasses import replace
        from types import SimpleNamespace

        import pandas as pd
        from geecs_portal import resources
        from test_app import _LV

        np = pytest.importorskip("numpy")
        loads: list[int] = []
        frame = np.full((12, 16), 900, dtype=np.uint16)

        def counting_load(folder, device, shot, **kwargs):
            loads.append(shot)
            return SimpleNamespace(array=frame, kind="png", reason=None)

        monkeypatch.setattr(resources, "load_shot_array", counting_load)
        catalog = FakeCatalog()
        # a run of 20 events, every shot with a timestamp
        detail = replace(
            _detail(2),
            data=pd.DataFrame(
                {
                    "scan_event_index": list(range(1, 21)),
                    "cam-acq_timestamp": [_LV + i for i in range(1, 21)],
                }
            ),
        )
        detail.start_doc["scan_folder"] = str(scan_folder)
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog, processing_config_dir=configs_tree))
        assert self._post(client, self.RECIPE, 0, shots=500).status_code == 200
        assert loads == list(range(1, 9))
        loads.clear()
        # zero asks for the fewest, not the default
        assert self._post(client, self.RECIPE, 0, shots=0).status_code == 200
        assert loads == [1]

    def test_no_frames_at_all_is_404_and_bad_shots_400(self, scan_folder, configs_tree):
        client = _client(scan_folder, configs_tree)
        r = self._post(client, self.RECIPE, 0, shots=3)
        assert r.status_code == 404 and "none of shots 1-3" in r.json()["detail"]
        assert self._post(client, self.RECIPE, 0, shots="many").status_code == 400
        assert (
            client.post(
                "/api/analysis/preview/summary",
                json={"document": self.RECIPE, "params": {}, "index": 0},
            ).status_code
            == 404
        )


_LINE_DOC = {
    "schema_version": 3,
    "device": "U_Scope",
    "input": {
        "kind": "line",
        "folder": "scope",
        "file_tail": ".tsv",
        "loading": {"data_type": "tsv"},
    },
    "measure": {"kind": "line"},
    "scan": {"priority": 100},
}


class TestLinePreview:
    """A line diagnostic previews on the shot's trace, not on an image."""

    @pytest.fixture()
    def seen(self, monkeypatch) -> list:
        import geecs_portal.processing as ephemeral
        from matplotlib.figure import Figure

        calls: list = []

        def fake_render(diag, frames, **kwargs):
            calls.append((frames, kwargs))
            return [Figure()]

        monkeypatch.setattr(ephemeral, "render_document_as_run", fake_render)
        return calls

    @staticmethod
    def _post(client, doc, device="scope", shot=1):
        return client.post(
            "/api/analysis/preview",
            json={
                "document": doc,
                "params": {"uid": "uid-002", "device": device, "shot": shot},
            },
        )

    def test_renders_a_native_trace_file_end_to_end(self, scan_folder, configs_tree):
        np = pytest.importorskip("numpy")
        (scan_folder / "scope").mkdir()
        xy = np.column_stack(
            [np.linspace(0, 1, 50), np.exp(-((np.arange(50) - 25) ** 2) / 20)]
        )
        np.savetxt(scan_folder / "scope" / "Scan002_scope_001.tsv", xy, delimiter="\t")
        client = _client(scan_folder, configs_tree)
        r = self._post(client, dict(_LINE_DOC))
        assert r.status_code == 200, r.text
        assert r.content[:4] == b"\x89PNG"

    def test_the_analyzer_gets_the_files_values_read_by_the_documents_loader(
        self, scan_folder, configs_tree, seen
    ):
        np = pytest.importorskip("numpy")
        (scan_folder / "scope").mkdir()
        xy = np.array([[0.0, 1.0, 5.0], [1.0, 3.0, 6.0], [2.0, 2.0, 7.0]])
        np.savetxt(scan_folder / "scope" / "Scan002_scope_001.tsv", xy, delimiter="\t")
        client = _client(scan_folder, configs_tree)
        doc = dict(_LINE_DOC)
        # the DOCUMENT's loader decides the columns: y from column 2; an
        # auxiliary column the measure does not read travels nowhere (a
        # measure that reads channels gets them appended to the array)
        doc["input"] = {
            **_LINE_DOC["input"],
            "loading": {
                "data_type": "tsv",
                "y_column": 2,
                "auxiliary_columns": {"other": 1},
            },
        }
        assert self._post(client, doc).status_code == 200
        ((frames, kwargs),) = seen
        np.testing.assert_array_equal(frames[0], xy[:, [0, 2]])
        assert "auxiliary_data" not in kwargs

    def test_an_array_stack_serves_its_frame(self, scan_folder, configs_tree, seen):
        np = pytest.importorskip("numpy")
        pytest.importorskip("h5py")
        from test_resources import _write_array_stack

        (scan_folder / "scope").mkdir()
        _write_array_stack(
            scan_folder / "scope" / "scope.h5",
            device="scope",
            variable="trace",
            frames=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            axis=[(0.0, 1.0, 3), (0.0, 1.0, 3)],
        )
        from test_app import _LV

        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        # the run joins by the DIAGNOSTIC's device (its name, not the folder):
        # shot 2's timestamp is the stack's second frame
        detail.data["U_Scope-acq_timestamp"] = [_LV + 0.5, _LV + 1.0, _LV + 2.0]
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog, processing_config_dir=configs_tree))
        doc = dict(_LINE_DOC)
        doc["input"] = {
            "kind": "line",
            "folder": "scope",
            "format": "device_hdf5",
            "loading": {"data_type": "pva_stack"},
        }
        r = self._post(client, doc, shot=2)
        assert r.status_code == 200, r.text
        ((frames, _),) = seen
        np.testing.assert_array_equal(frames[0][:, 1], [4.0, 5.0, 6.0])

    def test_a_stitched_document_previews_the_joined_trace(
        self, scan_folder, configs_tree, seen
    ):
        """The run joins the sibling devices' traces; the preview draws the same."""
        np = pytest.importorskip("numpy")
        low = np.column_stack([np.linspace(0, 1, 5), np.arange(5.0)])
        high = np.column_stack([np.linspace(2, 3, 5), np.arange(5.0) + 10])
        for folder, xy in (("scope", high), ("scope2", low)):
            (scan_folder / folder).mkdir()
            np.savetxt(
                scan_folder / folder / f"Scan002_{folder}_001.tsv", xy, delimiter="\t"
            )
        client = _client(scan_folder, configs_tree)
        doc = dict(_LINE_DOC)
        doc["input"] = {**_LINE_DOC["input"], "siblings": ["scope2"]}
        assert self._post(client, doc).status_code == 200
        ((frames, kwargs),) = seen
        np.testing.assert_array_equal(frames[0], np.concatenate([low, high]))
        assert not (kwargs.get("auxiliary_data") or {}).get("_aux_columns")

    def test_a_missing_shot_file_is_404(self, scan_folder, configs_tree, seen):
        (scan_folder / "scope").mkdir()
        (scan_folder / "scope" / "Scan002_scope_001.tsv").write_text("0\t1\n")
        client = _client(scan_folder, configs_tree)
        r = self._post(client, dict(_LINE_DOC), shot=2)
        assert r.status_code == 404, r.text
        assert "no scope file for shot 2" in r.text
        assert seen == []

    def test_a_shot_the_run_marks_invalid_is_not_previewed(
        self, scan_folder, configs_tree, seen
    ):
        """valid=False means the frame belongs to another shot: the run skips it."""
        pytest.importorskip("h5py")
        from test_app import _LV
        from test_resources import _write_array_stack

        (scan_folder / "scope").mkdir()
        _write_array_stack(
            scan_folder / "scope" / "scope.h5",
            device="scope",
            variable="trace",
            frames=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            axis=[(0.0, 1.0, 3), (0.0, 1.0, 3)],
        )
        catalog = FakeCatalog()
        detail = _detail(2)
        detail.start_doc["scan_folder"] = str(scan_folder)
        detail.data["U_Scope-acq_timestamp"] = [_LV + 0.5, _LV + 1.0, _LV + 2.0]
        detail.data["U_Scope-valid"] = [True, False, True]
        catalog.details["uid-002"] = detail
        client = TestClient(create_app(catalog, processing_config_dir=configs_tree))
        doc = dict(_LINE_DOC)
        doc["input"] = {
            "kind": "line",
            "folder": "scope",
            "format": "device_hdf5",
            "loading": {"data_type": "pva_stack"},
        }
        assert self._post(client, doc, shot=1).status_code == 200
        r = self._post(client, doc, shot=2)
        assert r.status_code == 404, r.text
        assert len(seen) == 1
