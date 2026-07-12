"""PresetStore: YAML round-trips against a tmp configs root, hermetic."""

import pytest

from geecs_console.request_builder import (
    ConsoleFormState,
    ConsoleMode,
    FormAxis,
    build_scan_request,
)
from geecs_console.services.presets import PRESET_FOLDER, PresetStore, PresetStoreError
from geecs_schemas import ScanRequest, ScanRequestMode


def one_d_request(variable="jet_x", description="") -> ScanRequest:
    return build_scan_request(
        ConsoleFormState(
            mode=ConsoleMode.ONE_D,
            axes=[FormAxis(variable=variable, start=0.0, stop=1.0, step=0.25)],
            shots_per_step=10,
            save_sets=["Amp4In"],
            description=description,
        )
    )


@pytest.fixture
def store(tmp_path):
    return PresetStore("HTU", experiments_root=tmp_path)


class TestRoundTrip:
    def test_save_load_round_trips_the_request(self, store):
        request = one_d_request(description="align the jet")
        store.save("align", request)
        assert store.load("align") == request

    def test_file_lands_in_the_experiment_presets_dir(self, store, tmp_path):
        path = store.save("align", one_d_request())
        assert path == tmp_path / "HTU" / PRESET_FOLDER / "align.yaml"
        assert path.is_file()

    def test_list_names_sorted(self, store):
        for name in ("zeta", "alpha", "mid"):
            store.save(name, one_d_request())
        assert store.list_names() == ["alpha", "mid", "zeta"]

    def test_overwrite_is_allowed(self, store):
        store.save("align", one_d_request(description="v1"))
        store.save("align", one_d_request(description="v2"))
        assert store.load("align").description == "v2"
        assert store.list_names() == ["align"]

    def test_delete_removes_the_preset(self, store):
        store.save("align", one_d_request())
        store.delete("align")
        assert store.list_names() == []
        with pytest.raises(PresetStoreError, match="not found"):
            store.load("align")

    def test_yml_twin_is_listed_and_loadable(self, store, tmp_path):
        folder = tmp_path / "HTU" / PRESET_FOLDER
        store.save("seed", one_d_request())
        (folder / "seed.yaml").rename(folder / "twin.yml")
        assert store.list_names() == ["twin"]
        assert store.load("twin").mode is ScanRequestMode.STEP

    def test_set_experiment_switches_folders(self, store, tmp_path):
        store.save("align", one_d_request())
        store.set_experiment("Bella")
        assert store.list_names() == []
        store.save("bella-only", one_d_request())
        assert (tmp_path / "Bella" / PRESET_FOLDER / "bella-only.yaml").is_file()


class TestErrors:
    def test_missing_dir_lists_empty(self, store):
        assert store.list_names() == []

    def test_load_missing_preset_raises_clearly(self, store):
        with pytest.raises(PresetStoreError, match="'ghost' not found"):
            store.load("ghost")

    def test_delete_missing_preset_raises_clearly(self, store):
        with pytest.raises(PresetStoreError, match="'ghost' not found"):
            store.delete("ghost")

    def test_invalid_yaml_raises_clearly(self, store, tmp_path):
        folder = tmp_path / "HTU" / PRESET_FOLDER
        folder.mkdir(parents=True)
        (folder / "broken.yaml").write_text("mode: [unclosed")
        with pytest.raises(PresetStoreError, match="not valid YAML"):
            store.load("broken")

    def test_non_mapping_yaml_raises_clearly(self, store, tmp_path):
        folder = tmp_path / "HTU" / PRESET_FOLDER
        folder.mkdir(parents=True)
        (folder / "list.yaml").write_text("- just\n- a\n- list\n")
        with pytest.raises(PresetStoreError, match="YAML mapping"):
            store.load("list")

    def test_schema_rejected_document_raises_clearly(self, store, tmp_path):
        folder = tmp_path / "HTU" / PRESET_FOLDER
        folder.mkdir(parents=True)
        (folder / "bad.yaml").write_text("mode: step\naxes: []\n")
        with pytest.raises(PresetStoreError, match="not a valid ScanRequest"):
            store.load("bad")

    def test_no_experiment_selected_save_raises(self, tmp_path):
        store = PresetStore("", experiments_root=tmp_path)
        with pytest.raises(PresetStoreError, match="No experiment selected"):
            store.save("align", one_d_request())
        assert store.list_names() == []

    def test_missing_configs_repo_lists_empty_and_save_raises(self, monkeypatch):
        # Offline: the lazy configs-root resolution finds nothing.
        monkeypatch.setattr(
            "geecs_console.services.presets._configs_base", lambda: None
        )
        store = PresetStore("HTU")
        assert store.list_names() == []
        with pytest.raises(PresetStoreError, match="Configs repo not found"):
            store.save("align", one_d_request())

    @pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "..", "."])
    def test_bad_names_rejected(self, store, name):
        with pytest.raises(PresetStoreError, match="name"):
            store.save(name, one_d_request())
