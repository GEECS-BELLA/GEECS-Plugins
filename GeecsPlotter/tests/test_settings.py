"""Named settings sets: the file store, where it lives, and the page's use of it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_plotter import settings as settings_module
from geecs_plotter.settings import (
    PlotterSettings,
    SettingsError,
    SettingsStore,
    merged,
    resolve_store,
    set_name,
)
from tests.conftest import post


# -- the store --------------------------------------------------------------


def test_a_set_round_trips_as_readable_json(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "sets")
    path = store.save("HTU-jet", PlotterSettings(x_filter="jet", y_alias=False))
    assert path == tmp_path / "sets" / "htu-jet.json"
    assert json.loads(path.read_text())["x_filter"] == "jet"
    assert store.load("htu-jet") == PlotterSettings(x_filter="jet", y_alias=False)
    assert not list(path.parent.glob("*.partial"))


def test_a_name_never_saved_is_the_defaults(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "missing")
    assert store.load("default") == PlotterSettings()
    assert store.names() == ["default"]
    assert not (tmp_path / "missing").exists()  # reading never creates


def test_the_default_set_is_listed_first(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    for name in ("zeta", "alpha", "default"):
        store.save(name, PlotterSettings())
    (tmp_path / "notes.txt").write_text("not a set")
    assert store.names() == ["default", "alpha", "zeta"]


@pytest.mark.parametrize("name", ["", "..", ".hidden", "a/b", "a\\b", "x" * 65, "a b"])
def test_a_name_cannot_leave_the_folder(name: str) -> None:
    with pytest.raises(SettingsError):
        set_name(name)


def test_an_older_file_loads_with_new_fields_at_their_defaults(tmp_path: Path) -> None:
    (tmp_path / "old.json").write_text('{"schema_version": 1, "y": "U_ICT charge"}')
    loaded = SettingsStore(tmp_path).load("old")
    assert loaded.y == "U_ICT charge"
    assert loaded.x_auto is True


def test_a_broken_file_is_an_error_not_the_defaults(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(SettingsError, match="bad.json"):
        SettingsStore(tmp_path).load("bad")


def test_merging_keeps_what_the_form_does_not_send() -> None:
    saved = PlotterSettings(x="a", y="b", y_filter="gas")
    out = merged(saved, {"x": "", "xauto": "false"})
    assert (out.x, out.y, out.y_filter, out.x_auto) == ("", "b", "gas", False)
    with pytest.raises(SettingsError):
        merged(saved, {"xalias": "yes"})


# -- where the folder is ----------------------------------------------------


def test_the_shared_user_data_folder_comes_from_config_ini(tmp_path: Path) -> None:
    user_data = tmp_path / "user data"
    user_data.mkdir()
    config = tmp_path / "config.ini"
    config.write_text(f"[Paths]\ngeecs_data = {user_data}\n")
    store = resolve_store(config_path=config)
    assert store.origin == "shared"
    assert store.folder == user_data / "GeecsPlotter" / "settings"


@pytest.mark.parametrize(
    "text", ["", "[Paths]\n", "[Paths]\ngeecs_data = /no/such/drive/user data\n"]
)
def test_an_unset_or_unreachable_user_data_folder_falls_back_locally(
    tmp_path: Path, text: str
) -> None:
    config = tmp_path / "config.ini"
    config.write_text(text)
    store = resolve_store(config_path=config)
    assert store.origin == "local"
    assert store.folder == settings_module.LOCAL_USER_DATA / "GeecsPlotter" / "settings"


def test_the_local_fallback_is_gitignored() -> None:
    gitignore = settings_module.LOCAL_USER_DATA.parent / ".gitignore"
    assert "/user data/" in gitignore.read_text().splitlines()


def test_a_given_folder_wins(tmp_path: Path) -> None:
    store = resolve_store(tmp_path / "mine", config_path=tmp_path / "absent.ini")
    assert (store.folder, store.origin) == (tmp_path / "mine", "given")


# -- the page ---------------------------------------------------------------


def test_a_pick_saves_the_set_and_the_url_stays_short(
    client: TestClient, settings_dir: Path
) -> None:
    response = client.post(
        "/",
        data={"day": "2026-10-02", "scan": "1", "set": "default", "y": "U_ICT charge"},
    )
    assert response.url.query.decode() == "day=2026-10-02&scan=1&set=default"
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["y"] == "U_ICT charge"


def test_save_as_writes_a_new_set_and_opens_it(
    client: TestClient, settings_dir: Path
) -> None:
    page = post(client, "/save-as", scan=1, yf="gas", save_as="Gas-Scan")
    assert 'name="set" value="gas-scan"' in page
    assert '<option value="gas-scan" selected>gas-scan</option>' in page
    assert json.loads((settings_dir / "gas-scan.json").read_text())["y_filter"] == "gas"
    assert not (settings_dir / "default.json").exists()  # the source is untouched


def test_save_as_refuses_a_name_already_saved(
    client: TestClient, settings_dir: Path
) -> None:
    post(client, "/save-as", scan=1, yf="first", save_as="mine")
    response = client.post(
        "/save-as",
        data={"day": "2026-10-02", "scan": "1", "set": "default", "save_as": "mine"},
    )
    assert response.status_code == 422
    assert "already saved" in response.text
    assert json.loads((settings_dir / "mine.json").read_text())["y_filter"] == "first"


def test_switching_sets_opens_the_other_one(client: TestClient) -> None:
    post(client, "/save-as", scan=1, y="U_ICT charge", xauto="false", x="", save_as="b")
    page = client.get("/", params={"day": "2026-10-02", "scan": 1, "set": "b"}).text
    assert '<option value="" selected>shot #</option>' in page
    other = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert '<option value="" selected>shot #</option>' not in other


def test_a_broken_set_is_shown_and_never_overwritten(
    client: TestClient, settings_dir: Path
) -> None:
    settings_dir.mkdir()
    (settings_dir / "default.json").write_text("{broken")
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert "Settings problem" in page
    response = client.post(
        "/", data={"day": "2026-10-02", "scan": "1", "set": "default"}
    )
    assert response.status_code == 422
    assert (settings_dir / "default.json").read_text() == "{broken"


def test_the_page_names_the_settings_folder(
    client: TestClient, settings_dir: Path
) -> None:
    assert str(settings_dir) in client.get("/").text


# -- the API ----------------------------------------------------------------


def test_the_api_lists_and_saves_sets(client: TestClient, settings_dir: Path) -> None:
    saved = client.post("/api/settings/default", data={"xf": "jet"})
    assert saved.status_code == 200
    assert saved.json()["x_filter"] == "jet"
    assert client.get("/api/settings/default").json()["x_filter"] == "jet"
    listing = client.get("/api/settings").json()
    assert listing == {
        "folder": str(settings_dir),
        "origin": "given",
        "sets": ["default"],
    }


def test_the_api_refuses_a_bad_name(client: TestClient) -> None:
    assert client.post("/api/settings/.secret", data={"xf": "x"}).status_code == 422
