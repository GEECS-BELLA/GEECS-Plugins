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


@pytest.mark.parametrize(
    ("typed", "stored"),
    [
        ("HTU jet scan", "htu jet scan"),
        ("  HTU   jet  ", "htu jet"),
        ("run_3.v2", "run_3.v2"),
    ],
)
def test_a_name_may_have_spaces(typed: str, stored: str, tmp_path: Path) -> None:
    """LabVIEW users name files with spaces; so may a set."""
    assert set_name(typed) == stored
    assert (
        SettingsStore(tmp_path).save(typed, PlotterSettings()).name == f"{stored}.json"
    )


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


@pytest.mark.parametrize(
    "name", ["", "  ", "..", ".hidden", "a/b", "a\\b", "x" * 65, "scan.", "-a"]
)
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
    gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
    assert "/user data/" in gitignore.read_text().splitlines()


def test_a_local_save_makes_the_folder_ignore_itself(tmp_path: Path) -> None:
    """A checkout on an older branch lacks the root rule; the folder has its own."""
    user_data = tmp_path / "user data"
    store = SettingsStore(
        user_data / "GeecsPlotter" / "settings", "local", root=tmp_path
    )
    store.save("default", PlotterSettings())
    assert (user_data / ".gitignore").read_text().splitlines()[-1] == "*"


def test_a_shared_save_writes_no_gitignore(tmp_path: Path) -> None:
    store = SettingsStore(
        tmp_path / "GeecsPlotter" / "settings", "shared", root=tmp_path
    )
    store.save("default", PlotterSettings())
    assert not (tmp_path / ".gitignore").exists()


@pytest.mark.parametrize(
    ("package_file", "root"),
    [
        ("/r/GEECS-Plugins/GeecsPlotter/geecs_plotter/settings.py", "/r/GEECS-Plugins"),
        (
            "/r/GEECS-Plugins/.claude/worktrees/plotter-x/GeecsPlotter/geecs_plotter/settings.py",
            "/r/GEECS-Plugins",
        ),
    ],
)
def test_every_worktree_shares_the_main_checkouts_folder(
    package_file: str, root: str
) -> None:
    assert settings_module.checkout_root(Path(package_file)) == Path(root)


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


# -- the share going away, and failed saves (review of #1166) ---------------


def test_an_unreachable_root_is_an_error_not_the_defaults(tmp_path: Path) -> None:
    """A dropped share must not read as "never saved", or a pick wipes the set."""
    store = SettingsStore(tmp_path / "gone" / "sets", "shared", root=tmp_path / "gone")
    with pytest.raises(SettingsError, match="not reachable"):
        store.load("default")
    with pytest.raises(SettingsError, match="not reachable"):
        store.save("default", PlotterSettings())
    assert not (tmp_path / "gone").exists()  # nothing built on a bare mount point
    assert store.names() == ["default"]


def test_a_dropped_share_keeps_the_saved_set(
    share: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The share goes away after start: the page says so and saves nothing."""
    from geecs_plotter import app as app_module
    from tests.conftest import EXPERIMENT

    user_data = tmp_path / "user data"
    user_data.mkdir()
    config = tmp_path / "config.ini"
    config.write_text(f"[Paths]\ngeecs_data = {user_data}\n")
    store = resolve_store(config_path=config)
    monkeypatch.setattr(app_module, "resolve_store", lambda _given: store)
    client = TestClient(app_module.create_app(EXPERIMENT, base_directory=share))
    post(client, scan=1, yf="gas")
    saved = store.path("default").read_text()

    user_data.rename(tmp_path / "unmounted")
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert "not reachable" in page
    response = client.post(
        "/", data={"day": "2026-10-02", "scan": "1", "set": "default", "yf": ""}
    )
    assert response.status_code == 422
    assert not user_data.exists()  # no shadow tree on the bare mount point
    (tmp_path / "unmounted").rename(user_data)
    assert store.path("default").read_text() == saved


def test_a_failed_save_is_a_settings_problem_not_a_500(
    client: TestClient, settings_dir: Path
) -> None:
    (settings_dir / "default.json").mkdir(
        parents=True
    )  # a file that cannot be replaced
    response = client.post(
        "/", data={"day": "2026-10-02", "scan": "1", "set": "default", "yf": "x"}
    )
    assert response.status_code == 422
    assert "Settings problem" in response.text
    api = client.post("/api/settings/default", data={"xf": "x"})
    assert api.status_code == 422
    assert not list(settings_dir.glob("*.tmp"))


def test_a_given_folder_is_created_however_deep(tmp_path: Path) -> None:
    """``--settings-dir`` on a fresh host: no parent to check, the save makes it."""
    store = resolve_store(tmp_path / "state" / "geecs-plotter" / "settings")
    assert store.load("default") == PlotterSettings()
    store.save("default", PlotterSettings(x_filter="jet"))
    assert store.load("default").x_filter == "jet"


def test_a_refused_chmod_still_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CIFS mount without unix extensions refuses chmod; the set still saves."""

    def refuse(*_args: object) -> None:
        raise PermissionError("chmod not supported")

    monkeypatch.setattr(settings_module.os, "chmod", refuse)
    store = SettingsStore(tmp_path)
    store.save("default", PlotterSettings(y_filter="gas"))
    assert store.load("default").y_filter == "gas"


def test_a_refused_name_is_kept_in_the_box_and_explained(client: TestClient) -> None:
    response = client.post(
        "/save-as",
        data={
            "day": "2026-10-02",
            "scan": "1",
            "set": "default",
            "save_as": "jet/scan",
        },
    )
    assert response.status_code == 422
    assert 'value="jet/scan"' in response.text
    assert 'role="alert"' in response.text
    assert "spaces" in response.text  # the message says what a name may hold


def test_the_set_list_says_how_to_add_one(client: TestClient) -> None:
    """With only "default", the rail says picks save themselves and how to add a set."""
    page = client.get("/").text
    assert "saves into this set as you go" in page
    assert "Use Save as" in page
    post(client, "/save-as", scan=1, save_as="HTU jet scan")
    page = client.get("/").text
    assert '<option value="htu jet scan">htu jet scan</option>' in page
    assert "Use Save as" not in page
