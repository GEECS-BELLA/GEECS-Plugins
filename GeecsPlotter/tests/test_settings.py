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


def test_a_local_save_makes_the_folder_ignore_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout on an older branch lacks the root rule; the folder has its own.

    The store is the one ``resolve_store`` really returns for the local
    fallback, re-rooted in tmp_path, so a change of layout moves both.
    """
    monkeypatch.setattr(settings_module, "LOCAL_USER_DATA", tmp_path / "user data")
    config = tmp_path / "config.ini"
    config.write_text("[Paths]\ngeecs_data = /no/such/drive\n")
    store = resolve_store(config_path=config)
    assert store.origin == "local"
    store.save("default", PlotterSettings())
    marker = settings_module.LOCAL_USER_DATA / ".gitignore"
    assert marker.read_text().splitlines()[-1] == "*"
    assert store.path("default").is_relative_to(marker.parent)


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
    assert 'aria-invalid="true" aria-describedby="save-as-err"' in response.text
    assert "spaces" in response.text  # the message says what a name may hold


def test_the_set_list_says_how_to_add_one(client: TestClient) -> None:
    """With only "default", the rail says picks save themselves and how to add a set."""
    page = client.get("/").text
    assert "saves into this set (shared by everyone who opens it)" in page
    assert "Use Save as" in page
    post(client, "/save-as", scan=1, save_as="HTU jet scan")
    page = client.get("/").text
    assert '<option value="htu jet scan">htu jet scan</option>' in page
    assert "Use Save as" not in page


def test_a_store_problem_does_not_blame_the_name(
    client: TestClient, settings_dir: Path
) -> None:
    """A broken open set refuses the save, but the Save-as box is not marked invalid."""
    settings_dir.mkdir()
    (settings_dir / "default.json").write_text("{broken")
    response = client.post(
        "/save-as",
        data={
            "day": "2026-10-02",
            "scan": "1",
            "set": "default",
            "save_as": "good name",
        },
    )
    assert response.status_code == 422
    assert 'value="good name"' in response.text  # kept, nothing lost
    assert 'aria-invalid="true"' not in response.text
    assert "Settings problem" in response.text


def test_a_taken_name_is_marked_on_the_box(client: TestClient) -> None:
    post(client, "/save-as", scan=1, save_as="mine")
    response = client.post(
        "/save-as",
        data={"day": "2026-10-02", "scan": "1", "set": "default", "save_as": "mine"},
    )
    assert 'aria-invalid="true"' in response.text
    assert "already saved" in response.text
    assert "Settings problem" not in response.text


# -- the last scan opened (the set reopens it) ------------------------------


def test_opening_a_scan_saves_it_in_the_set(
    client: TestClient, settings_dir: Path
) -> None:
    post(client, scan=2)
    saved = json.loads((settings_dir / "default.json").read_text())
    assert (saved["day"], saved["scan"]) == ("2026-10-02", 2)


def test_starting_bare_reopens_the_last_scan(client: TestClient) -> None:
    """No day or scan in the URL: back to where this set left off."""
    post(client, scan=2, y="U_ICT charge")
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "./?day=2026-10-02&scan=2&set=default"
    page = client.get("/").text
    assert "Scan002" in page
    assert '<option value="U_ICT charge" selected>' in page


def test_switching_sets_opens_that_sets_scan(client: TestClient) -> None:
    post(client, "/save-as", scan=2, save_as="second")
    post(client, scan=1)  # default moves on to Scan001
    page = client.get("/", params={"set": "second"}).text
    assert "Scan002" in page
    assert 'name="set" value="second"' in page


def test_a_set_that_never_opened_a_scan_is_the_empty_form(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200
    assert 'data-state="empty"' in response.text


def test_a_url_scan_wins_over_the_remembered_one(client: TestClient) -> None:
    post(client, scan=2)
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert "Scan001" in page


def test_a_bad_remembered_name_is_reported_not_redirected(client: TestClient) -> None:
    response = client.get("/", params={"set": "../x"}, follow_redirects=False)
    assert response.status_code == 200
    assert "Settings problem" in response.text


@pytest.mark.parametrize(
    ("field", "value"), [("day", "yesterday"), ("scan", "x"), ("scan", "0")]
)
def test_a_malformed_day_or_scan_is_refused(field: str, value: str) -> None:
    with pytest.raises(SettingsError, match=field):
        merged(PlotterSettings(), {field: value})


def test_an_empty_day_or_scan_keeps_the_saved_one() -> None:
    saved = PlotterSettings(day="2026-10-02", scan=3)
    assert merged(saved, {"day": "", "scan": ""}) == saved


def test_switching_to_a_set_with_no_scan_keeps_the_open_one(
    client: TestClient, settings_dir: Path
) -> None:
    """A set saved before sets remembered scans must not close the open scan."""
    settings_dir.mkdir(exist_ok=True)
    (settings_dir / "old.json").write_text('{"schema_version": 1, "y_filter": "gas"}')
    post(client, scan=2)
    page = client.get(
        "/", params={"set": "old", "here_day": "2026-10-02", "here_scan": 2}
    ).text
    assert "Scan002" in page
    assert 'name="yf" value="gas"' in page  # the old set's controls


def test_the_sets_list_carries_the_open_scan(client: TestClient) -> None:
    page = post(client, scan=2)
    assert '<input type="hidden" name="here_day" value="2026-10-02">' in page
    assert '<input type="hidden" name="here_scan" value="2">' in page


def test_a_field_from_a_newer_copy_survives_a_save(
    client: TestClient, settings_dir: Path
) -> None:
    """The shared folder mixes copies: an unknown field loads and is kept."""
    settings_dir.mkdir(exist_ok=True)
    (settings_dir / "default.json").write_text(
        '{"schema_version": 1, "bin_mode": "mean"}'
    )
    post(client, scan=1, yf="gas")
    saved = json.loads((settings_dir / "default.json").read_text())
    assert saved["bin_mode"] == "mean"
    assert saved["y_filter"] == "gas"


@pytest.mark.parametrize("value", ["0", "4", "two"])
def test_a_bad_y_count_is_refused(value: str) -> None:
    with pytest.raises(SettingsError, match="ycount"):
        merged(PlotterSettings(), {"ycount": value})


def test_replacing_a_saved_set_needs_the_dialogs_yes(
    client: TestClient, settings_dir: Path
) -> None:
    """Save as onto a saved name replaces it only with overwrite=true."""
    post(client, "/save-as", scan=1, yf="first", save_as="mine")
    page = post(
        client, "/save-as", scan=2, yf="second", save_as="Mine", overwrite="true"
    )
    assert 'name="set" value="mine"' in page
    saved = json.loads((settings_dir / "mine.json").read_text())
    assert (saved["y_filter"], saved["scan"]) == ("second", 2)


def test_the_page_offers_saved_names_and_asks_before_replacing(
    client: TestClient,
) -> None:
    post(client, "/save-as", scan=1, save_as="htu jet scan")
    page = client.get("/", params={"day": "2026-10-02", "scan": 1}).text
    assert 'list="saved-sets"' in page
    assert '<option value="htu jet scan">' in page.split('id="saved-sets"')[1]
    dialog = page.split('<dialog id="replace-set"')[1].split("</dialog>")[0]
    assert "autofocus>Keep it</button>" in dialog  # focus never on Replace
    assert 'name="overwrite" value="true">Replace</button>' in dialog


def test_the_api_answers_for_a_set_that_remembers_a_scan(client: TestClient) -> None:
    """A set with a day in it: the filter autosave and the read must not 500."""
    post(client, scan=1)  # the set now remembers 2026-10-02, a date
    saved = client.post("/api/settings/default", data={"yf": "gas"})
    assert saved.status_code == 200
    assert (saved.json()["day"], saved.json()["y_filter"]) == ("2026-10-02", "gas")
    read = client.get("/api/settings/default")
    assert read.status_code == 200
    assert read.json()["scan"] == 1
