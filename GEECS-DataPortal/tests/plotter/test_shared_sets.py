"""Shared sets: read from the configs root, saved only by Save as, never by a pick."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from geecs_schemas.plot_settings import PlotterSettings

from geecs_portal.plotter.app import create_app
from geecs_portal.plotter.settings import (
    SettingsError,
    SettingsStore,
    set_name,
    shared_stem,
)
from geecs_portal.plotter.shared_sets import PENDING_S, SetBook, SharedSets
from tests.plotter.conftest import EXPERIMENT, post


# -- names --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "stem"),
    [("1BL 2BL timing", "1bl-2bl-timing"), ("shared/jet", "jet"), (" a  b ", "a-b")],
)
def test_a_save_as_name_folds_to_a_file_stem(typed: str, stem: str) -> None:
    assert shared_stem(typed) == stem
    assert set_name(f"shared/{typed.removeprefix('shared/')}") == f"shared/{stem}"


def test_this_hosts_store_never_holds_a_shared_name(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="changes only by Save as"):
        SettingsStore(tmp_path).save("shared/jet", PlotterSettings())
    assert not list(tmp_path.iterdir())


# -- the page and the API -----------------------------------------------------


def test_a_pick_under_a_shared_set_saves_into_the_viewers_own_set(
    client: TestClient, settings_dir: Path, shared_dir: Path, configurator
) -> None:
    """Opening a shared set and changing it copies it into this host's working copy."""
    post(client, "/save-as", scan=1, yf="gas", save_as="team", owner="ada")
    page = post(client, scan=1, set="shared/team", owner="bo", xf="jet")
    assert 'name="set" value="bo"' in page
    mine = SettingsStore(settings_dir).load("bo")
    assert (mine.y_filter, mine.x_filter) == ("gas", "jet")
    assert (
        yaml.safe_load((shared_dir / "team.yaml").read_text()).get("x_filter") is None
    )
    assert configurator.saves == [("team", "ada")]  # one commit: the Save as


def test_the_live_save_refuses_a_shared_set(client: TestClient, configurator) -> None:
    post(client, "/save-as", scan=1, save_as="team")
    response = client.post("/api/settings/shared/team", data={"xf": "jet"})
    assert response.status_code == 422
    assert "changes only by Save as" in response.json()["detail"]
    assert len(configurator.saves) == 1


def test_save_as_names_the_viewer(client: TestClient, configurator) -> None:
    response = client.post(
        "/save-as",
        data={"day": "2026-10-02", "scan": "1", "set": "default", "save_as": "team"},
    )
    assert response.status_code == 422
    assert "say who you are" in response.text
    assert configurator.saves == []


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"configs_root": "configs"}, "--configurator-url"),
        ({"set_saver": "fake"}, "--configs-root"),
    ],
)
def test_without_a_configurator_or_configs_root_save_as_says_which_flag(
    share: Path,
    settings_dir: Path,
    tmp_path: Path,
    configurator,
    kwargs: dict,
    flag: str,
) -> None:
    given = {
        k: (configurator if v == "fake" else tmp_path / v) for k, v in kwargs.items()
    }
    client = TestClient(
        create_app(EXPERIMENT, base_directory=share, settings_dir=settings_dir, **given)
    )
    response = client.post(
        "/save-as",
        data={
            "day": "2026-10-02",
            "scan": "1",
            "set": "default",
            "owner": "ada",
            "save_as": "team",
        },
    )
    assert response.status_code == 422
    assert flag in response.text
    assert 'aria-invalid="true"' not in response.text  # not the name's fault
    assert configurator.saves == []


def test_a_newer_portals_field_survives_a_save_as(
    client: TestClient, settings_dir: Path, shared_dir: Path
) -> None:
    settings_dir.mkdir()
    (settings_dir / "ada.json").write_text('{"y": "charge", "added_later": [1, 2]}')
    post(client, "/save-as", scan=1, set="ada", owner="ada", save_as="team")
    assert yaml.safe_load((shared_dir / "team.yaml").read_text())["added_later"] == [
        1,
        2,
    ]


def test_a_shared_file_written_by_hand_opens(
    client: TestClient, shared_dir: Path
) -> None:
    """A set committed by any route (a PR, the configurator's page) opens on the page."""
    shared_dir.mkdir(parents=True)
    (shared_dir / "by-hand.yml").write_text(
        "y: U_ICT charge\nscan: 2\nday: 2026-10-02\n"
    )
    (shared_dir / "Not A Stem.yaml").write_text("y: x\n")  # never listed
    (shared_dir / "notes.txt").write_text("not a set")
    assert client.get("/api/settings").json()["shared"] == ["shared/by-hand"]
    page = client.get("/", params={"set": "shared/by-hand"}).text
    assert "Scan002" in page


def test_a_broken_shared_file_is_a_settings_problem(
    client: TestClient, shared_dir: Path
) -> None:
    shared_dir.mkdir(parents=True)
    (shared_dir / "broken.yaml").write_text("y_count: 99\n")
    response = client.get("/api/settings/shared/broken")
    assert response.status_code == 422
    assert "broken.yaml" in response.json()["detail"]


# -- a set just saved, before the configs root shows it -----------------------


class _Committed:
    """A configurator whose commit has not reached the configs root yet."""

    def save(self, stem: str, document: dict, *, operator: str, overwrite: bool) -> str:
        return "committed"


def test_a_set_just_saved_is_answered_until_the_configs_root_shows_it(
    tmp_path: Path,
) -> None:
    now = [0.0]
    shared = SharedSets(folder=tmp_path, saver=_Committed(), clock=lambda: now[0])
    book = SetBook(local=SettingsStore(tmp_path / "local"), shared=shared)
    assert book.save_as("Team", PlotterSettings(y="a"), operator="ada", overwrite=False) == "shared/team"  # fmt: skip
    assert book.load("shared/team").y == "a"  # the file is not there yet
    assert book.shared_names() == ["shared/team"]
    (tmp_path / "team.yaml").write_text("y: a\n")  # the share fast-forwarded
    assert book.load("shared/team").y == "a"
    (tmp_path / "team.yaml").write_text("y: b\n")  # someone else's later commit
    assert book.load("shared/team").y == "b"  # the file is the set again


def test_a_save_the_configs_root_never_shows_is_forgotten(tmp_path: Path) -> None:
    now = [0.0]
    shared = SharedSets(folder=tmp_path, saver=_Committed(), clock=lambda: now[0])
    shared.save("team", PlotterSettings(y="a"), operator="ada", overwrite=False)
    now[0] = PENDING_S + 1
    with pytest.raises(SettingsError, match="no shared set shared/team"):
        shared.load("team")
    assert shared.stems() == []


def test_a_missing_shared_set_is_refused_not_the_defaults(
    client: TestClient, shared_dir: Path, tmp_path: Path
) -> None:
    """A pick under a missing shared set never saves the defaults over the viewer's own."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    assert client.get("/api/settings/shared/nope").status_code == 422
    reply = client.post(
        "/api/settings/ada", data={"copy_from": "shared/nope", "y": "a"}
    )
    assert reply.status_code == 422


def test_an_unlistable_shared_folder_lists_nothing(tmp_path: Path, monkeypatch) -> None:
    shared = SharedSets(folder=tmp_path)

    def broken(_self):
        raise PermissionError("share blip")

    monkeypatch.setattr(Path, "iterdir", broken)
    assert shared.stems() == []
