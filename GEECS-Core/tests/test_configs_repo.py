"""The one configs-repository resolver."""

from pathlib import Path

from geecs_core import configs_repo


def test_read_config_entry(tmp_path: Path) -> None:
    ini = tmp_path / "config.ini"
    ini.write_text(
        "[Paths]\nscanner_config_root_path = /x\nblank =\n", encoding="utf-8"
    )
    assert (
        configs_repo.read_config_entry("Paths", "scanner_config_root_path", ini) == "/x"
    )
    assert configs_repo.read_config_entry("Paths", "blank", ini) is None
    assert configs_repo.read_config_entry("Nope", "x", ini) is None
    assert configs_repo.read_config_entry("Paths", "x", tmp_path / "absent.ini") is None


def test_resolution_order(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "GEECS-Plugins-Configs"
    (repo / "scanner_configs" / "experiments" / "Exp" / "gateway").mkdir(parents=True)
    target = (
        repo
        / "scanner_configs"
        / "experiments"
        / "Exp"
        / "gateway"
        / "derived_channels.yaml"
    )
    target.write_text("schema_version: 1\n", encoding="utf-8")
    ini = tmp_path / "config.ini"
    ini.write_text(f"[Paths]\nscanner_config_root_path = {repo}\n", encoding="utf-8")

    monkeypatch.setenv("GEECS_SCANNER_CONFIG_DIR", str(tmp_path / "direct"))
    monkeypatch.setenv("GEECS_PLUGINS_CONFIGS", str(repo))
    assert configs_repo.scanner_configs_base(ini) == (tmp_path / "direct").resolve()

    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR")
    base = configs_repo.scanner_configs_base(ini)
    assert base == (repo / "scanner_configs" / "experiments").resolve()

    monkeypatch.delenv("GEECS_PLUGINS_CONFIGS")
    assert configs_repo.scanner_configs_base(ini) == base
    assert configs_repo.scanner_configs_base(tmp_path / "absent.ini") is None

    assert (
        configs_repo.experiment_config_path(
            "Exp", "gateway", "derived_channels.yaml", config_path=ini
        )
        == target
    )
    assert (
        configs_repo.experiment_config_path(
            "Exp", "archiver", "archive_policy.yaml", config_path=ini
        )
        is None
    )
    assert (
        configs_repo.experiment_config_path(
            "Exp", "gateway", "derived_channels.yaml", base=base
        )
        == target
    )
    assert (
        configs_repo.experiment_config_path(
            "Exp", "gateway", "x.yaml", config_path=tmp_path / "absent.ini"
        )
        is None
    )


def test_an_empty_folder_is_the_experiment_folders_root(tmp_path):
    (tmp_path / "Exp").mkdir()
    target = tmp_path / "Exp" / "native_devices.yaml"
    target.write_text("devices: {}\n")
    found = configs_repo.experiment_config_path(
        "Exp", "", "native_devices.yaml", base=tmp_path
    )
    assert found == target


def test_the_write_root_defaults_to_the_read_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEECS_SCANNER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("GEECS_PLUGINS_CONFIGS", str(tmp_path / "read"))
    monkeypatch.delenv("GEECS_CONFIGS_WRITE_ROOT", raising=False)
    assert (
        configs_repo.scanner_configs_write_base() == configs_repo.scanner_configs_base()
    )
    monkeypatch.setenv("GEECS_CONFIGS_WRITE_ROOT", str(tmp_path / "inbox"))
    assert (
        configs_repo.scanner_configs_write_base()
        == (tmp_path / "inbox" / "scanner_configs" / "experiments").resolve()
    )
    # the read root never moves with it
    assert (
        configs_repo.scanner_configs_base()
        == (tmp_path / "read" / "scanner_configs" / "experiments").resolve()
    )


def _worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A checkout laid out as a worktree: ``.git`` is a file naming its own git dir."""
    gitdir = tmp_path / "bare" / "worktrees" / "live"
    gitdir.mkdir(parents=True)
    live = tmp_path / "live"
    (live / "Exp").mkdir(parents=True)
    (live / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    return live, gitdir / "index.lock"


def test_a_missing_file_is_believed_at_once_outside_a_rewrite(tmp_path: Path) -> None:
    live, _ = _worktree(tmp_path)
    import time

    start = time.monotonic()
    assert not configs_repo.settled_exists(live / "Exp" / "absent.yaml")
    assert time.monotonic() - start < 0.5


def test_a_file_missing_mid_rewrite_is_waited_for(tmp_path: Path) -> None:
    import threading

    live, lock = _worktree(tmp_path)
    target = live / "Exp" / "native_devices.yaml"
    lock.write_text("", encoding="utf-8")  # a fast-forward holds the checkout

    def finish() -> None:
        target.write_text("devices: {}\n", encoding="utf-8")
        lock.unlink()

    threading.Timer(0.2, finish).start()
    assert configs_repo.settled_exists(target)


def test_a_rewrite_that_outlasts_the_wait_reads_as_missing(tmp_path: Path) -> None:
    live, lock = _worktree(tmp_path)
    lock.write_text("", encoding="utf-8")
    assert not configs_repo.settled_exists(live / "Exp" / "x.yaml", wait_s=0.1)


def test_a_plain_clone_has_its_lock_in_dot_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert (
        configs_repo._index_lock(tmp_path / "a" / "b.yaml")
        == tmp_path / ".git" / "index.lock"
    )
