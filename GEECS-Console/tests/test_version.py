"""console_version: source-tree pyproject wins, metadata second, then unknown."""

import importlib.metadata

import geecs_console.version as version_mod
from geecs_console.version import console_version


def test_reads_version_from_pyproject(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.poetry]\nname = "geecs-console"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    assert console_version(pyproject) == "9.9.9"


def test_default_reads_the_dev_checkout_pyproject():
    """In this repo the adjacent pyproject.toml exists and wins."""
    import tomllib
    from pathlib import Path

    expected = tomllib.loads(
        (Path(version_mod.__file__).parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["tool"]["poetry"]["version"]
    assert console_version() == expected


def test_missing_pyproject_falls_back_to_metadata(tmp_path):
    got = console_version(tmp_path / "nope" / "pyproject.toml")
    assert got == importlib.metadata.version("geecs-console")


def test_missing_pyproject_and_metadata_is_unknown(tmp_path, monkeypatch):
    def raise_not_found(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    assert console_version(tmp_path / "nope" / "pyproject.toml") == "unknown"


def test_malformed_pyproject_falls_back(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("not toml at [ all", encoding="utf-8")

    def raise_not_found(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    assert console_version(pyproject) == "unknown"


def test_httpx_logger_quieted_by_configure_logging():
    """The Tiled health probe's httpx INFO chatter is capped at WARNING."""
    import logging
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(version_mod.__file__).parent.parent))
    try:
        import main as console_main
    finally:
        sys.path.pop(0)

    root = logging.getLogger()
    before = list(root.handlers)
    level_before = root.level
    try:
        console_main.configure_logging("INFO")
        assert logging.getLogger("httpx").level == logging.WARNING
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(level_before)
