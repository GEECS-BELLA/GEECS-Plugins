"""Tests for the CLI entry point argument parsing (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_ca_gateway.__main__ import _parse_args


def test_defaults() -> None:
    """Only --experiment is required; the monitoring subset is the default."""
    args = _parse_args(["--experiment", "Undulator"])
    assert args.experiment == "Undulator"
    assert args.all_variables is False
    assert args.include_disabled is False
    assert args.derived_channels is None
    assert args.show_missing is False


def test_flags() -> None:
    """The boolean flags flip their defaults."""
    args = _parse_args(
        ["--experiment", "X", "--all-variables", "--include-disabled", "--show-missing"]
    )
    assert args.all_variables is True
    assert args.include_disabled is True
    assert args.show_missing is True


def test_derived_channels_path_flag() -> None:
    """A derived-channel overlay path is parsed for startup loading."""
    args = _parse_args(["--experiment", "X", "--derived-channels", "derived.yaml"])
    assert args.derived_channels == Path("derived.yaml")


def test_experiment_required() -> None:
    """--experiment is mandatory."""
    with pytest.raises(SystemExit):
        _parse_args([])


def test_main_exits_with_restart_code(monkeypatch) -> None:
    """A restart-requested run makes main() exit with RESTART_EXIT_CODE."""
    from geecs_ca_gateway import __main__ as entry

    async def fake_run(experiment: str, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(entry, "_run", fake_run)
    # --show-missing: without it main() installs a process-global log filter
    # that would swallow the warning test_transport asserts later in the run.
    with pytest.raises(SystemExit) as exc_info:
        entry.main(["--experiment", "X", "--show-missing"])
    assert exc_info.value.code == entry.RESTART_EXIT_CODE


def test_main_returns_normally_without_restart(monkeypatch) -> None:
    """A normal shutdown does not raise SystemExit."""
    from geecs_ca_gateway import __main__ as entry

    seen: dict[str, object] = {}

    async def fake_run(experiment: str, **kwargs: object) -> bool:
        seen.update(kwargs)
        return False

    monkeypatch.setattr(entry, "_run", fake_run)
    entry.main(
        ["--experiment", "X", "--derived-channels", "derived.yaml", "--show-missing"]
    )  # no exception
    assert seen["derived_channels_path"] == Path("derived.yaml")
