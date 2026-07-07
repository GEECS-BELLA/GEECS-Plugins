"""Tests for the CLI entry point argument parsing (no network)."""

from __future__ import annotations

import pytest

from geecs_ca_gateway.__main__ import _parse_args


def test_defaults() -> None:
    """Only --experiment is required; the monitoring subset is the default."""
    args = _parse_args(["--experiment", "Undulator"])
    assert args.experiment == "Undulator"
    assert args.all_variables is False
    assert args.include_disabled is False
    assert args.show_missing is False


def test_flags() -> None:
    """The boolean flags flip their defaults."""
    args = _parse_args(
        ["--experiment", "X", "--all-variables", "--include-disabled", "--show-missing"]
    )
    assert args.all_variables is True
    assert args.include_disabled is True
    assert args.show_missing is True


def test_experiment_required() -> None:
    """--experiment is mandatory."""
    with pytest.raises(SystemExit):
        _parse_args([])
