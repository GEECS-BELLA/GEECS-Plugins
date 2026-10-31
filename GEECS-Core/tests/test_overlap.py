"""The overlap check: devices enabled in more than one live experiment."""

from __future__ import annotations

import pytest

from geecs_core.db import geecs_db, overlap

SETS = {
    "Main": ["bench-a", "bench-b", "main-only"],
    "Bench stand": ["bench-a", "bench-b"],
    "Third": ["bench-b", "third-only"],
}


def _fake(experiment: str) -> list[str]:
    return SETS[experiment]


def test_names_each_shared_device_and_every_holder() -> None:
    shared = overlap.shared_devices(["Main", "Bench stand", "Third"], devices_of=_fake)
    assert shared == {
        "bench-a": ("Main", "Bench stand"),
        "bench-b": ("Main", "Bench stand", "Third"),
    }


def test_disjoint_sets_and_repeats_report_nothing() -> None:
    assert overlap.shared_devices(["Main", "Main"], devices_of=_fake) == {}
    assert overlap.format_report({}) == []


def test_the_report_names_device_and_both_experiments() -> None:
    lines = overlap.format_report({"bench-a": ("Main", "Bench stand")})
    assert (
        lines[0] == "WARNING: device 'bench-a' is enabled in 'Main' and 'Bench stand'"
    )
    assert "DB edit" in lines[-1]


def test_reads_the_enabled_sets_from_the_db(monkeypatch) -> None:
    """The default source is the existing enabled-only query, one per experiment."""
    queries: list = []

    def fake_query(sql, params=(), *, fetch="all"):
        queries.append((sql, params))
        return [(d, "10.0.0.1", 1) for d in SETS[params[0]]]

    monkeypatch.setattr(geecs_db, "_query", fake_query)
    assert overlap.shared_devices(["Main", "Third"]) == {"bench-b": ("Main", "Third")}
    assert [p for _, p in queries] == [("Main",), ("Third",)]
    assert all("LOWER(ed.enabled) = 'yes'" in sql for sql, _ in queries)
    assert all(sql.lstrip().upper().startswith("SELECT") for sql, _ in queries)


def test_cli_warns_and_exits_zero_unless_strict(monkeypatch, capsys) -> None:
    monkeypatch.setattr(overlap, "_db_devices", _fake)
    assert overlap.main(["Main", "Bench stand"]) == 0
    out = capsys.readouterr().out
    assert "'bench-a' is enabled in 'Main' and 'Bench stand'" in out
    assert overlap.main(["Main", "Bench stand", "--strict"]) == 1
    assert overlap.main(["Main", "--strict"]) == 0


def test_cli_db_failure_is_not_checked_not_clean(monkeypatch, capsys) -> None:
    def broken(_experiment: str) -> list[str]:
        raise ConnectionError("no route")

    monkeypatch.setattr(overlap, "_db_devices", broken)
    assert overlap.main(["Main", "Bench stand"]) == 3
    assert "overlap check skipped" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--strict"]])
def test_cli_needs_an_experiment(argv) -> None:
    with pytest.raises(SystemExit):
        overlap.main(argv)


def test_cli_a_code_bug_is_not_dressed_as_a_db_failure(monkeypatch) -> None:
    def buggy(_experiment: str) -> list[str]:
        raise TypeError("a bug")

    monkeypatch.setattr(overlap, "_db_devices", buggy)
    with pytest.raises(TypeError):
        overlap.main(["Main", "Bench stand"])
