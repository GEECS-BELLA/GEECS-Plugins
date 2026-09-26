"""The recipe asks for workers; the host caps the count and keeps small runs serial."""

import logging

import pytest

from scan_analysis import core_workers
from scan_analysis.core_workers import (
    MIN_UNITS_FOR_POOL,
    default_worker_cap,
    effective_workers,
    host_worker_cap,
)


def test_default_cap_leaves_one_core_free(monkeypatch):
    monkeypatch.setattr(core_workers.os, "cpu_count", lambda: 8)
    assert default_worker_cap() == 7
    monkeypatch.setattr(core_workers.os, "cpu_count", lambda: 1)
    assert default_worker_cap() == 1
    monkeypatch.setattr(core_workers.os, "cpu_count", lambda: None)
    assert default_worker_cap() == 1


def test_host_cap_reads_the_client_config(tmp_path, monkeypatch):
    monkeypatch.setattr(core_workers.os, "cpu_count", lambda: 8)
    config = tmp_path / "config.ini"
    assert host_worker_cap(config) == 7  # no file: the default
    config.write_text("[Paths]\nx = y\n")
    assert host_worker_cap(config) == 7  # no section: the default
    config.write_text("[analysis]\nworker_cap = 3\n")
    assert host_worker_cap(config) == 3


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_unusable_cap_values_are_logged_and_ignored(
    tmp_path, monkeypatch, caplog, value
):
    monkeypatch.setattr(core_workers.os, "cpu_count", lambda: 4)
    config = tmp_path / "config.ini"
    config.write_text(f"[analysis]\nworker_cap = {value}\n")
    with caplog.at_level(logging.WARNING, logger="scan_analysis.core_workers"):
        assert host_worker_cap(config) == 3
    assert any("worker_cap" in r.getMessage() for r in caplog.records)


def test_effective_workers_honours_request_cap_and_small_run_floor(monkeypatch):
    assert effective_workers(1, 10_000, cap=8) == 1
    assert effective_workers(8, MIN_UNITS_FOR_POOL - 1, cap=8) == 1
    assert effective_workers(8, MIN_UNITS_FOR_POOL, cap=8) == 8
    assert effective_workers(8, 3600, cap=3) == 3
    assert effective_workers(2, 3600, cap=8) == 2
    # Without an explicit cap the host's config decides.
    monkeypatch.setattr(core_workers, "host_worker_cap", lambda: 5)
    assert effective_workers(16, 3600) == 5
