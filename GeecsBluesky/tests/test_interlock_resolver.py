"""ConfigsRepoResolver.resolve_interlocks: absent, empty, valid, invalid."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_bluesky.config_resolver import ConfigsRepoResolver


@pytest.fixture
def resolver(tmp_path):
    (tmp_path / "TestExp").mkdir()
    return ConfigsRepoResolver("TestExp", experiments_root=tmp_path)


def _write(resolver, tmp_path, text: str) -> None:
    (tmp_path / "TestExp" / ConfigsRepoResolver.INTERLOCKS_FILE).write_text(text)


def test_absent_is_none(resolver) -> None:
    assert resolver.resolve_interlocks() is None


def test_empty_is_the_empty_set(resolver, tmp_path) -> None:
    _write(resolver, tmp_path, "")
    assert resolver.resolve_interlocks().interlocks == []


def test_valid_document(resolver, tmp_path) -> None:
    _write(
        resolver,
        tmp_path,
        """\
schema_version: 1
interlocks:
  - name: vacuum
    condition: {kind: threshold, device: U_Gauge, variable: Pressure, operator: ">", limit: 1.0e-5}
    on_trip: close_shutters
""",
    )
    (entry,) = resolver.resolve_interlocks().interlocks
    assert entry.name == "vacuum" and entry.on_trip == "close_shutters"


def test_invalid_document_raises(resolver, tmp_path) -> None:
    _write(
        resolver, tmp_path, "interlocks:\n  - name: x\n    condition: {kind: custom}\n"
    )
    with pytest.raises(ValidationError):
        resolver.resolve_interlocks()
