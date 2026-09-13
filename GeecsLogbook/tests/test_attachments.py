"""The host-side attachment store: claimed names, contained serving, no debris."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from geecs_logbook import _fs
from geecs_logbook.attachments import AttachmentStore


@pytest.fixture
def blobs(tmp_path: Path) -> AttachmentStore:
    """A store under a directory that does not exist yet."""
    return AttachmentStore(tmp_path / "state" / "attachments")


def test_save_numbers_repeats_and_serves_them(blobs: AttachmentStore) -> None:
    """image.png twice is image.png and image-2.png; path() finds both."""
    assert blobs.save("e1", "image.png", b"one") == "image.png"
    assert blobs.save("e1", "image.png", b"two") == "image-2.png"
    assert blobs.path("e1", "image-2.png").read_bytes() == b"two"
    assert [p.name for p in blobs.files("e1")] == ["image-2.png", "image.png"]
    assert blobs.files("nobody") == []


def test_parallel_same_name_saves_all_get_their_own_file(
    blobs: AttachmentStore,
) -> None:
    """Four pastes of image.png in flight at once: four files, four names."""
    with ThreadPoolExecutor(4) as pool:
        names = sorted(
            pool.map(lambda i: blobs.save("e", "image.png", bytes([i]) * 8), range(4))
        )
    assert names == ["image-2.png", "image-3.png", "image-4.png", "image.png"]
    assert sorted(p.name for p in blobs.files("e")) == names


def test_serving_is_contained(blobs: AttachmentStore) -> None:
    """Neither the filename nor the entry id can escape the attachment root.

    The decoys sit where an escape would land: the database beside the
    store (``..`` as the entry id) and a file at the root itself (``.``).
    """
    blobs.save("e", "a.png", b"x")
    (blobs.root.parent / "logbook.db").write_text("SQLite format 3")
    (blobs.root / "loose.txt").write_text("no")
    assert blobs.path("e", "a.png") is not None
    assert blobs.path("e", "../loose.txt") is None
    assert blobs.path("e", "../../logbook.db") is None
    assert blobs.path("..", "logbook.db") is None
    assert blobs.path(".", "loose.txt") is None
    assert blobs.path("e", "missing.png") is None


def test_failed_write_leaves_no_placeholder(
    blobs: AttachmentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claimed name whose bytes never land is released; the retry gets it back."""

    def hiccup(path: Path, data: bytes) -> None:
        raise OSError("disk hiccup")

    monkeypatch.setattr(_fs, "replace_with", hiccup)
    with pytest.raises(OSError):
        blobs.save("e", "hic.png", b"x")
    assert not (blobs.root / "e" / "hic.png").exists()
    monkeypatch.undo()
    assert blobs.save("e", "hic.png", b"x") == "hic.png"
