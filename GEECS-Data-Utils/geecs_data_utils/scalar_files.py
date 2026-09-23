"""Shared generated-scalar table persistence, independent of analysis classes.

Hosts own destination naming and directory creation. These helpers never
create a parent directory, including a missing raw scan folder.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_updates(data: pd.DataFrame, key: str = "Shotnumber") -> pd.DataFrame | None:
    """Copy updates, normalize the key spelling and discard rows without a key.

    The first case-insensitive key column wins. Drop other spellings before
    renaming it, so an existing canonical spelling cannot delete the chosen
    key. Duplicate shot rows are resolved by the persistence operation.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.warning("No scalar DataFrame updates to persist")
        return None
    matches = [
        i for i, column in enumerate(data.columns) if str(column).lower() == key.lower()
    ]
    if not matches:
        logger.warning("Updates missing key column %s; skipping write", key)
        return None
    if len(matches) > 1:
        logger.warning("Multiple %s-like columns found; using the first", key)
    chosen = matches[0]
    positions = [i for i in range(len(data.columns)) if i == chosen or i not in matches]
    updates = data.iloc[:, positions].copy()
    columns = list(updates.columns)
    columns[positions.index(chosen)] = key
    updates.columns = columns
    updates = updates.dropna(subset=[key])
    if updates.empty:
        logger.warning("All scalar update rows are missing %s", key)
        return None
    if len(updates) != len(data):
        logger.warning(
            "Dropped %d scalar update rows missing %s", len(data) - len(updates), key
        )
    return updates


def merge_updates(
    current: pd.DataFrame, updates: pd.DataFrame, key: str = "Shotnumber"
) -> pd.DataFrame:
    """Merge on identity, keeping unrelated cells and legacy missing-value policy.

    Last duplicate update wins. Nonmissing updates replace existing cells;
    missing updates retain existing cells through pandas combine_first. Keep
    current column order, append new columns, and sort rows by shot identity.
    Inputs must already have a canonical key column and are never mutated.
    """
    clean = updates.drop_duplicates(subset=[key], keep="last")
    if current.empty:
        return clean.sort_values(by=key)
    merged = (
        clean.set_index(key)
        .combine_first(current.set_index(key))
        .reset_index()
        .sort_values(by=key)
    )
    original = [column for column in current.columns if column in merged]
    return merged[
        original + [column for column in merged.columns if column not in original]
    ]


def merge_sfile(
    path: Path,
    updates: pd.DataFrame,
    key: str = "Shotnumber",
    *,
    timeout: float = 10.0,
    interval: float = 0.1,
) -> pd.DataFrame | None:
    """Read/merge/write one s-file under its exclusive sidecar lock.

    Return the merged table, or None for invalid updates, unreadable current
    data, missing parent directories or lock timeout. Never break another
    writer's lock. Write failures propagate after releasing our lock. Preserve
    the existing .txt.lock convention so legacy and new runners coordinate.
    """
    updates = prepare_updates(updates, key)
    if updates is None:
        return None
    path = Path(path)
    lock = path.with_suffix(path.suffix + ".lock")
    started = time.monotonic()
    while True:
        try:
            with lock.open("x") as handle:
                handle.write(str(os.getpid()))
            break
        except FileExistsError:
            if time.monotonic() - started >= timeout:
                logger.warning("Could not acquire s-file lock %s", lock)
                return None
            time.sleep(interval)
        except OSError as exc:
            logger.warning("Lock error on %s: %s", lock, exc)
            return None
    try:
        current = pd.DataFrame()
        if path.exists():
            try:
                current = pd.read_csv(path, sep="\t")
            except Exception as exc:
                logger.warning("Failed reading s-file %s: %s", path, exc)
                return None
            if key not in current:
                logger.warning(
                    "Existing s-file missing key column %s; skipping merge", key
                )
                return None
        merged = merge_updates(current, updates, key)
        merged.to_csv(path, sep="\t", index=False, header=True)
        return merged
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove s-file lock %s: %s", lock, exc)


def write_scalar_sidecar(
    path: Path, updates: pd.DataFrame, key: str = "Shotnumber"
) -> Path | None:
    """Replace one generated-scalar sidecar; its parent must already exist.

    Store only the supplied columns, sorted by key with the last duplicate row
    retained. Unlike s-file merging, missing generated values remain missing.
    """
    updates = prepare_updates(updates, key)
    if updates is None:
        return None
    path = Path(path)
    updates = updates.drop_duplicates(subset=[key], keep="last").sort_values(by=key)
    updates.set_index(key).to_csv(path, sep="\t", index=True, header=True)
    return path
