"""Pin: the portal uses GEECS-Data-Utils' shared browser helpers by identity.

``resolve_scan_folder`` and ``metadata_rows`` live in
``geecs_data_utils.tiled_catalog`` so every front end resolves a scan folder
and renders the metadata table the same way.  This pin moved here from the
Qt console's suite when that package was deleted (2026-09-14): a portal-local
re-growth of either helper would silently drift the folder resolution (and
its read-only invariant) or the metadata rows away from the shared layer.
"""

from __future__ import annotations

from geecs_data_utils import tiled_catalog

from geecs_portal import app as portal_app


def test_resolve_scan_folder_is_the_shared_data_utils_implementation() -> None:
    """The portal must not grow a shadowing resolver of its own."""
    assert portal_app.resolve_scan_folder is tiled_catalog.resolve_scan_folder


def test_metadata_rows_is_the_shared_data_utils_implementation() -> None:
    """Same pin for the metadata table helper."""
    assert portal_app.metadata_rows is tiled_catalog.metadata_rows
