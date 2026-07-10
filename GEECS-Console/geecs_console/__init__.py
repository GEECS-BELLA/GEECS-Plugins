"""geecs_console — the greenfield PySide6 operator console for GEECS.

The console builds one :class:`geecs_schemas.ScanRequest` per scan and
submits it to the Bluesky engine (``geecs-bluesky``).  It never imports the
legacy GEECS python API package (pinned by a test) — manual set/readback
goes through gateway PVs, DB autocompletes through ``GeecsDb`` (via
``geecs_ca_gateway``, an allowed transitive of ``geecs-bluesky``).  See
this package's ``CLAUDE.md``.
"""

__all__ = ["MAXIMUM_SCAN_SIZE"]

from geecs_console.request_builder import MAXIMUM_SCAN_SIZE
