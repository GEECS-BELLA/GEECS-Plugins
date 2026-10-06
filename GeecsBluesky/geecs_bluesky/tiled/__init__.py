"""Tiled for the RunEngine: the engine spools each run, a separate service registers it.

Four modules, imported by their own names (this package imports none of
them, so the scanner's ``spool`` import stays light):

- :mod:`~geecs_bluesky.tiled.integration` — the engine side:
  ``subscribe_tiled_spool`` and the shared checks
  (``tiled_server_reachable``, ``SafeDocumentCallback``);
- :mod:`~geecs_bluesky.tiled.spool` — the per-run JSONL spool both sides
  share, and the writer's heartbeat model;
- :mod:`~geecs_bluesky.tiled.writer` — ``geecs-tiled-writer``, the service
  that registers spooled runs;
- :mod:`~geecs_bluesky.tiled.parquet` — the stream table as a Parquet file
  beside the s-file, registered like a camera stack.
"""
