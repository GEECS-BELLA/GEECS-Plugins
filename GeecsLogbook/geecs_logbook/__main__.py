"""Run the GEECS logbook: ``geecs-logbook`` / ``python -m geecs_logbook``.

One experiment's logbook on one port, served with uvicorn. Entries need a
SQLite file (``--notes-db``); under systemd it defaults to
``$STATE_DIRECTORY/logbook.db`` — the unit template's ``StateDirectory=``
— so a deployed logbook has somewhere durable to write without a site
value in code or in the unit. See ``deploy/DEPLOYMENT.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main() -> None:
    """Parse CLI arguments, build the app, serve."""
    parser = argparse.ArgumentParser(description="GEECS logbook")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8400, help="HTTP port")
    parser.add_argument(
        "--experiment",
        required=True,
        help="the experiment whose share this logbook reads (a site value: "
        "GEECS_EXPERIMENT in site.env)",
    )
    parser.add_argument(
        "--notes-db",
        default=None,
        help=(
            "SQLite file for the entries; uploads go to attachments/ beside "
            "it (a WRITE verb: rows here, markdown mirrored into each day's "
            "logbook/ folder on the share). Default: logbook.db under "
            "systemd's $STATE_DIRECTORY when set, else none — a read-only "
            "logbook. Its directory must already exist."
        ),
    )
    parser.add_argument(
        "--templates-dir",
        default="",
        help=(
            "directory of *.md seed templates — the type buttons on every "
            "composer; logbook_templates/ at the top of the configs checkout. "
            "Omitted = plain composers"
        ),
    )
    parser.add_argument(
        "--root-path",
        default="",
        help=(
            "URL prefix the logbook is mounted under behind a reverse proxy "
            "(e.g. /log); a proxy-sent X-Forwarded-Prefix header overrides "
            "this per request"
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    notes_db: Path | None = Path(args.notes_db) if args.notes_db else None
    if notes_db is None and os.environ.get("STATE_DIRECTORY"):
        # StateDirectory= in the unit: systemd creates it, owns it to the
        # service user and exports this variable.
        notes_db = Path(os.environ["STATE_DIRECTORY"].split(":")[0]) / "logbook.db"

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn

    from geecs_logbook.app import create_app

    app = create_app(
        args.experiment,
        notes_db=notes_db,
        templates_dir=Path(args.templates_dir) if args.templates_dir else None,
        root_path=args.root_path,
    )
    logging.getLogger(__name__).info(
        "logbook for %s on :%d (%s; templates %s)",
        args.experiment,
        args.port,
        f"entries in {notes_db}" if notes_db else "read-only",
        args.templates_dir or "none",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
