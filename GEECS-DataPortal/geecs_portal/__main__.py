"""Run the GEECS Data Portal: ``geecs-data-portal`` / ``python -m geecs_portal``.

Injects the real catalog (``TiledScanCatalog.from_config()`` — the
``[tiled]`` section of ``~/.config/geecs_python_api/config.ini``) and
serves with uvicorn.  The service is read-only by doctrine; see the
package ``CLAUDE.md``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    """Parse CLI arguments, build the app over the real catalog, serve."""
    parser = argparse.ArgumentParser(description="GEECS Data Portal (read-only)")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8200, help="HTTP port")
    parser.add_argument(
        "--experiment",
        default="",
        help="default experiment for day listings (query param overrides)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--processing-configs",
        default="",
        help=(
            "scan-analysis configs tree for the Images tab's processing "
            "selector (the parent of analyzers/). Omitted = feature OFF "
            "(the portal never falls back to the global config "
            "resolution); also needs the 'analysis' extra installed"
        ),
    )
    parser.add_argument(
        "--root-path",
        default="",
        help=(
            "URL prefix the portal is mounted under behind a reverse proxy "
            "(e.g. /portal); a proxy-sent X-Forwarded-Prefix header "
            "overrides this per request"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn

    from geecs_data_utils.tiled_catalog import TiledScanCatalog

    from geecs_portal.app import create_app
    from geecs_portal.cache import CachingScanCatalog

    # Completed runs are immutable: cache their details so plot/image
    # requests stop re-downloading the full event table from Tiled.
    catalog = CachingScanCatalog(TiledScanCatalog.from_config())
    app = create_app(
        catalog,
        default_experiment=args.experiment,
        processing_config_dir=(
            Path(args.processing_configs) if args.processing_configs else None
        ),
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        root_path=args.root_path,
    )


if __name__ == "__main__":
    main()
