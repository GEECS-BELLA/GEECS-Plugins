"""Run the GEECS Data Portal: ``geecs-data-portal`` / ``python -m geecs_portal``.

Injects the real catalog and serves with uvicorn. The catalog is the
data share's scan folders (``FolderScanCatalog``), read directly; with
``--tiled`` it is the Tiled catalog (the ``[tiled]`` section of
``~/.config/geecs_python_api/config.ini``) with every scan folder Tiled
does not claim beside it, which is what the lab service runs.  The service is read-only except explicit analysis
runs; see the package ``CLAUDE.md``.

Once the server is up, the portal opens in a browser, as GEECSplotter
opens its window: the system default, ``--browser chrome`` (any name
Python's :mod:`webbrowser` knows), or ``--browser none`` (what the
service unit passes). A Linux host with no display opens nothing either
way.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional

import matplotlib

# Pinned BEFORE any ScanAnalysis / GEECS-Analysis import: the analysis
# runs execute ScanAnalysis renderers (pyplot) on a worker thread, and a
# GUI backend selected lazily on a non-main thread is a crash. The
# portal's own figures use the Figure API and need no backend at all.
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

#: ``--browser`` values that open nothing.
NO_BROWSER = ("none", "")


def page_url(host: str, port: int, root_path: str = "") -> str:
    """The address to open: a wildcard bind is opened on localhost."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    shown = f"[{shown}]" if ":" in shown else shown
    return f"http://{shown}:{port}{root_path.rstrip('/')}/"


def open_browser(url: str, browser: str) -> bool:
    """Open *url* in *browser* ("default" is the system's); whether it did."""
    if browser.lower() in NO_BROWSER:
        return False
    if not has_display():
        # Over SSH, webbrowser would pick a console browser (lynx, w3m) and
        # take over the terminal the server is logging to.
        logger.info("no display: open %s by hand", url)
        return False
    try:
        controller = webbrowser.get(None if browser.lower() == "default" else browser)
        return controller.open(url)
    except webbrowser.Error as exc:
        logger.warning(
            "could not open %s in %r: %s — open it by hand", url, browser, exc
        )
        return False


def has_display(env: Optional[dict[str, str]] = None) -> bool:
    """Whether a graphical browser can open here (always, off Linux)."""
    if not sys.platform.startswith("linux"):
        return True
    env = os.environ if env is None else env
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def open_when_started(
    started: Callable[[], bool],
    stopping: Callable[[], bool],
    url: str,
    browser: str,
    *,
    poll_s: float = 0.1,
) -> Optional[threading.Thread]:
    """Open the page from a background thread once *started* is true.

    Gives up without opening if the server is asked to stop before it
    starts (a Ctrl-C during startup). A startup failure, such as a port in
    use, exits the process, and this daemon thread with it.
    """
    if browser.lower() in NO_BROWSER:
        return None

    def wait() -> None:
        while not started():
            if stopping():
                return
            time.sleep(poll_s)
        open_browser(url, browser)

    thread = threading.Thread(target=wait, name="open-browser", daemon=True)
    thread.start()
    return thread


def main() -> None:
    """Parse CLI arguments, build the app over the real catalog, serve."""
    parser = argparse.ArgumentParser(description="GEECS Data Portal")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8200, help="HTTP port")
    parser.add_argument(
        "--experiment",
        default="",
        help="default experiment for day listings (query param overrides)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--settings-dir",
        default=None,
        help=(
            "where the Plot tab's named settings sets are saved (default: "
            "config.ini [Paths] geecs_data/DataPortal/settings, else "
            "user data/DataPortal/settings in the main checkout)"
        ),
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help=(
            "list runs from the Tiled catalog (config.ini [tiled]) as well as "
            "the scan folders. Off by default: the scan folders alone, read "
            "straight off the share, so a machine that cannot reach Tiled "
            "never waits for it. The lab service passes it"
        ),
    )
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
        "--config-editor",
        action="store_true",
        help=(
            "mount the analysis config editor at /configs over the "
            "--processing-configs tree (a WRITE verb: saves YAML into that "
            "tree; live preview on the scan page). Needs the analysis extra."
        ),
    )
    parser.add_argument(
        "--root-path",
        default="",
        help=(
            "URL prefix the portal is mounted under behind a PREFIX-PRESERVING "
            "reverse proxy (e.g. /portal): the /static and /theme mounts then "
            "answer at the prefixed path only. A proxy that strips the prefix "
            "must send X-Forwarded-Prefix instead, which overrides this per "
            "request (DEPLOYMENT.md § Behind a reverse proxy)"
        ),
    )
    parser.add_argument(
        "--logbook-url",
        default="",
        help=(
            "the logbook's base URL for the run page's link to a scan's card "
            "(a site value: http://<host>:8400 until the front door routes "
            "/log to it, then /log — a path is same-origin and carries the "
            "portal's own prefix). Empty hides the link; the portal never "
            "mounts the logbook"
        ),
    )
    parser.add_argument(
        "--browser",
        default="default",
        help="open the portal once the server is up: 'default' (the system "
        "browser), a name such as 'chrome', 'safari' or 'firefox', or 'none' "
        "(what the service unit passes)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn

    from geecs_data_utils.folder_catalog import FolderScanCatalog, MergedScanCatalog
    from geecs_data_utils.tiled_catalog import TiledScanCatalog

    from geecs_portal.app import create_app
    from geecs_portal.cache import CachingScanCatalog
    from geecs_portal.plotter.app import create_app as create_plotter

    # Completed runs are immutable: cache their details so plot/image
    # requests stop re-reading them. With Tiled, scan folders no Tiled run
    # claims (Master Control scans, other experiments) list beside the
    # runs, and an unreachable Tiled is skipped for a while after it fails
    # rather than costing every page its connect timeout.
    source = (
        MergedScanCatalog(TiledScanCatalog.from_config(), FolderScanCatalog())
        if args.tiled
        else FolderScanCatalog()
    )
    catalog = CachingScanCatalog(source)
    app = create_app(
        catalog,
        default_experiment=args.experiment,
        processing_config_dir=(
            Path(args.processing_configs) if args.processing_configs else None
        ),
        config_editor=args.config_editor,
        logbook_url=args.logbook_url,
        plotter=create_plotter(
            args.experiment,
            settings_dir=args.settings_dir,
            mount="/plot",
            logbook_url=args.logbook_url,
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            root_path=args.root_path,
        )
    )
    open_when_started(
        lambda: server.started,
        lambda: server.should_exit,
        page_url(args.host, args.port, args.root_path),
        args.browser,
    )
    # uvicorn re-raises the Ctrl-C after a clean shutdown; uvicorn.run
    # swallows it, and so does this.
    with contextlib.suppress(KeyboardInterrupt):
        server.run()


if __name__ == "__main__":
    main()
