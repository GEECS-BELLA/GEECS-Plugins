"""Run the plotter: ``geecs-plotter`` / ``python -m geecs_plotter``.

Binds ``127.0.0.1`` by default — localhost first; deployment is a later
PR. The data root and experiment come from ``config.ini`` unless given.
Once the server is up, the page opens in a browser, as the LabVIEW tool
opens its window: the system default, ``--browser chrome`` (any name
Python's :mod:`webbrowser` knows), or ``--browser none`` (what a service
unit passes). A Linux host with no display opens nothing either way.
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
from typing import Callable, Optional

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
    """Parse CLI arguments, build the app, serve."""
    parser = argparse.ArgumentParser(description="GEECS plotter")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8600, help="HTTP port")
    parser.add_argument(
        "--experiment",
        default="",
        help="the experiment folder to read (default: config.ini [Experiment] expt)",
    )
    parser.add_argument(
        "--settings-dir",
        default=None,
        help="where named settings sets are saved (default: config.ini "
        "[Paths] geecs_data/GeecsPlotter/settings, else user data/… in the "
        "main checkout)",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="URL prefix behind a prefix-preserving reverse proxy",
    )
    parser.add_argument(
        "--browser",
        default="default",
        help="open the page once the server is up: 'default' (the system "
        "browser), a name such as 'chrome', 'safari' or 'firefox', or 'none'",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import uvicorn

    from geecs_plotter.app import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(
                args.experiment,
                settings_dir=args.settings_dir,
                root_path=args.root_path,
            ),
            host=args.host,
            port=args.port,
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
