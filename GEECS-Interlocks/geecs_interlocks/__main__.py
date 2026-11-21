"""``geecs-interlocks``: run the interlock server, or listen to one.

::

    geecs-interlocks serve --config main           # the experiment's interlocks/main.yaml, real devices
    geecs-interlocks serve --demo --interlocks examples/demo_interlocks.yaml
    geecs-interlocks serve --fake-devices --interlocks examples/demo_interlocks.yaml
    geecs-interlocks listen 127.0.0.1:5001         # print what Master Control receives (either format)

``--demo`` replays scripted readings; ``--fake-devices`` starts fake GEECS
devices on localhost and reads them through geecs_core. Neither needs the
lab network.

``serve`` opens the server window (the ``gui`` extra) unless
``--no-window``; without PyQt6 it runs headless and says so. Either way
the log goes to the terminal and to the daily log file.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from geecs_schemas import InterlockSet

from geecs_interlocks.catalog import Catalog, DbCatalog, StaticCatalog, named_in
from geecs_interlocks.config import (
    Settings,
    configuration_names,
    interlocks_path,
    load_interlocks,
)
from geecs_interlocks.engine import Engine, SourceFactory, geecs_sources
from geecs_interlocks.logs import LogBuffer, configure_logging
from geecs_interlocks.monitor import NothingToArm
from geecs_interlocks.sources import DeviceSource, PreviewFactory
from geecs_interlocks.wire import WIRE_FORMATS, parse_status, read_frame

logger = logging.getLogger("geecs_interlocks")

#: Seconds between two reconnect attempts of ``listen``.
LISTEN_RETRY_S = 2.0
#: Seconds the web client gets to bind before the server goes on without it.
WEB_START_S = 5.0
#: How long ``--arm-at-start`` waits for the settings to arrive.
ARM_WAIT_S = 30.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geecs-interlocks", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the interlock server")
    where = serve.add_mutually_exclusive_group()
    where.add_argument(
        "--config",
        help="run the named configuration <experiment>/interlocks/<CONFIG>.yaml",
    )
    where.add_argument(
        "--interlocks", type=Path, help="read this interlocks file instead"
    )
    serve.add_argument(
        "--experiment", help="the experiment (default: config.ini [Experiment] expt)"
    )
    fake = serve.add_mutually_exclusive_group()
    fake.add_argument(
        "--demo", action="store_true", help="scripted readings, no devices"
    )
    fake.add_argument(
        "--fake-devices", action="store_true", help="fake GEECS devices on localhost"
    )
    serve.add_argument("--status-host", help="Master Control stream's bind address")
    serve.add_argument("--status-port", type=int, help="Master Control stream's port")
    serve.add_argument(
        "--wire-format",
        choices=WIRE_FORMATS,
        help="the stream's message format (default: config.ini, else summary)",
    )
    serve.add_argument(
        "--external-host", help="where external (UDP) signals are received"
    )
    serve.add_argument(
        "--no-window", action="store_true", help="run headless (no server window)"
    )
    serve.add_argument("--web-host", help="the web client's bind address")
    serve.add_argument("--web-port", type=int, help="the web client's port")
    serve.add_argument(
        "--no-web", action="store_true", help="do not serve the web client"
    )
    serve.add_argument(
        "--arm-at-start",
        action="store_true",
        help="arm the config_change interlocks once their settings have arrived",
    )

    listen = commands.add_parser("listen", help="print the frames a server sends")
    listen.add_argument("address", help="HOST:PORT of a running server")
    listen.add_argument(
        "--every", action="store_true", help="print every frame, not only state changes"
    )
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_config_ini()
    overrides = {
        key: value
        for key in (
            "status_host",
            "status_port",
            "wire_format",
            "external_host",
            "web_host",
            "web_port",
        )
        if (value := getattr(args, key)) is not None
    }
    return settings.model_copy(update=overrides)


def _path(args: argparse.Namespace, settings: Settings) -> Path | None:
    if args.interlocks is not None:
        return args.interlocks
    experiment = args.experiment or settings.experiment
    if experiment is None:
        raise SystemExit(
            "no experiment: pass --experiment or --interlocks, or set "
            "config.ini [Experiment] expt"
        )
    if args.config is None:
        names = configuration_names(experiment)
        raise SystemExit(
            "which configuration? pass --config NAME (saved: "
            f"{', '.join(names) or 'none'}) or --interlocks FILE"
        )
    return interlocks_path(experiment, args.config)


def _sources(
    args: argparse.Namespace, path: Path | None
) -> tuple[SourceFactory, object | None]:
    """The source factory, and the fake devices to stop at exit (if any)."""
    if not (args.demo or args.fake_devices):
        return geecs_sources(), None
    from geecs_interlocks.demo import DEMO_CATALOG, DemoScript, DemoSource, FakeDevices

    if args.demo:
        return (
            lambda doc: DemoSource(
                DemoScript.from_interlocks(doc).with_catalog(DEMO_CATALOG)
            )
        ), None
    # The fake devices are made once, from the file as it is at start.
    interlocks, _ = load_interlocks(path)
    fakes = FakeDevices(DemoScript.from_interlocks(interlocks))
    return geecs_sources(fakes.start()), fakes


def _builder_sources(
    args: argparse.Namespace, interlocks: InterlockSet, experiment: str | None
) -> tuple[Catalog, PreviewFactory]:
    """What the builder lists, and how it previews a device nothing watches."""
    named = StaticCatalog(named_in(interlocks))
    if args.demo or args.fake_devices:
        from geecs_interlocks.demo import DEMO_CATALOG, DemoScript, DemoSource

        script = DemoScript.from_interlocks(interlocks).with_catalog(DEMO_CATALOG)
        return named.merged(DEMO_CATALOG), lambda _variables: DemoSource(script)
    catalog: Catalog = DbCatalog(experiment) if experiment else named
    return catalog, lambda variables: DeviceSource(variables)


class WebClient:
    """The web client's server on a thread of its own (uvicorn), and where it listens."""

    def __init__(
        self, engine: Engine, settings: Settings, experiment: str | None
    ) -> None:
        import uvicorn

        from geecs_interlocks.web import create_app

        config = uvicorn.Config(
            create_app(engine, experiment=experiment),
            host=settings.web_host,
            port=settings.web_port,
            log_level="warning",
            log_config=None,  # uvicorn's own config would close the server's log handlers
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="interlock-web", daemon=True
        )
        host = (
            "127.0.0.1" if settings.web_host in ("0.0.0.0", "") else settings.web_host
        )
        self.url = f"http://{host}:{settings.web_port}/"

    def start(self) -> bool:
        """Start serving; ``False`` (and a warning) if it did not come up in time."""
        self._thread.start()
        deadline = time.monotonic() + WEB_START_S
        while not self._server.started and self._thread.is_alive():
            if time.monotonic() > deadline:
                break
            time.sleep(0.05)
        if not self._server.started:
            logger.warning(
                "the web client did not start (is port %s taken?); serving Master "
                "Control without it",
                self.url.rsplit(":", 1)[-1].strip("/"),
            )
            return False
        logger.info("web client on %s", self.url)
        return True

    def stop(self) -> None:
        """Stop serving."""
        self._server.should_exit = True
        self._thread.join(5)


def _wait_headless() -> None:
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    while not done.wait(0.5):
        pass


def serve(args: argparse.Namespace, buffer: LogBuffer | None = None) -> int:
    """Run the server until the window closes or Ctrl-C."""
    settings = _settings(args)
    path = _path(args, settings)
    make_source, fakes = _sources(args, path)
    engine = Engine(path, make_source, settings)
    engine.start()
    sender = None
    if args.demo or args.fake_devices:
        from geecs_interlocks.demo import FakeSender

        sender = FakeSender(engine.signals.ports())
        sender.start()
    if args.arm_at_start:
        threading.Thread(target=_arm_when_ready, args=(engine,), daemon=True).start()
    host, port = engine.stream.address
    logger.info(
        "serving %s; Master Control stream on %s:%d",
        engine.snapshot().source,
        host,
        port,
    )

    experiment = None if args.interlocks else (args.experiment or settings.experiment)
    web = None if args.no_web else WebClient(engine, settings, experiment)
    if web is not None and not web.start():
        web = None
    try:
        window = None if args.no_window else _window_runner()
        if window is None:
            _wait_headless()
        else:
            catalog, preview = _builder_sources(args, engine.interlocks, experiment)
            window(
                engine,
                buffer or LogBuffer(),
                catalog,
                preview,
                experiment=experiment,
                web_url=None if web is None else web.url,
            )
    finally:
        if web is not None:
            web.stop()
        if sender is not None:
            sender.stop()
        engine.stop()
        if fakes is not None:
            fakes.stop()  # type: ignore[attr-defined]
    return 0


def _window_runner() -> Callable[..., int] | None:
    """``gui.run_window``, or ``None`` (and a warning) when PyQt6 is not installed."""
    if importlib.util.find_spec("PyQt6") is None:
        logger.warning(
            "no server window: PyQt6 is not installed (poetry install -E gui); "
            "running headless"
        )
        return None
    from geecs_interlocks.gui import run_window

    return run_window


def _arm_when_ready(engine: Engine) -> None:
    """Arm the config_change interlocks as soon as every setting has a value."""
    deadline = time.monotonic() + ARM_WAIT_S
    while True:
        try:
            logger.info("armed at start: %s", ", ".join(engine.arm()))
            return
        except NothingToArm:
            return  # nothing in the file holds settings
        except ValueError:  # a setting has no value yet: try again
            if time.monotonic() > deadline:
                logger.exception("could not arm at start")
                return
        time.sleep(0.5)


def listen(args: argparse.Namespace) -> int:
    """Print the frames a server sends until Ctrl-C (reconnecting as needed)."""
    host, _, port = args.address.rpartition(":")
    last = None
    try:
        while True:
            try:
                with socket.create_connection(
                    (host or "127.0.0.1", int(port)), timeout=5
                ) as sock:
                    print(f"-- connected to {host}:{port}", flush=True)
                    while True:
                        message = read_frame(sock)
                        parsed = parse_status(message)
                        key = (
                            parsed.verdict,
                            tuple((p.state, p.name) for p in parsed.parts),
                        )
                        if args.every or key != last:
                            print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)
                            last = key
            except (OSError, ConnectionError) as exc:
                print(f"-- {exc}; retrying in {LISTEN_RETRY_S:g} s", flush=True)
                last = None
                time.sleep(LISTEN_RETRY_S)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    """The ``geecs-interlocks`` command."""
    args = _parser().parse_args(argv)
    if args.command == "serve":
        buffer, _ = configure_logging(
            args.log_level, Settings.from_config_ini().log_dir
        )
        return serve(args, buffer)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return listen(args)


if __name__ == "__main__":
    sys.exit(main())
