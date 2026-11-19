"""``geecs-interlocks``: run the interlock server, or listen to one.

::

    geecs-interlocks serve                         # the experiment's interlocks.yaml, real devices
    geecs-interlocks serve --demo --interlocks examples/demo_interlocks.yaml
    geecs-interlocks serve --fake-devices --interlocks examples/demo_interlocks.yaml
    geecs-interlocks listen 127.0.0.1:5001         # print what Master Control receives

``--demo`` replays scripted readings; ``--fake-devices`` starts fake GEECS
devices on localhost and reads them through geecs_core. Neither needs the
lab network.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from geecs_interlocks.config import Settings, interlocks_path, load_interlocks
from geecs_interlocks.engine import Engine, SourceFactory, geecs_sources
from geecs_interlocks.wire import parse_status, read_frame

logger = logging.getLogger("geecs_interlocks")

#: Seconds between two reconnect attempts of ``listen``.
LISTEN_RETRY_S = 2.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geecs-interlocks", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the interlock server")
    where = serve.add_mutually_exclusive_group()
    where.add_argument(
        "--experiment", help="read <experiment>/interlocks.yaml from the configs repo"
    )
    where.add_argument(
        "--interlocks", type=Path, help="read this interlocks file instead"
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
        for key in ("status_host", "status_port")
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
    return interlocks_path(experiment)


def _sources(
    args: argparse.Namespace, path: Path | None
) -> tuple[SourceFactory, object | None]:
    """The source factory, and the fake devices to stop at exit (if any)."""
    if not (args.demo or args.fake_devices):
        return geecs_sources(), None
    from geecs_interlocks.demo import DemoScript, DemoSource, FakeDevices

    if args.demo:
        return (lambda doc: DemoSource(DemoScript.from_interlocks(doc))), None
    # The fake devices are made once, from the file as it is at start.
    interlocks, _ = load_interlocks(path)
    fakes = FakeDevices(DemoScript.from_interlocks(interlocks))
    return geecs_sources(fakes.start()), fakes


def serve(args: argparse.Namespace) -> int:
    """Run the server until Ctrl-C."""
    settings = _settings(args)
    path = _path(args, settings)
    make_source, fakes = _sources(args, path)
    engine = Engine(path, make_source, settings)
    engine.start()
    host, port = engine.stream.address
    logger.info(
        "serving %s; Master Control stream on %s:%d",
        engine.snapshot().source,
        host,
        port,
    )

    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    try:
        while not done.wait(0.5):
            pass
    finally:
        engine.stop()
        if fakes is not None:
            fakes.stop()  # type: ignore[attr-defined]
    return 0


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
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return serve(args) if args.command == "serve" else listen(args)


if __name__ == "__main__":
    sys.exit(main())
