"""Run the capture daemon: ``python -m geecs_bluesky.capture`` or ``geecs-capture-daemon``.

Wires DB-driven camera discovery, the p4p frame source, and the Bluesky
document stream (the worker's 0MQ proxy out-port, from the shared
``[qserver]`` config's ``doc_addr``) into a long-running ``CaptureDaemon``.
Requires the ``capture`` extra (p4p + h5py) and, for the document stream,
the worker's proxy reachable on the network.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True, help="GEECS experiment name")
    ap.add_argument(
        "--doc-addr",
        default=None,
        help="document-stream address host:port (default: [qserver] doc_addr)",
    )
    ap.add_argument(
        "--queue-size", type=int, default=100, help="p4p monitor queue depth"
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    doc_addr = args.doc_addr
    if doc_addr is None:
        from geecs_bluesky.qs_client.client import read_qserver_config

        config = read_qserver_config()
        if config is None:
            print(
                "ERROR: no --doc-addr and no [qserver] section in the shared config",
                file=sys.stderr,
            )
            return 2
        doc_addr = config.doc_addr

    from .daemon import CaptureDaemon
    from .discovery import discover_capture_cameras
    from .subscriber import P4pFrameSource

    targets = discover_capture_cameras(args.experiment)
    if not targets:
        print(
            f"ERROR: no capture-eligible cameras in {args.experiment}", file=sys.stderr
        )
        return 2

    daemon = CaptureDaemon(
        experiment=args.experiment,
        targets=targets,
        source_factory=lambda: P4pFrameSource(queue_size=args.queue_size),
    )

    from bluesky.callbacks.zmq import RemoteDispatcher

    dispatcher = RemoteDispatcher(doc_addr)
    dispatcher.subscribe(daemon)
    logger.info(
        "capture daemon: %d cameras, doc stream %s — running (Ctrl-C to stop)",
        len(targets),
        doc_addr,
    )
    try:
        dispatcher.start()  # blocks
    except KeyboardInterrupt:
        pass
    finally:
        daemon.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
