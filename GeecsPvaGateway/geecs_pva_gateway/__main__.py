"""CLI: ``geecs-pva-gateway --experiment NAME`` — DB-scoped serve, then run."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from geecs_pva_gateway.config import PvaGatewayConfig
from geecs_pva_gateway.server import GeecsPvaGateway, __version__


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``geecs-pva-gateway`` console script."""
    parser = argparse.ArgumentParser(
        description="Serve this host's GEECS camera images as NTNDArray PVs."
    )
    parser.add_argument("--experiment", required=True, help="GEECS experiment name")
    parser.add_argument(
        "--host",
        default=None,
        help="endpoint IP to scope to (default: this machine's addresses)",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated device subset (default: all scoped cameras)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the served devices and PV names, then exit",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="logging level (default INFO)"
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    devices = args.devices.split(",") if args.devices else None
    config = PvaGatewayConfig.from_geecs_experiment(
        args.experiment, host=args.host, devices=devices
    )
    if not config.cameras:
        print(
            "no cameras to serve (host scope matched no enabled devices with "
            "image variables)",
            file=sys.stderr,
        )
        return 1

    gateway = GeecsPvaGateway(config)
    if args.list:
        for name in gateway.pv_names:
            print(name)
        return 0

    try:
        asyncio.run(gateway.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
