"""CLI: ``geecs-pva-gateway --experiment NAME`` — DB-scoped serve, then run.

``geecs-pva-gateway fleet ...`` is the read-only fleet probe
(:func:`geecs_pva_gateway.fleet.fleet_main`) and ``geecs-pva-gateway diff
...`` the file plugin's parity check against the native PNGs
(:func:`geecs_pva_gateway.diff.main`); the serve form stays flat because
``deploy/launch.bat`` on every camera server invokes it that way.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from geecs_pva_gateway.config import PvaGatewayConfig
from geecs_pva_gateway.server import RESTART_EXIT_CODE, GeecsPvaGateway, __version__


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``geecs-pva-gateway`` console script."""
    args_in = sys.argv[1:] if argv is None else argv
    if args_in[:1] == ["fleet"]:
        from geecs_pva_gateway.fleet import fleet_main

        return fleet_main(args_in[1:])
    if args_in[:1] == ["diff"]:
        from geecs_pva_gateway.diff import main as diff_main

        return diff_main(args_in[1:])
    parser = argparse.ArgumentParser(
        description="Serve this host's GEECS camera images as NTNDArray PVs.",
        epilog=(
            "`geecs-pva-gateway fleet --experiment NAME` probes the deployed fleet "
            "(read-only); `geecs-pva-gateway diff <scan folder>` compares a scan's "
            "plugin stacks against its native PNGs."
        ),
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
    if not config.devices:
        # Not an error: the instance serves its identity PVs (version,
        # heartbeat, restart) so the fleet screen sees it, and picks up the
        # host's devices on the next restart (the roster is read at start).
        # Exiting here made a freshly bootstrapped array-only host crash-loop
        # under NSSM until array support landed.
        logging.getLogger(__name__).warning(
            "no devices to serve on this host (no enabled device with image or "
            "served array variables); serving the instance PVs only"
        )

    # The supervisors' endpoint re-resolve (#854): a watched device that stays
    # unreachable is re-asked of the DB at the backoff ceiling.  GeecsDb is
    # already imported by the config build above.
    from geecs_core.db.geecs_db import GeecsDb

    gateway = GeecsPvaGateway(config, endpoint_resolver=GeecsDb.find_device)
    if args.list:
        for name in gateway.pv_names:
            print(name)
        return 0

    try:
        asyncio.run(gateway.run())
    except KeyboardInterrupt:
        pass
    if gateway.restart_requested:
        return RESTART_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
