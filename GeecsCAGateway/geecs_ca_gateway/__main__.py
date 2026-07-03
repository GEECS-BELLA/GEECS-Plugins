"""Run the gateway for a GEECS experiment as a Channel Access server.

    python -m geecs_ca_gateway --experiment Undulator

Builds the config live from the GEECS database (the ``get='yes'`` monitoring
subset by default), connects to the devices, and serves their PVs over CA until
interrupted.  Devices that are unreachable at startup are not fatal — the
per-device supervisor reconnects and their PVs report ``INVALID`` until live.

Network scoping (which subnet the CA server beacons on) is controlled by the
standard EPICS env vars, e.g. ``EPICS_CAS_INTF_ADDR_LIST`` /
``EPICS_CAS_BEACON_ADDR_LIST`` — set them in the deployment environment.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .config import GatewayConfig
from .gateway import GeecsCaGateway

logger = logging.getLogger("geecs_ca_gateway")


class _QuietMissingVariables(logging.Filter):
    """Drop the transport's 'missing variable(s)' notices.

    For a best-effort monitoring gateway, subscribed-but-not-currently-streaming
    variables (analysis off, idle scopes, non-triggered devices lacking
    ``acq_timestamp``) are normal, not warnings — the corresponding PVs simply
    stay ``INVALID`` until data flows. Pass ``--show-missing`` to keep them.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return "missing variable(s)" not in record.getMessage()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="geecs-ca-gateway",
        description="Serve a GEECS experiment's devices as EPICS Channel Access PVs.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="GEECS experiment name (also the PV namespace prefix), e.g. Undulator.",
    )
    parser.add_argument(
        "--all-variables",
        action="store_true",
        help="Expose every device variable, not just the get='yes' monitoring set.",
    )
    parser.add_argument(
        "--no-settable",
        action="store_true",
        help=(
            "Do not add settable (control) variables to the subscribed set. "
            "By default the control surface (e.g. camera save/localsavingpath) "
            "is exposed even when not in the get='yes' monitoring subset."
        ),
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include devices not enabled in the experiment.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Log the transport's 'missing variable(s)' notices (quiet by default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default INFO).",
    )
    return parser.parse_args(argv)


async def _run(
    experiment: str,
    *,
    subscribed_only: bool,
    enabled_only: bool,
    include_settable: bool,
) -> None:
    config = GatewayConfig.from_geecs_experiment(
        experiment,
        subscribed_only=subscribed_only,
        enabled_only=enabled_only,
        include_settable=include_settable,
    )
    gateway = GeecsCaGateway(config)
    logger.info(
        "serving %d PV(s) across %d device(s) for experiment %r",
        len(gateway.pvdb),
        len(config.devices),
        experiment,
    )
    await gateway.run()  # connect -> subscribe -> serve (until cancelled)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.show_missing:
        logging.getLogger("geecs_bluesky.transport.tcp_subscriber").addFilter(
            _QuietMissingVariables()
        )
    try:
        asyncio.run(
            _run(
                args.experiment,
                subscribed_only=not args.all_variables,
                enabled_only=not args.include_disabled,
                include_settable=not args.no_settable,
            )
        )
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
