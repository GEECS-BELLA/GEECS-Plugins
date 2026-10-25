"""Run the device panels: ``geecs-panels`` / ``python -m geecs_panels``.

Binds ``127.0.0.1`` by default: the prototype runs on a laptop over the
VPN, reaching the IOCs over CA. The experiment comes from ``config.ini``
(``[Experiment] expt``) unless given; the CA address list from
``config.ini [epics] ca_addr_list``, which must name the IOC hosts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """Parse CLI arguments, build the app, serve."""
    parser = argparse.ArgumentParser(description="GEECS device panels")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8500, help="HTTP port")
    parser.add_argument(
        "--experiment",
        default=None,
        help="the experiment whose native-device registry to read "
        "(default: config.ini [Experiment] expt)",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=3.0,
        help="seconds without a new frame before a value or the image is stale",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=10.0,
        help="the most frames a page asks for per second",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="URL prefix behind a prefix-preserving reverse proxy",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="refuse every write (crosshairs, analysis ROI): a viewer only",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        metavar="NAME=PREFIX",
        help="an oscilloscope IOC to give a panel, as NAME=PREFIX (e.g. scope1=<experiment>:scope1:) "
        "(repeatable; bench scopes are not in the native-device registry yet)",
    )
    parser.add_argument(
        "--bench",
        default=None,
        metavar="PATH",
        help="a bench file (YAML): devices on bench IOCs the registry does not list, "
        "as panel devices (each with its family, prefix and options: allow, note) "
        "or in the older per-kind form; see the README",
    )
    parser.add_argument(
        "--push-period",
        type=float,
        default=0.25,
        help="seconds between a device stream's frames (/api/devices/{name}/events)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from geecs_core.configs_repo import read_config_entry
    from geecs_core.epics_env import apply_epics_address_config

    # Before anything imports aioca: libca reads the address list once.
    apply_epics_address_config()
    # A laptop drops off the VPN: libca's searches for the lost channels back
    # off to 5 min by default, and no IOC beacon crosses subnets to reset them.
    # 60 s (libca's floor) bounds the reconnect; an explicit setting wins.
    os.environ.setdefault("EPICS_CA_MAX_SEARCH_PERIOD", "60")

    experiment = args.experiment or read_config_entry("Experiment", "expt")
    if not experiment:
        sys.exit(
            "geecs-panels: no experiment (pass --experiment or set config.ini [Experiment] expt)"
        )

    import uvicorn

    from geecs_schemas.epics.family import load_families

    from geecs_panels.app import create_app
    from geecs_panels.bridge.catalog import load_experiment
    from geecs_panels.bridge.routes import Bridge
    from geecs_panels.bridge.stream import DeviceStreams
    from geecs_panels.ca import (
        LiveCameras,
        LiveMotors,
        LiveScopes,
        LiveSupplies,
        read_pvs,
        subscribe_reading,
    )
    from geecs_panels.registry import RegistryError

    families = load_families()
    try:
        loaded = load_experiment(
            experiment,
            families,
            bench=Path(args.bench) if args.bench else None,
            scopes=args.scope,
        )
    except RegistryError as exc:
        sys.exit(f"geecs-panels: {exc}")
    bridge = Bridge(
        experiment,
        loaded.served,
        loaded.errors + loaded.unserved,
        DeviceStreams(
            {d.name: d for d in loaded.served},
            subscribe_reading,
            read=read_pvs,
            stale_after=args.stale_after,
            push_s=args.push_period,
        ),
        families,
    )
    app = create_app(
        experiment,
        loaded.pages,
        # The floor between upstream fetches for ALL viewers of a camera
        # (FrameSource.latest). Half the page's poll period, so timer jitter
        # never hands a viewer its own previous frame.
        LiveCameras(min_interval=0.5 / args.max_fps),
        registry_path=loaded.registry_path,
        registry_error="; ".join(loaded.errors) or None,
        stale_after=args.stale_after,
        max_fps=args.max_fps,
        root_path=args.root_path,
        writable=not args.read_only,
        scopes=LiveScopes(),
        motors=LiveMotors(),
        supplies=LiveSupplies(),
        bridge=bridge,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
