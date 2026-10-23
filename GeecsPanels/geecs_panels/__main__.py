"""Run the device panels: ``geecs-panels`` / ``python -m geecs_panels``.

Binds ``127.0.0.1`` by default: the prototype runs on a laptop over the
VPN, reaching the IOCs over CA. The experiment comes from ``config.ini``
(``[Experiment] expt``) unless given; the CA address list from
``config.ini [epics] ca_addr_list``, which must name the IOC hosts.
"""

from __future__ import annotations

import argparse
import logging
import sys


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

    experiment = args.experiment or read_config_entry("Experiment", "expt")
    if not experiment:
        sys.exit(
            "geecs-panels: no experiment (pass --experiment or set config.ini [Experiment] expt)"
        )

    import uvicorn

    from geecs_panels.app import create_app
    from geecs_panels.ca import LiveCameras
    from geecs_panels.registry import RegistryError, load_devices, registry_path

    path = registry_path(experiment)
    error = None
    try:
        devices = load_devices(experiment, path)
    except RegistryError as exc:
        devices, error = [], str(exc)
    app = create_app(
        experiment,
        devices,
        # Only used when the camera's frame counter is unknown (normally a
        # frame is reused exactly while the counter stands still). Half the
        # page's poll period, so timer jitter never hands a viewer its own
        # previous frame.
        LiveCameras(min_interval=0.5 / args.max_fps),
        registry_path=path,
        registry_error=error,
        stale_after=args.stale_after,
        max_fps=args.max_fps,
        root_path=args.root_path,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
