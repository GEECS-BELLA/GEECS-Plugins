"""``geecs-control-room`` — serve the Control Room, settings from the environment."""

from __future__ import annotations

import argparse

import uvicorn

from geecs_control_room.app import create_app
from geecs_control_room.settings import Settings


def main() -> None:
    """Read the settings (``GEECS_CONTROL_ROOM_*``) and serve."""
    parser = argparse.ArgumentParser(prog="geecs-control-room", description=__doc__)
    parser.add_argument(
        "--root-path",
        default="",
        help="the front door's prefix, for a proxy that keeps it and sends no X-Forwarded-Prefix",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(app, host=settings.bind, port=settings.port, root_path=args.root_path)


if __name__ == "__main__":
    main()
