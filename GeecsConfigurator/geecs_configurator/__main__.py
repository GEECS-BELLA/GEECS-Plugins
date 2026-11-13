"""``geecs-configurator`` — serve the configurator, settings from the environment."""

from __future__ import annotations

import argparse

import uvicorn

from geecs_configurator.app import create_app
from geecs_configurator.settings import Settings


def main() -> None:
    """Read the settings (``GEECS_CONFIGURATOR_*``) and serve, one worker."""
    parser = argparse.ArgumentParser(prog="geecs-configurator", description=__doc__)
    parser.add_argument("--root-path", default="", help="the front door's prefix, for a proxy that sends no X-Forwarded-Prefix")  # fmt: skip
    args = parser.parse_args()
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.bind, port=settings.port, root_path=args.root_path)  # fmt: skip


if __name__ == "__main__":
    main()
