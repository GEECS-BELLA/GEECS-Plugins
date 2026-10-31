"""``geecs-canopy`` — the package's command line.

``squirrel-sync``  derive Squirrel's PV registry from the module's devices
                   (the native documents and, when the DB is on, the
                   gateway's served set) and reconcile Squirrel with it;
                   ``--check`` prints the drift and writes nothing. Only
                   rows under the experiment's prefix (or naming one of its
                   devices) are removed; a DB-sourced row never while the DB
                   is off; more than REMOVE_GUARD removals need ``--yes``.

Run at deploy and after a device-file change, inside the api container
(``dc run --rm api geecs-canopy squirrel-sync``): the module's settings
and the Squirrel write key (``GEECS_CANOPY_SQUIRREL_KEY``, the secrets
file's) are in its environment. Exit status: 0 in step (or reconciled) ·
1 drift under ``--check``, or a request Squirrel refused (what was
already written is printed) · 2 usage, or a remove step refused (nothing
written) · 3 Squirrel or the GEECS DB did not answer (nothing written: a
half-read device list must never empty the registry).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping, Sequence

import httpx
from geecs_core.configs_repo import EXPERIMENTS_SUBDIR

from geecs_core.pv_naming import pv_name

from geecs_canopy.gateway import source_from_env
from geecs_canopy.module import ModuleSettings
from geecs_canopy.native import NativeSource, SourceError
from geecs_canopy.squirrel import (
    REMOVE_GUARD,
    RemoveRefused,
    SquirrelClient,
    SquirrelError,
    derive,
    reconcile,
)

EXIT_OK, EXIT_DRIFT, EXIT_USAGE, EXIT_UNREACHABLE = 0, 1, 2, 3


async def squirrel_sync(check: bool, url: str | None = None, env: Mapping[str, str] = os.environ, transport: httpx.AsyncBaseTransport | None = None, yes: bool = False) -> int:  # fmt: skip
    """Reconcile (or, with ``check``, compare) Squirrel's registry with the derived one; the exit status."""
    try:
        settings = ModuleSettings.from_env(env)
    except TypeError as exc:
        print(f"usage: the module's settings are incomplete: {exc}", file=sys.stderr)
        return EXIT_USAGE
    url = url or settings.squirrel_url
    if not url or not settings.squirrel_key:
        print("usage: set GEECS_CANOPY_SQUIRREL_URL (or --url) and GEECS_CANOPY_SQUIRREL_KEY", file=sys.stderr)  # fmt: skip
        return EXIT_USAGE
    root = settings.configs_root.joinpath(*EXPERIMENTS_SUBDIR)
    try:
        devices = NativeSource(settings.experiment, root).resolve()
        gateway = source_from_env(settings.experiment, env)
        if gateway is not None:  # a native device wins over a DB device of its name
            devices = gateway() + devices
    except SourceError as exc:
        print(f"the devices cannot be read ({exc}); nothing written", file=sys.stderr)
        return EXIT_UNREACHABLE
    wanted = derive({d.name: d for d in devices}.values())
    client = SquirrelClient(url, settings.squirrel_key, transport=transport)
    prefix = pv_name(settings.experiment) + ":"
    try:
        plan = await reconcile(client, wanted, prefix=prefix, gateway_read=gateway is not None, check=check, yes=yes)  # fmt: skip
    except RemoveRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except SquirrelError as exc:
        if exc.plan is not None and exc.plan.applied:
            print("applied before the refusal:\n" + "\n".join(exc.plan.applied))
        print(exc.detail, file=sys.stderr)
        return EXIT_UNREACHABLE if exc.status == 0 else EXIT_DRIFT
    finally:
        await client.aclose()
    print("\n".join(plan.lines()))
    if plan.refused:  # --check: the write form would stop here
        print(f"note: the removals would be refused: {plan.refused}")
    n = len(plan.add), len(plan.update), len(plan.remove)
    verb = "would change" if check else "reconciled"
    left = f"; {plan.foreign} rows outside {prefix} left alone" if plan.foreign else ""
    print(f"{verb}: {n[0]} to add, {n[1]} to update, {n[2]} to remove; {len(wanted)} rows derived from {len(devices)} devices{left}")  # fmt: skip
    return EXIT_DRIFT if check and plan.drift else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the command."""
    parser = argparse.ArgumentParser(prog="geecs-canopy", description=__doc__.split("\n\n")[1])  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("squirrel-sync", help="reconcile Squirrel's PV registry with the module's devices")  # fmt: skip
    sync.add_argument("--check", action="store_true", help="print the drift, write nothing; exit 1 on drift")  # fmt: skip
    sync.add_argument("--url", help="Squirrel's base URL (default: GEECS_CANOPY_SQUIRREL_URL)")  # fmt: skip
    sync.add_argument("--yes", action="store_true", help=f"allow more than {REMOVE_GUARD} removals in one run")  # fmt: skip
    args = parser.parse_args(argv)
    return asyncio.run(squirrel_sync(args.check, args.url, yes=args.yes))


if (
    __name__ == "__main__"
):  # python -m geecs_canopy.cli, from a bind-mounted source tree
    sys.exit(main())
