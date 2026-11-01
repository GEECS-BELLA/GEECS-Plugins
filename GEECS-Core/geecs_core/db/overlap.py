"""Devices enabled in more than one live experiment — a read-only check.

Each experiment instance (``deploy/experiment.sh``) runs its own gateway and
worker over its own ``expt_device`` set, so a device enabled in two live
experiments is driven by two gateways and two workers at once: two TCP
subscriptions, two scan-mode announcers, two experiments that may both set
it. Nothing here refuses that — the device sets are the DB's, and fixing
one is a human's DB edit — but ``experiment up`` prints this report as a
warning naming every shared device and each experiment holding it.

    python -m geecs_core.db.overlap EXPERIMENT [EXPERIMENT ...]

Exit 0 whether or not devices are shared (``--strict``: 1 when some are);
3 when the DB could not be read, so a caller can tell "no overlap" from
"not checked".
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence

#: ``experiment -> its enabled device names`` (the DB query by default).
DevicesOf = Callable[[str], Iterable[str]]


def _db_devices(experiment: str) -> Iterable[str]:
    from geecs_core.db.geecs_db import GeecsDb

    return GeecsDb.get_experiment_devices(experiment, enabled_only=True)


def shared_devices(
    experiments: Iterable[str], *, devices_of: DevicesOf | None = None
) -> dict[str, tuple[str, ...]]:
    """``{device: (experiment, ...)}`` for each device enabled in two or more.

    Parameters
    ----------
    experiments:
        The live experiments (DB names); repeats count once.
    devices_of:
        Source of each experiment's enabled devices; the GEECS DB by default.

    Returns
    -------
    dict
        Sorted by device; each tuple in the order *experiments* gave.
    """
    source = devices_of or _db_devices
    owners: dict[str, list[str]] = {}
    for experiment in dict.fromkeys(experiments):
        for device in source(experiment):
            owners.setdefault(device, []).append(experiment)
    return {
        device: tuple(holders)
        for device, holders in sorted(owners.items())
        if len(holders) > 1
    }


def format_report(shared: Mapping[str, Sequence[str]]) -> list[str]:
    """One ``WARNING:`` line per shared device, then the remedy; empty when none."""
    if not shared:
        return []
    lines = [
        f"WARNING: device {device!r} is enabled in "
        + " and ".join(repr(e) for e in holders)
        for device, holders in shared.items()
    ]
    lines.append(
        f"WARNING: {len(shared)} device(s) are enabled in more than one live "
        "experiment — each is driven by every one of them; disable it in all "
        "but one (expt_device.enabled, a DB edit by its owner)"
    )
    return lines


def _db_errors() -> tuple[type[BaseException], ...]:
    """What reading the DB raises: no driver, no credentials/config, the driver's own."""
    errors: tuple[type[BaseException], ...] = (ImportError, OSError, KeyError)
    try:
        import mysql.connector
    except ImportError:
        return errors
    return (*errors, mysql.connector.Error)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the overlap report for the experiments on the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m geecs_core.db.overlap",
        description="List devices enabled in more than one of the given experiments (read-only).",
    )
    parser.add_argument("experiments", nargs="+", help="live experiments (DB names)")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 when any device is shared"
    )
    args = parser.parse_args(argv)
    try:
        shared = shared_devices(args.experiments)
    except _db_errors() as exc:  # "not checked", said so; a code bug still raises
        import getpass
        from pathlib import Path

        print(
            f"WARNING: overlap check skipped — the GEECS DB could not be read "
            f"as {getpass.getuser()!r} (its DB credentials come from "
            f"{Path('~/.config/geecs_python_api/config.ini').expanduser()}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3
    for line in format_report(shared):
        print(line)
    if not shared:
        print(
            f"overlap check: no device is enabled in more than one of {args.experiments}"
        )
    return 1 if (shared and args.strict) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
