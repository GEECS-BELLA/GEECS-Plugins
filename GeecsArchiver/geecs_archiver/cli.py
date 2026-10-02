"""``geecs-archiver`` — the command line.

``onboard``   derive the experiment's archive set, diff it against the
              appliance, apply, then wait for the new PVs to connect
``list``      print the derived set (database only; no appliance needed)
``status``    the appliance's metrics and this experiment's PV counts
``export-config``  the appliance's own configuration snapshot as JSON

Exit status: 0 ok · 1 drift (a wanted PV the gateway does not serve, or a
request the appliance refused) · 2 usage (no experiment / URL, or a pause
larger than the guard without ``--yes``) · 3 the appliance did not answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from geecs_archiver import config
from geecs_archiver.archive_set import (
    ArchiveCandidate,
    Sampling,
    build_archive_set,
    sampling_for,
)
from geecs_archiver.mgmt_client import MgmtClient, MgmtError
from geecs_archiver.onboard import apply, plan_onboarding, stuck_requests, verify

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3

#: More pauses than this in one run need ``--yes`` — a half-empty device
#: table (DB maintenance, flipped ``enabled`` flags) must not silently pause
#: the experiment.
PAUSE_GUARD = 10


class UsageError(Exception):
    """A missing input the caller must supply (exit 2)."""


def _prefix(experiment: str) -> str:
    from geecs_core.pv_naming import pv_name

    return pv_name(experiment) + ":"


def _resolve_experiment(args: argparse.Namespace) -> str:
    experiment = args.experiment or config.experiment_name()
    if not experiment:
        raise UsageError(
            "no experiment: pass --experiment or set config.ini [Experiment] expt"
        )
    return experiment


def _resolve_url(args: argparse.Namespace) -> str:
    url = args.url or config.archiver_url()
    if not url:
        raise UsageError(
            "no appliance URL: pass --url, set GEECS_ARCHIVER_URL, or config.ini [archiver] url"
        )
    return url


def _desired(
    args: argparse.Namespace, experiment: str
) -> dict[str, tuple[ArchiveCandidate, Sampling]]:
    base = config.scanner_configs_base()
    policy_file = (
        Path(args.policy) if args.policy else config.policy_path(experiment, base)
    )
    derived_file = (
        Path(args.derived)
        if args.derived
        else config.derived_channels_path(experiment, base)
    )
    policy = config.load_policy(policy_file)
    derived = config.load_derived_channels(derived_file)
    candidates = build_archive_set(experiment, policy=policy, derived=derived)
    return {c.pv: (c, sampling_for(c.pv, policy)) for c in candidates}


def cmd_list(args: argparse.Namespace) -> int:
    """Print the derived archive set, one PV per line (``--json`` for the full records)."""
    experiment = _resolve_experiment(args)
    desired = _desired(args, experiment)
    if args.json:
        json.dump(
            [
                {
                    "pv": c.pv,
                    "device": c.device,
                    "variable": c.variable,
                    "kind": c.kind,
                    "dtype": c.dtype,
                    "samplingperiod": s.period,
                    "samplingmethod": s.method,
                }
                for c, s in desired.values()
            ],
            sys.stdout,
            indent=1,
        )
        print()
    else:
        for pv, (c, s) in desired.items():
            print(f"{pv:60s} {c.kind:9s} {c.dtype:7s} {s.period:g}s {s.method}")
        print(f"# {len(desired)} PVs", file=sys.stderr)
    return EXIT_OK


def cmd_onboard(args: argparse.Namespace) -> int:
    """Reconcile the appliance with the derived set; exit 1 on drift, 2 on an unguarded mass pause."""
    experiment = _resolve_experiment(args)
    url = _resolve_url(args)
    desired = _desired(args, experiment)
    prefix = _prefix(experiment)
    rc = EXIT_OK
    with MgmtClient(url) as client:
        archived = client.get_all_pvs()
        to_check = sorted(
            set(desired) | {pv for pv in archived if pv.lower().startswith(prefix)}
        )
        statuses = client.get_pv_status(to_check)
        plan = plan_onboarding(
            {pv: s for pv, (_, s) in desired.items()},
            statuses=statuses,
            archived_pvs=archived,
            prefix=prefix,
        )
        if args.no_pause:
            plan.to_pause.clear()
        print(f"{experiment}: {len(desired)} PVs wanted; plan: {plan.summary()}")
        if args.verbose or args.dry_run:
            for req in plan.to_archive:
                print(
                    f"  + {req['pv']}  ({req['samplingperiod']} s {req['samplingmethod']})"
                )
            for pv in plan.to_resume:
                print(f"  > resume {pv}")
            for pv, s in plan.to_retune:
                print(f"  ~ retune {pv} -> {s.period:g} s {s.method}")
            for pv in plan.to_pause:
                print(f"  - pause  {pv}")
        # The drift alarm, on EVERY run: wanted PVs stuck in the appliance's
        # archive-request workflow because they never answered on CA.
        stuck = stuck_requests(client, desired)
        for pv in stuck:
            print(
                f"  ! never connected: {pv}  (the rule wants it; the gateway does not serve it)"
            )
        if stuck:
            rc = EXIT_DRIFT
        if len(plan.to_pause) > PAUSE_GUARD and not args.yes:
            print(
                f"refusing to pause {len(plan.to_pause)} PVs (> {PAUSE_GUARD}) without --yes — "
                "is the database complete? (nothing was sent)",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if args.dry_run or plan.is_noop:
            return rc
        report = apply(client, plan)
        for rejected in report.rejected:
            print(f"  ! {rejected.get('pvName')}: {rejected.get('status')}")
            rc = EXIT_DRIFT
        new_pvs = [req["pv"] for req in plan.to_archive] + plan.to_resume
        if not new_pvs or args.wait <= 0:
            return rc
        print(
            f"waiting up to {args.wait:g} s for {len(new_pvs)} PV(s) to archive and connect …"
        )
        result = verify(client, new_pvs, wait_s=args.wait)
        print(
            f"  archived+connected {len(result.archived)}, never connected {len(result.never_connected)}, "
            f"pending {len(result.pending)}, other {len(result.other)}"
        )
        for pv in result.never_connected:
            print(
                f"  ! never connected: {pv}  (the rule wants it; the gateway does not serve it)"
            )
        for pv in result.pending:
            print(
                f"  ? still pending:   {pv}  (rerun later; it stays on the appliance's request list)"
            )
        for status in result.other:
            print(f"  ? {status.pv}: {status.status} connected={status.connected}")
        return rc if result.ok else EXIT_DRIFT


def cmd_status(args: argparse.Namespace) -> int:
    """The appliance's metrics, and this experiment's share of its PVs."""
    url = _resolve_url(args)
    experiment = args.experiment or config.experiment_name()
    with MgmtClient(url) as client:
        versions = client.versions()
        metrics = client.appliance_metrics()
        print(f"{url}: {versions.get('mgmt_version', '?')}")
        for key in (
            "pvCount",
            "connectedPVCount",
            "disconnectedPVCount",
            "eventRate",
            "dataRateGBPerDay",
            "dataRateGBPerYear",
            "status",
        ):
            if key in metrics:
                print(f"  {key:22s} {metrics[key]}")
        if experiment:
            prefix = _prefix(experiment)
            mine = [pv for pv in client.get_all_pvs() if pv.lower().startswith(prefix)]
            disconnected = [
                str(row.get("pvName"))
                for row in client.currently_disconnected()
                if str(row.get("pvName", "")).lower().startswith(prefix)
            ]
            stuck = [
                str(row.get("pvName"))
                for row in client.never_connected()
                if str(row.get("pvName", "")).lower().startswith(prefix)
            ]
            print(
                f"  {experiment}: {len(mine)} PVs under {prefix!r}, {len(disconnected)} disconnected, {len(stuck)} never connected"
            )
            for pv in sorted(disconnected)[:20]:
                print(f"    disconnected:    {pv}")
            for pv in sorted(stuck)[:20]:
                print(f"    never connected: {pv}")
    return EXIT_OK


def cmd_export_config(args: argparse.Namespace) -> int:
    """Write the appliance's ``exportConfig`` JSON (stdout or ``--out``)."""
    url = _resolve_url(args)
    with MgmtClient(url) as client:
        data = client.export_config()
    text = json.dumps(data, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(data)} PVs)", file=sys.stderr)
    else:
        print(text)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """The ``geecs-archiver`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="geecs-archiver",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, needs_url: bool) -> None:
        p.add_argument(
            "--experiment",
            help="GEECS experiment (default: config.ini [Experiment] expt)",
        )
        if needs_url:
            p.add_argument(
                "--url",
                help="appliance base URL, e.g. http://host:17665 (default: GEECS_ARCHIVER_URL or config.ini [archiver] url)",
            )

    def rule_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--policy",
            help="archive_policy.yaml (default: the experiment's file in the configs repo, if any)",
        )
        p.add_argument(
            "--derived",
            help="the gateway's derived_channels.yaml (default: the experiment's file, if any)",
        )

    p = sub.add_parser("list", help="print the derived archive set (database only)")
    common(p, needs_url=False)
    rule_args(p)
    p.add_argument("--json", action="store_true", help="full records as JSON")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("onboard", help="reconcile the appliance with the derived set")
    common(p, needs_url=True)
    rule_args(p)
    p.add_argument(
        "--dry-run", action="store_true", help="print the plan; send nothing"
    )
    p.add_argument(
        "--no-pause",
        action="store_true",
        help="never pause PVs the rule no longer wants",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help=f"allow pausing more than {PAUSE_GUARD} PVs in one run",
    )
    p.add_argument(
        "--wait",
        type=float,
        default=300.0,
        help="seconds to wait for new PVs to connect (0 = do not wait)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="list every planned change"
    )
    p.set_defaults(func=cmd_onboard)

    p = sub.add_parser(
        "status", help="appliance metrics and this experiment's PV counts"
    )
    common(p, needs_url=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "export-config",
        help="the appliance's configuration snapshot (importConfig restores it)",
    )
    common(p, needs_url=True)
    p.add_argument("--out", help="write here instead of stdout")
    p.set_defaults(func=cmd_export_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except UsageError as exc:
        print(f"geecs-archiver: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (ValidationError, FileNotFoundError) as exc:
        # A malformed or missing policy / derived-channels file: refuse to run,
        # as a usage error, never as a traceback with the drift exit code.
        print(
            f"geecs-archiver: cannot load the configuration overlay: {exc}",
            file=sys.stderr,
        )
        return EXIT_USAGE
    except MgmtError as exc:
        print(f"geecs-archiver: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
