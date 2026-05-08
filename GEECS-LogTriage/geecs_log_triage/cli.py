r"""Command-line entry point: ``geecs-log-triage``.

Usage examples::

    # Whole day — writes triage.md next to scans/ automatically
    poetry run geecs-log-triage --date 2026-05-08 --experiment HTU

    # Single scan
    poetry run geecs-log-triage --date 2026-05-08 --experiment HTU --scan 42

    # Date range, JSON to stdout
    poetry run geecs-log-triage --date-range 2026-05-01:2026-05-08 \
        --experiment HTU --format json

    # Explicit scan folder (no date/experiment needed)
    poetry run geecs-log-triage --scan-folder /path/to/Scan037

    # Override output path
    poetry run geecs-log-triage --date 2026-05-08 --experiment HTU \
        --output /tmp/triage.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from geecs_log_triage.harvester import (
    day_folder_for,
    harvest_date,
    harvest_date_range,
    harvest_scan,
    harvest_scan_folder,
)
from geecs_log_triage.render import render_markdown
from geecs_log_triage.schemas import Severity, TriageReport


def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_date_range(s: str) -> tuple[date, date]:
    """Parse YYYY-MM-DD:YYYY-MM-DD."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--date-range expects 'YYYY-MM-DD:YYYY-MM-DD', got {s!r}"
        )
    a, b = s.split(":", 1)
    return _parse_date(a), _parse_date(b)


def _parse_severity(s: str) -> Severity:
    """Parse a severity name (case-insensitive)."""
    try:
        return Severity(s.upper())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid level {s!r}; choose one of {[lvl.value for lvl in Severity]}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the CLI."""
    p = argparse.ArgumentParser(
        prog="geecs-log-triage",
        description="Harvest and triage GEECS scan execution logs.",
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--date", type=_parse_date, help="Single date YYYY-MM-DD.")
    target.add_argument(
        "--date-range",
        dest="date_range",
        type=_parse_date_range,
        help="Date range start:end inclusive, both YYYY-MM-DD.",
    )
    target.add_argument(
        "--scan-folder",
        dest="scan_folder",
        type=Path,
        help="Path to a single scan folder (no date or experiment needed).",
    )

    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (required with --date / --date-range).",
    )
    p.add_argument(
        "--scan",
        type=int,
        default=None,
        metavar="N",
        help="Scan number to triage (only with --date). If omitted, all scans for the day.",
    )
    p.add_argument(
        "--level",
        type=_parse_severity,
        default=Severity.ERROR,
        help="Minimum severity to include (default: ERROR).",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        choices=["json", "md"],
        default="md",
        help="Output format: 'md' (default) or 'json'.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write report to this path. "
            "If omitted with --date/--experiment: auto-writes triage.md "
            "to the day folder. Otherwise prints to stdout."
        ),
    )
    return p


def _default_output_path(args: argparse.Namespace) -> Optional[Path]:
    """Return the auto-output path for --date + --experiment, or None."""
    if args.date is not None and args.experiment:
        return day_folder_for(args.date, args.experiment) / "triage.md"
    return None


def _emit(report: TriageReport, output: Optional[Path], fmt: str) -> None:
    """Write or print a triage report in the requested format."""
    if fmt == "md":
        payload = render_markdown(report)
    else:
        payload = report.model_dump_json(indent=2)

    if output is None:
        print(payload)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"Triage report written to: {output}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector. Defaults to ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on argument error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.scan_folder is not None:
        if args.scan is not None:
            parser.error("--scan cannot be used with --scan-folder")
        report = harvest_scan_folder(args.scan_folder, min_level=args.level)
    else:
        if not args.experiment:
            parser.error("--experiment is required when using --date or --date-range")
        if args.date is not None:
            if args.scan is not None:
                report = harvest_scan(
                    args.date, args.experiment, args.scan, min_level=args.level
                )
            else:
                report = harvest_date(args.date, args.experiment, min_level=args.level)
        else:
            if args.scan is not None:
                parser.error("--scan can only be used with --date, not --date-range")
            start, end = args.date_range
            report = harvest_date_range(
                start, end, args.experiment, min_level=args.level
            )

    output = args.output or _default_output_path(args)
    _emit(report, output, args.fmt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
