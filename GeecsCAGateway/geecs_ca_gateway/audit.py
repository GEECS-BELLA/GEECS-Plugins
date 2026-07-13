"""Read-only DB-hygiene audit for the GEECS experiment database.

The LabVIEW DB-editing GUIs do not cascade deletions (there are no foreign-key
constraints on ``expt_device_variable``), so the table accumulates ``get='yes'``
rows that no longer map to anything the gateway can serve.  Each such row makes
the CA gateway / scanner try to connect a PV that will never exist, which shows
up as a per-device stall.  This module reports them so a human can clean the DB.

Three flavours are flagged:

GHOST
    A ``get='yes'`` row whose ``variablename`` is not a real variable of the
    device (a deleted variable, or an alias-derived name).  The gateway serves
    PVs by *real* variablename, so a ghost names a PV that is never created.
SKIP:<type>
    A ``get='yes'`` variable whose *effective* type is in the gateway's
    ``_SKIP_VARTYPES`` (``image`` / ``1darray``).  These are not scalar CA data,
    so no PV is served for them.  Effective types come from
    :func:`geecs_ca_gateway.config.effective_vartype` — the same rules the
    gateway's config builder applies.
FK-orphan
    An ``expt_device_variable`` row whose ``expt_device_id`` points at an
    ``expt_device.id`` that no longer exists (a whole experiment removed).

The pure classification logic (:func:`audit_subscribed_variables`) has **zero**
database imports and is fully unit-testable over plain dicts.  The DB-facing
:func:`run_audit` and the CLI import :class:`GeecsDb` / ``mysql`` lazily so this
module imports with no database on the machine.

**This tool is strictly read-only.**  It never DELETEs or modifies a row.  The
optional ``--sql`` mode only *prints* the DELETE statements a human could
review and run themselves; it never executes them.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

from .config import _SKIP_VARTYPES, effective_vartype

logger = logging.getLogger("geecs_ca_gateway.audit")

#: Reason tag for a subscribed variable that is not a real device variable.
GHOST = "GHOST"


def _skip_reason(vartype: str) -> str:
    """Return the ``SKIP:<type>`` reason tag for a skipped variable type."""
    return f"SKIP:{vartype}"


@dataclass(frozen=True)
class Finding:
    """One ``get='yes'`` variable the gateway will not serve.

    Attributes
    ----------
    device:
        GEECS device name.
    variable:
        The subscribed ``variablename`` that is a problem.
    reason:
        Why it will not be served — ``"GHOST"`` (name is not a real variable of
        the device) or ``"SKIP:<type>"`` (variabletype is a non-scalar type the
        gateway skips, e.g. ``"SKIP:image"`` / ``"SKIP:1darray"``).
    """

    device: str
    variable: str
    reason: str


@dataclass
class FkOrphanReport:
    """FK-orphan summary: rows whose ``expt_device_id`` has no ``expt_device``.

    Attributes
    ----------
    count:
        Total number of orphaned ``expt_device_variable`` rows.
    sample:
        A capped sample of orphan rows, each a dict with ``id``,
        ``expt_device_id`` and ``variablename`` keys, for eyeballing.
    """

    count: int = 0
    sample: list[dict] = field(default_factory=list)


@dataclass
class AuditResult:
    """The full audit outcome for one experiment.

    Attributes
    ----------
    experiment:
        GEECS experiment name audited.
    findings:
        GHOST / SKIP findings from :func:`audit_subscribed_variables`.
    orphans:
        FK-orphan summary (global — orphans are not experiment-scoped because
        their owning ``expt_device`` is gone).
    subscribed:
        The ``{device: [variablename, ...]}`` monitoring set that was audited,
        kept for the ``--full`` per-device listing.
    """

    experiment: str
    findings: list[Finding] = field(default_factory=list)
    orphans: FkOrphanReport = field(default_factory=FkOrphanReport)
    subscribed: dict[str, list[str]] = field(default_factory=dict)


def audit_subscribed_variables(
    subscribed: dict[str, list[str]],
    device_variables: dict[str, list[dict]],
    skip_types: set[str],
) -> list[Finding]:
    """Classify every subscribed variable the gateway will not serve.

    Pure function — no database, no network.  Cross-checks each subscribed
    (``get='yes'``) variablename against the device's real variable list and
    returns a :class:`Finding` for each one that is a ghost or a skipped type.

    Parameters
    ----------
    subscribed:
        ``{device: [variablename, ...]}`` — the ``get='yes'`` monitoring set,
        e.g. from :meth:`GeecsDb.get_subscribed_variables`.
    device_variables:
        ``{device: [metadata, ...]}`` where each metadata dict has at least a
        ``name`` and a ``variabletype`` key, plus ``choices`` when present (the
        shape of :meth:`GeecsDb.get_device_variables` /
        :meth:`GeecsDb.get_experiment_device_variables`).  A device absent from
        this mapping has no known real variables, so every subscribed variable
        for it is reported as a GHOST rather than crashing.
    skip_types:
        Variable types the gateway does not serve (the gateway's
        ``_SKIP_VARTYPES``, e.g. ``{"image", "1darray"}``).  Matched against
        the effective type from
        :func:`geecs_ca_gateway.config.effective_vartype`.

    Returns
    -------
    list[Finding]
        One finding per problem variable, ordered by device (sorted) then by
        the input order of that device's subscribed variables.  A clean
        variable produces no finding.
    """
    skip = {str(t).strip().lower() for t in skip_types}
    findings: list[Finding] = []
    for device in sorted(subscribed):
        real = device_variables.get(device)
        by_name: dict[str, str] = {}
        if real:
            for meta in real:
                name = meta.get("name")
                if name is not None:
                    by_name[name] = effective_vartype(
                        meta.get("variabletype"), meta.get("choices")
                    )
        for variable in subscribed[device]:
            if variable not in by_name:
                findings.append(Finding(device, variable, GHOST))
                continue
            vartype = by_name[variable]
            if vartype in skip:
                findings.append(Finding(device, variable, _skip_reason(vartype)))
    return findings


def build_delete_sql(experiment: str, result: AuditResult) -> list[str]:
    """Return DELETE statements a human could review to remove the flagged rows.

    Pure string builder — it returns SQL for human review and never executes
    anything.  Ghost/skip deletes are scoped by experiment + device +
    variablename + ``get='yes'``; the FK-orphan delete removes rows whose
    ``expt_device`` is gone (global, since they are not experiment-scoped).

    Parameters
    ----------
    experiment:
        GEECS experiment name (used to scope the ghost/skip deletes).
    result:
        The audit result whose findings/orphans to emit deletes for.

    Returns
    -------
    list[str]
        One SQL statement per line, safe to hand to a DBA — nothing is run here.
    """

    def esc(value: str) -> str:
        return value.replace("'", "''")

    statements: list[str] = []
    for finding in result.findings:
        statements.append(
            "DELETE edv FROM expt_device_variable edv "
            "JOIN expt_device ed ON ed.id = edv.expt_device_id "
            f"WHERE ed.expt = '{esc(experiment)}' "
            f"AND ed.device = '{esc(finding.device)}' "
            f"AND edv.variablename = '{esc(finding.variable)}' "
            f"AND edv.get = 'yes';  -- {finding.reason}"
        )
    if result.orphans.count:
        statements.append(
            "DELETE edv FROM expt_device_variable edv "
            "LEFT JOIN expt_device ed ON ed.id = edv.expt_device_id "
            f"WHERE ed.id IS NULL;  -- {result.orphans.count} FK-orphan row(s)"
        )
    return statements


def format_report(result: AuditResult, *, full: bool = False) -> str:
    """Render *result* as a clean text report (a PROBLEMS summary, then detail).

    Parameters
    ----------
    result:
        The audit outcome to render.
    full:
        When true, append the complete per-device subscribed-variable listing
        (each variable marked OK / GHOST / SKIP:<type>).

    Returns
    -------
    str
        The formatted, printable report.
    """
    lines: list[str] = []
    lines.append(f"DB-hygiene audit for experiment {result.experiment!r}")
    lines.append("=" * 60)

    ghosts = [f for f in result.findings if f.reason == GHOST]
    skips = [f for f in result.findings if f.reason != GHOST]

    lines.append("")
    lines.append("PROBLEMS")
    lines.append("-" * 60)
    lines.append(f"  GHOST vars (get='yes', not a real device variable): {len(ghosts)}")
    lines.append(f"  SKIP vars  (get='yes', non-scalar type not served): {len(skips)}")
    lines.append(
        f"  FK-orphan rows (expt_device_id with no expt_device): {result.orphans.count}"
    )

    if result.findings:
        lines.append("")
        lines.append("  Flagged variables:")
        for finding in result.findings:
            lines.append(
                f"    [{finding.reason:<12}] {finding.device} :: {finding.variable}"
            )

    if result.orphans.sample:
        lines.append("")
        lines.append(
            f"  FK-orphan sample (first {len(result.orphans.sample)} of "
            f"{result.orphans.count}):"
        )
        for row in result.orphans.sample:
            lines.append(
                f"    id={row.get('id')} "
                f"expt_device_id={row.get('expt_device_id')} "
                f"variablename={row.get('variablename')!r}"
            )

    if not result.findings and not result.orphans.count:
        lines.append("")
        lines.append("  No problems found.")

    if full:
        lines.append("")
        lines.append("FULL PER-DEVICE LISTING")
        lines.append("-" * 60)
        flagged = {(f.device, f.variable): f.reason for f in result.findings}
        for device in sorted(result.subscribed):
            lines.append(f"  {device}")
            for variable in result.subscribed[device]:
                reason = flagged.get((device, variable), "OK")
                lines.append(f"    [{reason:<12}] {variable}")

    return "\n".join(lines)


def run_audit(
    experiment: str,
    *,
    enabled_only: bool = True,
    skip_types: set[str] | None = None,
) -> AuditResult:
    """Run the full read-only audit against the live GEECS database.

    Pulls the ``get='yes'`` monitoring set and per-device variable metadata,
    runs the pure :func:`audit_subscribed_variables`, and adds the FK-orphan
    count + sample.  ``GeecsDb`` (and thereby ``mysql``) is imported lazily so
    this module imports on machines with no database.

    Parameters
    ----------
    experiment:
        GEECS experiment name.
    enabled_only:
        Restrict the subscribed set + device metadata to enabled devices
        (default true, matching the gateway's default served set).
    skip_types:
        Variable types treated as non-servable; defaults to the gateway's
        ``_SKIP_VARTYPES`` from :mod:`geecs_ca_gateway.config`.

    Returns
    -------
    AuditResult
        Findings, FK-orphan summary, and the audited subscribed set.
    """
    from .db.geecs_db import GeecsDb

    skip = set(_SKIP_VARTYPES if skip_types is None else skip_types)

    subscribed = GeecsDb.get_subscribed_variables(experiment, enabled_only=enabled_only)
    device_variables = GeecsDb.get_experiment_device_variables(
        experiment, enabled_only=enabled_only
    )
    findings = audit_subscribed_variables(subscribed, device_variables, skip)
    orphan_dict = GeecsDb.get_fk_orphan_variables()
    orphans = FkOrphanReport(
        count=int(orphan_dict.get("count", 0)),
        sample=list(orphan_dict.get("sample", [])),
    )
    return AuditResult(
        experiment=experiment,
        findings=findings,
        orphans=orphans,
        subscribed=subscribed,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="geecs-ca-gateway-audit",
        description=(
            "Read-only DB-hygiene audit: report every get='yes' variable the "
            "gateway will NOT serve (ghosts / skipped types) plus FK-orphans. "
            "Never modifies the database."
        ),
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="GEECS experiment name, e.g. Undulator.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include devices not enabled in the experiment.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also print the complete per-device subscribed-variable listing.",
    )
    parser.add_argument(
        "--sql",
        action="store_true",
        help=(
            "Print (do NOT execute) the DELETE statements a human could review "
            "to remove the flagged rows. The audit never modifies the DB."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (default WARNING — keep the report clean).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m geecs_ca_gateway.audit --experiment NAME``."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_audit(args.experiment, enabled_only=not args.include_disabled)
    print(format_report(result, full=args.full))
    if args.sql:
        statements = build_delete_sql(args.experiment, result)
        print("")
        print("-- REVIEW-ONLY SQL (not executed) " + "-" * 26)
        if statements:
            for statement in statements:
                print(statement)
        else:
            print("-- nothing to delete")


if __name__ == "__main__":
    main()
