"""Render a `TriageReport` to human-readable Markdown.

Intended for writing the per-day ``triage.md`` file that lives alongside the
``scans/`` directory so operators can open it without knowing the JSON schema.
"""

from __future__ import annotations

from datetime import timezone

from geecs_log_triage.schemas import Classification, TriageReport

_SECTION_ORDER = [
    Classification.BUG_CANDIDATE,
    Classification.HARDWARE_ISSUE,
    Classification.CONFIG_ISSUE,
    Classification.OPERATOR_ERROR,
    Classification.UNKNOWN,
]

_SECTION_LABEL = {
    Classification.BUG_CANDIDATE: "Bug candidates",
    Classification.HARDWARE_ISSUE: "Hardware issues",
    Classification.CONFIG_ISSUE: "Config / operator issues",
    Classification.OPERATOR_ERROR: "Operator actions",
    Classification.UNKNOWN: "Unclassified",
}


def render_markdown(report: TriageReport) -> str:
    """Render a `TriageReport` as a Markdown string.

    Parameters
    ----------
    report : TriageReport
        Triage report to render.

    Returns
    -------
    str
        Markdown text suitable for writing to a ``.md`` file.
    """
    lines: list[str] = []

    start, end = report.date_range
    date_str = str(start) if start == end else f"{start} – {end}"
    exp_str = report.experiment or "(folder scan)"
    generated = report.generated_at.astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    lines.append(f"# GEECS Scan Log Triage — {exp_str} — {date_str}")
    lines.append("")
    lines.append(
        f"Generated {generated} | "
        f"{report.total_scans_examined} scans examined | "
        f"{report.total_log_entries} log entries | "
        f"{report.total_errors} errors"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    if not report.grouped:
        lines.append("*No errors at or above the requested severity level.*")
        return "\n".join(lines)

    # Re-group occurrences by classification, then by fingerprint hash.
    by_class: dict[Classification, dict[str, list]] = {c: {} for c in _SECTION_ORDER}
    for fp_hash, occs in report.grouped.items():
        if not occs:
            continue
        klass = occs[0].fingerprint.classification
        bucket = by_class.get(klass, by_class[Classification.UNKNOWN])
        bucket[fp_hash] = occs

    for klass in _SECTION_ORDER:
        fps = by_class[klass]
        unique_count = len(fps)
        total_count = sum(len(v) for v in fps.values())
        label = _SECTION_LABEL[klass]

        lines.append(f"## {label} ({unique_count} unique, {total_count} occurrences)")
        lines.append("")

        if not fps:
            lines.append("*None.*")
            lines.append("")
            continue

        for fp_hash, group in fps.items():
            fp = group[0].fingerprint
            scan_ids = sorted({occ.scan_id for occ in group})
            n = len(group)

            title = fp.exception_type or fp.normalized_message[:80]
            lines.append(f"### `{title}`")
            lines.append("")
            lines.append(f"- **Fingerprint:** `{fp_hash}`")
            lines.append(f"- **Occurrences:** {n}")
            lines.append(f"- **Scans:** {', '.join(scan_ids)}")
            if fp.exception_type and fp.normalized_message:
                lines.append(f"- **Message:** {fp.normalized_message}")
            if fp.signature:
                lines.append(f"- **Signature:** `{fp.signature}`")
            lines.append("")

            if fp.sample_traceback:
                lines.append("<details>")
                lines.append("<summary>Sample traceback</summary>")
                lines.append("")
                lines.append("```")
                lines.append(fp.sample_traceback)
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)
