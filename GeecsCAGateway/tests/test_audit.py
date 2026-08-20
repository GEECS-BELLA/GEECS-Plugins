"""Tests for the read-only DB-hygiene audit (pure logic, no database).

The DB-facing wrapper (:func:`run_audit`) and CLI import ``GeecsDb`` / ``mysql``
lazily; these tests exercise only the pure classifier and the report/SQL
formatters, so nothing here touches the database or the network.
"""

from __future__ import annotations

from geecs_ca_gateway.audit import (
    GHOST,
    AuditResult,
    Finding,
    FkOrphanReport,
    audit_subscribed_variables,
    build_delete_sql,
    format_report,
)

# The gateway's real skip set (image / 1darray are non-scalar, not served).
SKIP_TYPES = {"image", "1darray"}


def _var(name: str, vartype: str) -> dict:
    """Build a minimal device-variable metadata dict (name + variabletype)."""
    return {"name": name, "variabletype": vartype}


def test_clean_device_has_no_findings() -> None:
    """A device whose subscribed vars are all real scalars yields nothing."""
    subscribed = {"U_Dev": ["current", "voltage"]}
    device_variables = {
        "U_Dev": [_var("current", "numeric"), _var("voltage", "numeric")]
    }
    assert audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES) == []


def test_ghost_variable_is_flagged() -> None:
    """A get='yes' name absent from the device's real variables is a GHOST."""
    subscribed = {"PicoscopeV2": ["charge_pC", "voltage"]}
    device_variables = {"PicoscopeV2": [_var("voltage", "numeric")]}
    findings = audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES)
    assert findings == [Finding("PicoscopeV2", "charge_pC", GHOST)]


def test_skip_variable_is_flagged_with_its_type() -> None:
    """A get='yes' variable typed image/1darray is a SKIP:<type> finding."""
    subscribed = {"UC_Cam": ["frame", "lineout", "exposure"]}
    device_variables = {
        "UC_Cam": [
            _var("frame", "image"),
            _var("lineout", "1darray"),
            _var("exposure", "numeric"),
        ]
    }
    findings = audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES)
    assert findings == [
        Finding("UC_Cam", "frame", "SKIP:image"),
        Finding("UC_Cam", "lineout", "SKIP:1darray"),
    ]


def test_device_missing_from_metadata_is_all_ghosts_not_a_crash() -> None:
    """A device absent from device_variables reports gracefully (all GHOST)."""
    subscribed = {"U_Gone": ["a", "b"]}
    device_variables: dict[str, list[dict]] = {}
    findings = audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES)
    assert findings == [
        Finding("U_Gone", "a", GHOST),
        Finding("U_Gone", "b", GHOST),
    ]


def test_empty_inputs_yield_no_findings() -> None:
    """Empty subscribed / device_variables produce an empty finding list."""
    assert audit_subscribed_variables({}, {}, SKIP_TYPES) == []


def test_mixed_devices_ordered_by_device_then_input_order() -> None:
    """Findings sort by device (sorted) then the input order of variables."""
    subscribed = {
        "B_Dev": ["ghost_b", "img_b"],
        "A_Dev": ["real_a", "ghost_a"],
    }
    device_variables = {
        "A_Dev": [_var("real_a", "numeric")],
        "B_Dev": [_var("img_b", "image")],
    }
    findings = audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES)
    assert findings == [
        Finding("A_Dev", "ghost_a", GHOST),
        Finding("B_Dev", "ghost_b", GHOST),
        Finding("B_Dev", "img_b", "SKIP:image"),
    ]


def test_choice_descriptor_rows_are_flagged_like_the_gateway_skips_them() -> None:
    """variabletype='choice' with a bare image/1darray descriptor is SKIP:<type>.

    The gateway's config builder resolves the effective type from ``choices``
    first (a bare descriptor word is authoritative), so these rows are skipped
    by the gateway and must be flagged by the audit too (#512).
    """
    subscribed = {"UC_Cam": ["frame", "trace", "mode"]}
    device_variables = {
        "UC_Cam": [
            {"name": "frame", "variabletype": "choice", "choices": "image"},
            {"name": "trace", "variabletype": "choice", "choices": "1darray"},
            {"name": "mode", "variabletype": "choice", "choices": "on,off"},
        ]
    }
    findings = audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES)
    assert findings == [
        Finding("UC_Cam", "frame", "SKIP:image"),
        Finding("UC_Cam", "trace", "SKIP:1darray"),
    ]


def test_scalar_choice_descriptors_are_not_flagged() -> None:
    """Bare numeric/string/path descriptors resolve to served scalar types."""
    subscribed = {"U_Dev": ["gain", "label", "savedir"]}
    device_variables = {
        "U_Dev": [
            {"name": "gain", "variabletype": "choice", "choices": "numeric"},
            {"name": "label", "variabletype": "choice", "choices": "string"},
            {"name": "savedir", "variabletype": "choice", "choices": "path"},
        ]
    }
    assert audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES) == []


def test_variabletype_case_is_normalized() -> None:
    """Skip matching is case-insensitive on the variabletype."""
    subscribed = {"U_Dev": ["frame"]}
    device_variables = {"U_Dev": [_var("frame", "IMAGE")]}
    findings = audit_subscribed_variables(subscribed, device_variables, SKIP_TYPES)
    assert findings == [Finding("U_Dev", "frame", "SKIP:image")]


def test_format_report_clean_says_no_problems() -> None:
    """A clean audit result renders a 'No problems found.' report."""
    result = AuditResult(experiment="Undulator")
    report = format_report(result)
    assert "No problems found." in report
    assert "Undulator" in report


def test_format_report_summarizes_counts_and_orphans() -> None:
    """The PROBLEMS summary counts ghosts, skips and FK-orphans separately."""
    result = AuditResult(
        experiment="Undulator",
        findings=[
            Finding("D", "ghost", GHOST),
            Finding("D", "img", "SKIP:image"),
        ],
        orphans=FkOrphanReport(count=1557, sample=[{"id": 1}]),
        subscribed={"D": ["ghost", "img", "ok"]},
    )
    report = format_report(result, full=True)
    assert "GHOST vars" in report
    assert "1557" in report
    # --full listing marks the clean variable OK and echoes the reasons.
    assert "[OK" in report
    assert "GHOST" in report


def test_build_delete_sql_is_review_only_and_scoped() -> None:
    """DELETE builder emits scoped ghost/skip deletes plus the orphan delete."""
    result = AuditResult(
        experiment="Undulator",
        findings=[Finding("PicoscopeV2", "charge_pC", GHOST)],
        orphans=FkOrphanReport(count=3),
    )
    statements = build_delete_sql("Undulator", result)
    assert len(statements) == 2
    assert "ed.expt = 'Undulator'" in statements[0]
    assert "ed.device = 'PicoscopeV2'" in statements[0]
    assert "edv.variablename = 'charge_pC'" in statements[0]
    assert "ed.id IS NULL" in statements[1]
