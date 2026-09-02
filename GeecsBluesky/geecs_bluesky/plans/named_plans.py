"""The three named scan plans — per-mode vocabulary at the queue gate.

Phase 2b-ii of ``Planning/schema_refactor/00_overview.md``: beside the
funnel (:func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`,
which takes one whole ``ScanRequest`` document) the worker registers one
plan per scan mode, each taking only that mode's fields:

- :func:`geecs_noscan_plan` — capture settings + actions (+ background);
- :func:`geecs_scan_plan` — axes (1-D or grid) + the shared components;
- :func:`geecs_optimize_plan` — the optimization spec (which carries its
  own variables and bounds) + the shared components.

They exist for the humans and the gates, not the machines: a generic
client renders a per-mode form (one published JSON Schema per parameter —
see ``geecs_schemas.schema_export.EXPORTED_SCHEMAS``), the manager's
``user_group_permissions`` can allow noscans to a group while gating
optimize, and the manager history names the kind of scan that ran.

**Shared-execution invariant (non-negotiable):** every named plan assembles
a canonical ``ScanRequest`` document and yields from the funnel — the same
preamble, validation, claim, run discipline, and run metadata — so the
document recorded in every start doc is a ``ScanRequest`` regardless of
entry plan, and every downstream reader sees one shape.  Nothing here
validates, resolves, or touches hardware; a bad parameter fails inside the
funnel's authoritative validation, pre-claim.

The funnel stays as the programmatic/compat entry point for GEECS-Console
and GEECS-MCP (the document API); these are wrappers, not a migration.

Signature constraint (queueserver contract, as on the funnel): the RE
Manager re-evaluates the annotation strings in a bare namespace at ``queue
add``, so parameters carry builtin annotations only and the worker-internal
seams stay unannotated.  Pinned by ``tests/test_named_plans.py``.
"""

from __future__ import annotations

from geecs_bluesky.plans.scan_request_plan import (
    SCAN_REQUEST_PLAN_ANNOTATION,
    geecs_scan_request_plan,
)

__all__ = [
    "geecs_noscan_plan",
    "geecs_scan_plan",
    "geecs_optimize_plan",
    "NOSCAN_PLAN_ANNOTATION",
    "SCAN_PLAN_ANNOTATION",
    "OPTIMIZE_PLAN_ANNOTATION",
]


def _request(mode: str, **fields: object) -> dict:
    """Assemble the canonical ScanRequest document for *mode*.

    ``None`` values are omitted so the schema's own default applies (an
    omitted ``actions`` is the empty bindings); the plans' other defaults
    (``description=""``, ``background=False``) equal the schema's, so a
    document assembled here validates to the same request the funnel would
    be handed for the same operator input.
    """
    document: dict = {"mode": mode}
    document.update({key: value for key, value in fields.items() if value is not None})
    return document


def geecs_noscan_plan(
    capture: dict,
    *,
    actions=None,
    description: str = "",
    background: bool = False,
    failed_move_policy=None,
    submission=None,
    session=None,
    resolver=None,
):
    """Take shots without moving anything — a ``noscan`` ScanRequest.

    Parameters
    ----------
    capture : dict
        ``CaptureSettings`` as JSON: shots, acquisition discipline, save
        sets, telemetry / native-image toggles, trigger profile.
    actions : dict, optional
        ``ActionBindings`` as JSON (setup / per_step / closeout plan names).
    description : str
        Free-text note for the scan metadata and the experiment log.
    background : bool
        Mark the shots as background/calibration data.
    failed_move_policy, submission, session, resolver :
        As on :func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`
        — the funnel's queue-settable surface, unchanged.

    Yields
    ------
    Msg
        The funnel plan's messages, unchanged.
    """
    request = _request(
        "noscan",
        capture=capture,
        actions=actions,
        description=description,
        background=background,
    )
    return (
        yield from geecs_scan_request_plan(
            request,
            submission=submission,
            session=session,
            resolver=resolver,
            failed_move_policy=failed_move_policy,
        )
    )


def geecs_scan_plan(
    axes: list,
    capture: dict,
    *,
    actions=None,
    description: str = "",
    background: bool = False,
    failed_move_policy=None,
    submission=None,
    session=None,
    resolver=None,
):
    """Sweep one axis (1-D) or several (a grid) — a ``step`` ScanRequest.

    Parameters
    ----------
    axes : list
        One or more ``ScanAxis`` objects as JSON; the first is the
        outermost (slowest) loop, the last the innermost.
    capture : dict
        ``CaptureSettings`` as JSON.
    actions : dict, optional
        ``ActionBindings`` as JSON.
    description : str
        Free-text note for the scan metadata and the experiment log.
    background : bool
        Mark the shots as background/calibration data.
    failed_move_policy, submission, session, resolver :
        As on :func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`
        — the funnel's queue-settable surface, unchanged.

    Yields
    ------
    Msg
        The funnel plan's messages, unchanged.
    """
    request = _request(
        "step",
        axes=axes,
        capture=capture,
        actions=actions,
        description=description,
        background=background,
    )
    return (
        yield from geecs_scan_request_plan(
            request,
            submission=submission,
            session=session,
            resolver=resolver,
            failed_move_policy=failed_move_policy,
        )
    )


def geecs_optimize_plan(
    optimization: dict,
    capture: dict,
    *,
    actions=None,
    description: str = "",
    background: bool = False,
    failed_move_policy=None,
    submission=None,
    session=None,
    resolver=None,
):
    """Let an algorithm pick the settings — an ``optimize`` ScanRequest.

    Parameters
    ----------
    optimization : dict
        ``OptimizationSpec`` as JSON: the variables the optimizer may move
        and their bounds (``variables: {name: [low, high]}``), objectives,
        evaluator, generator, iteration budget.  An optimize request has no
        ``axes`` — the search space lives here, as it always has (the
        optimization refactor decides whether bounds ever earn another
        shape).
    capture : dict
        ``CaptureSettings`` as JSON (shots per iteration, save sets, …).
    actions : dict, optional
        ``ActionBindings`` as JSON — recorded, not run (optimize has no
        action hooks yet; see the funnel).
    description : str
        Free-text note for the scan metadata and the experiment log.
    background : bool
        Mark the shots as background/calibration data.
    failed_move_policy, submission, session, resolver :
        As on :func:`~geecs_bluesky.plans.scan_request_plan.geecs_scan_request_plan`
        — the funnel's queue-settable surface, unchanged.

    Yields
    ------
    Msg
        The funnel plan's messages, unchanged.
    """
    request = _request(
        "optimize",
        optimization=optimization,
        capture=capture,
        actions=actions,
        description=description,
        background=background,
    )
    return (
        yield from geecs_scan_request_plan(
            request,
            submission=submission,
            session=session,
            resolver=resolver,
            failed_move_policy=failed_move_policy,
        )
    )


_ARTIFACT = "docs/geecs_schemas/{name}.schema.json in GEECS-Plugins"
_SHARED_PARAMETERS: dict = {
    "capture": {
        "description": (
            "CaptureSettings as a JSON object: shots per step, acquisition "
            "discipline, save sets, telemetry / native-image toggles, "
            "trigger profile. JSON Schema: " + _ARTIFACT.format(name="capture_settings")
        ),
        "annotation": "dict",
    },
    "actions": {
        "description": (
            "Optional ActionBindings as a JSON object (setup / per_step / "
            "closeout plan names). JSON Schema: "
            + _ARTIFACT.format(name="action_bindings")
        ),
    },
    "description": {
        "description": "Free-text note for the scan metadata and the experiment log.",
        "annotation": "str",
    },
    "failed_move_policy": {
        "description": (
            "What a failed scan-axis move does: 'pause' (the default — the "
            "RE Manager renders the paused state; resume/stop are queue "
            "verbs) or 'raise'. As on geecs_scan_request_plan."
        ),
    },
    # The funnel's own prose for its seams — one wording, imported.
    **{
        key: SCAN_REQUEST_PLAN_ANNOTATION["parameters"][key]
        for key in ("submission", "session", "resolver")
    },
}
_BACKGROUND_PARAMETER: dict = {
    "background": {
        "description": "Mark the shots as background/calibration data.",
        "annotation": "bool",
    }
}
_AXES_PARAMETER = {
    "description": (
        "One or more ScanAxis objects as JSON — the first axis is the outermost "
        "(slowest) loop. JSON Schema per element: " + _ARTIFACT.format(name="scan_axis")
    ),
    "annotation": "list",
}

#: ``parameter_annotation_decorator`` payloads (see the funnel's for the
#: constraints); the worker startup applies them so ``plans_allowed`` carries
#: typed, described parameters per plan.
NOSCAN_PLAN_ANNOTATION: dict = {
    "description": (
        "Take shots without moving anything (a noscan ScanRequest). Assembles "
        "the canonical ScanRequest and runs the same validated path as "
        "geecs_scan_request_plan."
    ),
    "parameters": {**_SHARED_PARAMETERS, **_BACKGROUND_PARAMETER},
}
SCAN_PLAN_ANNOTATION: dict = {
    "description": (
        "Sweep one axis (1-D) or several (a grid) — a step ScanRequest, run "
        "through the same validated path as geecs_scan_request_plan."
    ),
    "parameters": {
        "axes": _AXES_PARAMETER,
        **_SHARED_PARAMETERS,
        **_BACKGROUND_PARAMETER,
    },
}
OPTIMIZE_PLAN_ANNOTATION: dict = {
    "description": (
        "Let an algorithm pick the settings — an optimize ScanRequest, run "
        "through the same validated path as geecs_scan_request_plan. The "
        "optimization block carries the search space (variables and bounds), "
        "the objectives, the evaluator, and the generator."
    ),
    "parameters": {
        "optimization": {
            "description": (
                "OptimizationSpec as a JSON object: variables with bounds, "
                "objectives, evaluator, generator, iteration budget. JSON "
                "Schema: " + _ARTIFACT.format(name="optimization_spec")
            ),
            "annotation": "dict",
        },
        **_SHARED_PARAMETERS,
        **_BACKGROUND_PARAMETER,
    },
}
