"""The three named plans (Phase 2b-ii): thin wrappers over the funnel.

Hermetic pins of the shared-execution invariant at the seam — each named
plan assembles exactly the canonical ScanRequest document for its mode and
delegates to ``geecs_scan_request_plan`` with the worker seams passed
through — plus the queueserver signature/annotation contract (skipped
without the ``qserver`` extra, as the funnel's own pins are).  Document
parity on a real mock RunEngine lives in ``test_scan_request_plan.py``.
"""

from __future__ import annotations

import pytest

from geecs_bluesky.plans import named_plans
from geecs_bluesky.plans.named_plans import (
    NOSCAN_PLAN_ANNOTATION,
    OPTIMIZE_PLAN_ANNOTATION,
    SCAN_PLAN_ANNOTATION,
    geecs_noscan_plan,
    geecs_optimize_plan,
    geecs_scan_plan,
)
from geecs_schemas import ScanRequest

CAPTURE = {"shots_per_step": 3, "acquisition": "free_run", "save_sets": ["UC_Test"]}
AXIS = {"variable": "jet_z", "positions": {"start": 0.0, "end": 1.0, "step": 0.5}}
OPTIMIZATION = {
    "variables": {"jet_z": [0.0, 1.0]},
    "objectives": {"counts": "MAXIMIZE"},
    "evaluator": {"module": "m", "class": "C"},
    "generator": {"name": "bayes_default"},
    "max_iterations": 4,
}


def _recording_funnel(monkeypatch):
    """Replace the funnel with a recorder that yields once and returns a uid."""
    calls: list[tuple[dict, dict]] = []

    def fake_funnel(request, **kwargs):
        calls.append((request, kwargs))
        yield "funnel-message"
        return "uid-funnel"

    monkeypatch.setattr(named_plans, "geecs_scan_request_plan", fake_funnel)
    return calls


def _drive(plan):
    messages = []
    try:
        while True:
            messages.append(plan.send(None))
    except StopIteration as stop:
        return messages, stop.value


def test_noscan_plan_assembles_the_canonical_request(monkeypatch) -> None:
    calls = _recording_funnel(monkeypatch)
    messages, uid = _drive(
        geecs_noscan_plan(
            CAPTURE,
            actions={"setup": ["scan_prep"]},
            description="stats",
            background=True,
            submission={"client": "t", "preflight": []},
            session="S",
            resolver="R",
        )
    )
    assert messages == ["funnel-message"] and uid == "uid-funnel"
    (request, kwargs) = calls[0]
    assert request == {
        "mode": "noscan",
        "capture": CAPTURE,
        "actions": {"setup": ["scan_prep"]},
        "description": "stats",
        "background": True,
    }
    assert kwargs == {
        "submission": {"client": "t", "preflight": []},
        "session": "S",
        "resolver": "R",
    }
    # The assembled document IS a valid canonical ScanRequest.
    assert ScanRequest.model_validate(request).capture.shots_per_step == 3


def test_scan_plan_assembles_a_step_request_with_axes(monkeypatch) -> None:
    calls = _recording_funnel(monkeypatch)
    _drive(geecs_scan_plan([AXIS, {**AXIS, "variable": "jet_x"}], CAPTURE))
    (request, kwargs) = calls[0]
    assert request["mode"] == "step"
    assert [a["variable"] for a in request["axes"]] == ["jet_z", "jet_x"]
    assert "actions" not in request and request["background"] is False
    assert kwargs == {"submission": None, "session": None, "resolver": None}
    validated = ScanRequest.model_validate(request)
    assert validated.grid_shape() == (3, 3)


def test_optimize_plan_assembles_an_optimize_request(monkeypatch) -> None:
    calls = _recording_funnel(monkeypatch)
    _drive(geecs_optimize_plan(OPTIMIZATION, CAPTURE, description="opt"))
    (request, _kwargs) = calls[0]
    assert request["mode"] == "optimize"
    assert request["optimization"] == OPTIMIZATION
    assert "axes" not in request and "background" not in request
    validated = ScanRequest.model_validate(request)
    assert validated.optimization.max_iterations == 4


# ---------------------------------------------------------------------------
# Queueserver contract: builtin-only signatures, described parameters
# ---------------------------------------------------------------------------

_PLANS = [
    (geecs_noscan_plan, NOSCAN_PLAN_ANNOTATION, [CAPTURE]),
    (geecs_scan_plan, SCAN_PLAN_ANNOTATION, [[AXIS], CAPTURE]),
    (geecs_optimize_plan, OPTIMIZE_PLAN_ANNOTATION, [OPTIMIZATION, CAPTURE]),
]


@pytest.mark.parametrize(
    "plan,annotation,args", _PLANS, ids=[p[0].__name__ for p in _PLANS]
)
def test_named_plan_signature_passes_manager_validation(plan, annotation, args):
    """A real ``queue add`` item validates against each annotated signature,
    and ``plans_allowed`` carries a description for every annotated parameter
    (the funnel's own pins, applied per named plan)."""
    pytest.importorskip("bluesky_queueserver")
    from bluesky_queueserver import parameter_annotation_decorator
    from bluesky_queueserver.manager.profile_ops import _process_plan, validate_plan

    wrapped = parameter_annotation_decorator(annotation)(plan)
    processed = _process_plan(wrapped, existing_devices={}, existing_plans={})
    described = {p["name"]: p.get("description") for p in processed["parameters"]}
    assert set(annotation["parameters"]) <= set(described)
    assert all(described[name] for name in annotation["parameters"])
    ok, msg = validate_plan(
        {"name": plan.__name__, "args": args, "item_type": "plan"},
        allowed_plans={plan.__name__: processed},
        allowed_devices={},
    )
    assert ok, msg


def test_every_named_plan_points_at_a_published_artifact() -> None:
    """Each JSON-object parameter's description names the artifact a generic
    client grafts, and every named artifact is a registry entry (published)."""
    import re

    from geecs_schemas.schema_export import EXPORTED_SCHEMAS, artifact_path

    published = {artifact_path(name).as_posix() for name in EXPORTED_SCHEMAS}
    for annotation in (
        NOSCAN_PLAN_ANNOTATION,
        SCAN_PLAN_ANNOTATION,
        OPTIMIZE_PLAN_ANNOTATION,
    ):
        named = {
            m
            for p in annotation["parameters"].values()
            for m in re.findall(
                r"docs/geecs_schemas/\w+\.schema\.json", p["description"]
            )
        }
        assert named and named <= published, (named, published)
