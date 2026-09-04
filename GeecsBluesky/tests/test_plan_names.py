"""Pin ``geecs_bluesky.plan_names`` to the plans it names (#793).

The readiness check asserts the manager lists ``GEECS_PLAN_NAMES`` after
``environment open``, and the startup profile exports exactly that list —
so every name here must be a real plan function in ``geecs_bluesky.plans``
and every exported plan must be named here.  The module itself must stay
import-light (the client and the readiness entry point both import it
without the engine).
"""

from __future__ import annotations

import subprocess
import sys

from geecs_bluesky import plan_names


def test_every_plan_name_is_a_real_plan_and_vice_versa() -> None:
    from geecs_bluesky.plans import named_plans, scan_request_plan

    exported = {
        name
        for module in (scan_request_plan, named_plans)
        for name in module.__all__
        if name.startswith("geecs_") and name.endswith("_plan")
    }
    assert set(plan_names.GEECS_PLAN_NAMES) == exported
    for name in plan_names.GEECS_PLAN_NAMES:
        module = named_plans if hasattr(named_plans, name) else scan_request_plan
        assert callable(getattr(module, name)), name
        assert getattr(module, name).__name__ == name


def test_names_are_unique_and_the_funnel_comes_first() -> None:
    assert len(set(plan_names.GEECS_PLAN_NAMES)) == len(plan_names.GEECS_PLAN_NAMES)
    assert plan_names.GEECS_PLAN_NAMES[0] == plan_names.SCAN_REQUEST_PLAN
    assert set(plan_names.GEECS_WORKER_FUNCTIONS) == {
        "geecs_move_variable",
        "geecs_describe_action",
    }


def test_module_imports_light() -> None:
    code = (
        "import sys; import geecs_bluesky.plan_names; "
        "heavy = [m for m in ('bluesky', 'ophyd_async', 'aioca', "
        "'bluesky_queueserver', 'bluesky_queueserver_api') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
