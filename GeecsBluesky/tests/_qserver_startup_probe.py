"""Subprocess probe: qserver/startup/startup.py's import order + namespace shape.

Run as ``python _qserver_startup_probe.py <startup_path>`` in a **fresh**
interpreter — that is the whole point. ``runpy.run_path`` executed
in-process (inside the pytest session) can never actually exercise the
module docstring's "import order is load-bearing" claim: by the time any
test function runs, ``geecs_bluesky`` and ``aioca`` are already sitting in
``sys.modules`` from earlier test-file collection, so the ordering
``qserver/startup/startup.py`` itself performs is untestable in-process —
the cache hides it either way. A subprocess starts with an empty
``sys.modules`` (PR #644 review row 8), so a ``sys.meta_path`` finder can
watch the *live* first-import order and catch a real regression (e.g. some
future import creeping in ahead of ``import geecs_bluesky`` that pulls in
``aioca`` before ``EPICS_CA_ADDR_LIST`` is set — see
``geecs_bluesky/epics_env.py``).

Prints ``PROBE_OK`` and exits 0 on success; prints ``FAIL: <reason>`` and
exits 1 otherwise. Never raises past ``main()`` — every failure mode is a
reported string, so the parent test gets a legible reason instead of a
subprocess traceback dump.
"""

from __future__ import annotations

import os
import runpy
import sys


def _fail(reason: str) -> None:
    print(f"FAIL: {reason}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        _fail(f"usage: {sys.argv[0]} <startup_path>")
        return
    startup_path = sys.argv[1]

    import_order: list[str] = []
    epics_addr_at_aioca_import: dict[str, str | None] = {}

    class _RecordingFinder:
        """Records first-import order of the two names that matter here.

        Declines every lookup (returns ``None``) so the real finders run
        unmodified — this only observes, it never changes what imports.
        """

        def find_spec(self, fullname, path, target=None):
            top = fullname.split(".", 1)[0]
            if top in ("geecs_bluesky", "aioca") and top not in import_order:
                import_order.append(top)
                if top == "aioca":
                    epics_addr_at_aioca_import["EPICS_CA_ADDR_LIST"] = os.environ.get(
                        "EPICS_CA_ADDR_LIST"
                    )
            return None

    sys.meta_path.insert(0, _RecordingFinder())

    try:
        ns = runpy.run_path(startup_path, run_name="__not_main__")
    except Exception as exc:
        _fail(f"startup.py raised {exc!r}")
        return

    if import_order != ["geecs_bluesky", "aioca"]:
        _fail(
            f"import order was {import_order!r}, expected "
            "['geecs_bluesky', 'aioca'] — aioca must not be importable "
            "before geecs_bluesky sets EPICS_CA_ADDR_LIST"
        )
        return

    if not epics_addr_at_aioca_import.get("EPICS_CA_ADDR_LIST"):
        _fail(
            "EPICS_CA_ADDR_LIST was not set in the environment by the time "
            "aioca was first imported — apply_epics_address_config() ran "
            "too late (or not at all)"
        )
        return

    from bluesky import RunEngine

    from geecs_bluesky.plans.named_plans import (
        geecs_noscan_plan,
        geecs_optimize_plan,
        geecs_scan_plan,
    )
    from geecs_bluesky.plans.scan_request_plan import (
        geecs_run_action_plan,
        geecs_scan_request_plan,
    )

    if not isinstance(ns.get("RE"), RunEngine):
        _fail(f"ns['RE'] is not a RunEngine: {ns.get('RE')!r}")
        return
    if ns["RE"] is not ns["session"].RE:
        _fail("ns['RE'] is not ns['session'].RE — --keep-re contract broken")
        return
    # The startup rebinds both plan names to parameter_annotation_decorator
    # wrappers (#727) — the namespace entry must be an annotated wrapper
    # around the real plan function, not the bare plan and not a stranger.
    for plan_name, real_plan in (
        ("geecs_scan_request_plan", geecs_scan_request_plan),
        ("geecs_run_action_plan", geecs_run_action_plan),
        ("geecs_noscan_plan", geecs_noscan_plan),
        ("geecs_scan_plan", geecs_scan_plan),
        ("geecs_optimize_plan", geecs_optimize_plan),
    ):
        wrapped = ns.get(plan_name)
        if getattr(wrapped, "__wrapped__", None) is not real_plan:
            _fail(f"ns[{plan_name!r}] does not wrap the real plan function")
            return
        if not getattr(wrapped, "_custom_parameter_annotation_", None):
            _fail(f"ns[{plan_name!r}] carries no queueserver parameter annotation")
            return
    for verb in ("geecs_move_variable", "geecs_describe_action"):
        if not callable(ns.get(verb)):
            _fail(f"ns[{verb!r}] is not callable — function_execute verb missing")
            return
    if ns.get("__all__") != [
        "RE",
        "geecs_scan_request_plan",
        "geecs_run_action_plan",
        "geecs_noscan_plan",
        "geecs_scan_plan",
        "geecs_optimize_plan",
        "geecs_move_variable",
        "geecs_describe_action",
    ]:
        _fail(f"ns['__all__'] was {ns.get('__all__')!r}")
        return

    print("PROBE_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
