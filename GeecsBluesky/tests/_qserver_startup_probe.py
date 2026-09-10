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

    import bluesky.plan_stubs as bps
    import bluesky.plans as bp
    from bluesky import RunEngine
    from bluesky_queueserver.manager.profile_ops import plans_from_nspace

    from geecs_bluesky.plan_names import GEECS_PLAN_NAMES
    from geecs_bluesky.preprocessors import connect_on_demand

    if not isinstance(ns.get("RE"), RunEngine):
        _fail(f"ns['RE'] is not a RunEngine: {ns.get('RE')!r}")
        return
    # The stock plans are bound by their own names — the manager discovers
    # every generator function in the namespace, so the discovered set must
    # be exactly the pinned list (a stray generator would become a plan).
    discovered = sorted(plans_from_nspace(ns))
    if discovered != sorted(GEECS_PLAN_NAMES):
        _fail(f"discovered plans {discovered!r} != GEECS_PLAN_NAMES")
        return
    for name in GEECS_PLAN_NAMES:
        real = getattr(bps if name == "mv" else bp, name)
        if ns[name] is not real:
            _fail(f"ns[{name!r}] is not bluesky's {name}")
            return
    funcs = [getattr(p, "func", p) for p in ns["RE"].preprocessors]
    if funcs[-1:] != [connect_on_demand] or funcs.count(connect_on_demand) != 1:
        _fail(f"connect_on_demand is not the outermost preprocessor: {funcs!r}")
        return
    if ns.get("__all__") != ["RE", *GEECS_PLAN_NAMES]:
        _fail(f"ns['__all__'] was {ns.get('__all__')!r}")
        return

    print("PROBE_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
