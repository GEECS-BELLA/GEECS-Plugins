"""The detached analysis worker behind ``run_scan_analysis`` (#686).

Spawned by :func:`geecs_mcp.analysis.run_tools._spawn_worker` with one
JSON argv payload; runs the ScanAnalysis task queue for one scan and
exits.  All progress narration goes through the ``analysis_status/``
YAMLs (claim → heartbeat → done/failed/no_data — ``run_worklist``'s own
machinery), which the server's ``get_scan_analysis`` polls; stdio is
detached, so the status files are the one observable surface.

The statuses were already initialized server-side (visible queued rows
even if this process dies before claiming); this worker only builds the
worklist and runs it.  ``gdoc_enabled`` stays at its hard-off default.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def build_analyzers(analyzer: str | None, group: str | None, config_root: Path) -> list:
    """Construct the analyzer(s) for one request — THE shared builder.

    Called by the tool (validation: the names resolve AND the classes
    import on this host) and by the worker (the actual instances it
    runs) — one implementation, so the pre-flight validation can never
    drift from what the worker executes.

    Parameters
    ----------
    analyzer : str or None
        A diagnostic ID; exactly one of ``analyzer``/``group`` is set
        (the tool enforces this before calling).
    group : str or None
        An analysis-group name.
    config_root : Path
        The scan-analysis configs root (``analyzers/`` + ``groups/``).

    Returns
    -------
    list
        Runnable ScanAnalyzer instances, in execution order.
    """
    if analyzer:
        from image_analysis.config import load_diagnostic
        from scan_analysis.config import create_scan_analyzer

        return [
            create_scan_analyzer(
                load_diagnostic(analyzer, config_dir=config_root), id=analyzer
            )
        ]
    from scan_analysis.task_queue import load_analyzers_from_config

    return load_analyzers_from_config(group, config_dir=config_root)


def main(argv: list[str] | None = None) -> None:
    """Run the worklist described by the JSON payload in ``argv[0]``."""
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:] if argv is None else argv
    payload = json.loads(args[0])

    from geecs_data_utils import ScanTag

    tag = ScanTag(
        year=int(payload["year"]),
        month=int(payload["month"]),
        day=int(payload["day"]),
        number=int(payload["number"]),
        experiment=payload["experiment"],
    )
    root = Path(payload["config_root"])
    raw_base = payload.get("base_directory")
    base = Path(raw_base) if raw_base else None

    analyzers = build_analyzers(payload.get("analyzer"), payload.get("group"), root)

    from scan_analysis import task_queue

    worklist = task_queue.build_worklist(
        [tag],
        analyzers,
        base_directory=base,
        rerun_failed=bool(payload.get("rerun_failed")),
        rerun_completed=bool(payload.get("rerun_completed")),
    )
    task_queue.run_worklist(worklist, base_directory=base)


if __name__ == "__main__":
    main()
