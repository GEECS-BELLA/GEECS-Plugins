"""Build a runnable scan-analyzer from any document of the analyzers tree.

An analysis document — the v3 :class:`~geecs_schemas.analysis.AnalysisRecipe`,
the analysis core's native shape — becomes a
:class:`~scan_analysis.core_analyzer.CoreScanAnalyzer`, the one route for
analyses: it compiles the document against the core's registry at
construction and runs it on ``geecs_analysis``. A ``.himg`` job (a
:class:`~geecs_schemas.analysis.HimgJob`, no analysis at all) goes straight
to its ScanAnalysis class in
:data:`~scan_analysis.analyzers.common.himg_kinds.HIMG_JOB_CLASSES`. Both
honour the same ``ScanAnalyzer`` contract, so the task queue, the portal and
MCP never see the difference. The legacy ImageAnalysis wrappers this factory
once fell back to were retired in #1039 (their last state: tag
``legacy-analysis-final``).

Pattern:

>>> from scan_analysis.config import create_scan_analyzer, load_diagnostic
>>> recipe = load_diagnostic("UC_VisaEBeam1")
>>> analyzer = create_scan_analyzer(recipe)
>>> analyzer.run_analysis(scan_tag)

For the group-loader path (production), the wrapping ``id`` and
``priority`` come from the group's :class:`ResolvedDiagnosticConfig`
rather than from the document's own ``scan.priority`` — pass them
explicitly via the keyword arguments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from geecs_schemas.analysis import AnalysisDocument, HimgJob

from scan_analysis.core_analyzer import CoreScanAnalyzer

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzer

logger = logging.getLogger(__name__)

__all__ = ["DestructiveKindRefused", "create_scan_analyzer"]


class DestructiveKindRefused(ValueError):
    """A job that deletes data files was asked for without the host's opt-in.

    ``destructive`` on the document marks such a job (the ``.himg``
    compaction, ``HimgJob.destructive``); it runs only where someone confirmed it — the data
    portal after its typed-scan-number check, or the shell — and every
    other host (the task queue, MCP, a script) gets this instead.
    """


def create_scan_analyzer(
    diag: AnalysisDocument,
    *,
    id: Optional[str] = None,
    priority: Optional[int] = None,
    allow_destructive: bool = False,
) -> "ScanAnalyzer":
    """Build a ScanAnalyzer from a validated analysis document.

    An analysis document becomes a :class:`CoreScanAnalyzer`; one that does
    not bind to the core's registry raises ``RecipeError`` here, one asking
    for what the core does not do ``UnsupportedRecipe``, before any scan is
    touched. A ``.himg`` job becomes its ScanAnalysis class. Both carry the
    runtime metadata the task queue needs.

    Parameters
    ----------
    diag : AnalysisRecipe or HimgJob
        Validated document.
    id : str, optional
        Task-queue ID for the analyzer instance. Defaults to
        ``diag.source_id`` when the document was loaded from a YAML
        file, otherwise ``diag.device``. The group loader passes the
        filename stem here explicitly (which may differ from the device).
    priority : int, optional
        Effective execution priority. Defaults to the document's own
        ``scan.priority``. The group loader passes the
        per-group-overridden value when present.
    allow_destructive : bool, default=False
        The host's opt-in for a kind that deletes data files
        (``HimgJob.destructive``, the ``.himg`` compaction).
        Without it such a document raises :class:`DestructiveKindRefused`
        before anything is built — the task queue and MCP never pass it;
        the data portal passes it after its typed-scan-number check.

    Returns
    -------
    ScanAnalyzer
        A ``CoreScanAnalyzer`` carrying ``id`` / ``priority``, or the
        ``.himg`` job's analyzer with ``id`` / ``priority`` /
        ``background_source`` attached as instance attributes.
    """
    source_id = getattr(diag, "source_id", None)
    effective_priority = priority if priority is not None else diag.scan.priority

    if getattr(diag, "destructive", False) and not allow_destructive:
        # The gate every host passes through: a kind that deletes data
        # files is built only for a caller that says it confirmed the run.
        raise DestructiveKindRefused(
            f"{source_id or diag.device}: "
            f"{getattr(diag, 'kind', 'this job')!r} deletes data files and runs only on an "
            "explicit, confirmed request (the data portal's typed scan number, "
            "or the shell) — never from the task queue or an unconfirmed host"
        )
    if isinstance(diag, HimgJob):
        from scan_analysis.analyzers.common.himg_kinds import HIMG_JOB_CLASSES

        analyzer = HIMG_JOB_CLASSES[diag.kind](device_name=diag.device)
        analyzer.id = id if id is not None else source_id or diag.device
        analyzer.priority = effective_priority
        analyzer.background_source = None
        logger.debug("Routing .himg job %r to ScanAnalysis", analyzer.id)
        return analyzer
    effective_id = id if id is not None else source_id or diag.device
    logger.debug("Routing document %r to the analysis core", effective_id)
    return CoreScanAnalyzer(diag, id=effective_id, priority=effective_priority)
