"""Run supported v2 diagnostics on the analysis core behind the ScanAnalyzer contract."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from geecs_analysis.compat.v2 import UnsupportedRecipe, compile_v2
from geecs_analysis.compat.v2_run import UnitResult
from geecs_data_utils.shot_files import StackMappingUnavailable
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.base import DataUnavailableWarning, ScanAnalyzer
from scan_analysis.core_products import ProductPlan, plan_products
from scan_analysis.core_scan import PreparedScan, prepare_scan
from scan_analysis.core_sink import save_products
from scan_analysis.core_source import source_directory

logger = logging.getLogger(__name__)

__all__ = ["CoreScanAnalyzer", "core_supports"]


def core_supports(document: AnalysisDiagnostic) -> bool:
    """Whether the whole recipe compiles for the core route, without any reads.

    Scan-context backgrounds, unported analyzer kinds and unported processing
    operations all fail compilation with ``UnsupportedRecipe``; those recipes
    keep the legacy wrappers. File backgrounds are only declared here; the
    prepared run loads them later through data-utils.
    """
    try:
        compile_v2(document, allow_file_backgrounds=True)
    except UnsupportedRecipe:
        return False
    return True


class CoreScanAnalyzer(ScanAnalyzer):
    """Explicit scan execution of one v2 diagnostic on ``geecs_analysis``.

    Honors the contract the task queue, the portal and MCP already call:
    ``run_analysis(scan_tag)`` returns the display files (``None`` when the
    s-file is missing), raises ``DataUnavailableWarning`` when the device
    recorded nothing, persists scalars to the sidecar and the s-file, and
    ``cleanup()`` drops per-scan state. Scan-tag handling, s-file reading and
    scalar persistence are inherited unchanged from :class:`ScanAnalyzer`.

    Deliberate differences from the legacy wrappers: scalars are persisted
    before products are written, so a product write failure never loses
    them, and the output directory is created only when a product is saved.
    """

    def __init__(self, document: AnalysisDiagnostic, *, id: str, priority: int) -> None:
        super().__init__(device_name=document.name)
        self.document = document.model_copy(deep=True)
        self.id = id
        self.priority = priority
        # scalar_sidecar_path prefers ``id``; keep the legacy fallback name too.
        self._output_name = document.effective_output_name
        self.display_contents: list[str] = []
        #: The products chosen by the last run, for inspection and tests.
        self.last_plan: Optional[ProductPlan] = None

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        document = self.document
        scan_folder = Path(self.scan_directory)
        data_dir = source_directory(document, scan_folder)
        if not data_dir.is_dir() or not any(data_dir.iterdir()):
            raise DataUnavailableWarning(
                f"Data directory '{data_dir}' does not exist or is empty for "
                f"device '{self.device_name}'. Skipping analysis."
            )
        try:
            prepared = prepare_scan(document, scan_folder, self.auxiliary_data)
        except StackMappingUnavailable as exc:
            raise DataUnavailableWarning(str(exc)) from exc
        self.display_contents = []
        self.last_plan = None
        if not prepared.groups:
            logger.warning(
                "No data files mapped for %s; nothing to analyze.", self.device_name
            )
            return []
        outcomes = self._execute(prepared)
        # One legacy knob: ``waterfall_sort_key`` both requests the per-shot
        # waterfall and names the s-file column; the column resolves against
        # the rows refreshed by the s-file merge above, as the wrapper did.
        renderer = document.scan.renderer.as_kwargs()
        sort_key = renderer.get("waterfall_sort_key")
        plan = plan_products(
            prepared.prepared.recipe,
            outcomes,
            self.auxiliary_data,
            average_before_analysis=prepared.average_before_analysis,
            noscan=self.noscan,
            parameter_column=None if self.noscan else self.find_scan_param_column()[0],
            sort_requested=bool(sort_key),
            sort_column=self.find_column_for_key(sort_key) if sort_key else None,
            sort_bounds=renderer.get("waterfall_sort_bounds"),
            sort_sigma=renderer.get("waterfall_sort_sigma", 3.0),
        )
        if plan.summary and not self.noscan and not sort_key:
            # Figures name the scan by the cleaned ScanInfo string, not the
            # s-file column, exactly as the legacy renderers did.
            plan = replace(plan, position_label=self.scan_parameter or "")
        self.last_plan = plan
        saved = save_products(plan, prepared.prepared.recipe, document, scan_folder)
        for note in saved.notes:
            logger.warning("%s: %s", self.device_name, note)
        self.display_contents = [str(path) for path in saved.display_files]
        return list(self.display_contents)

    def _execute(self, prepared: PreparedScan) -> list[UnitResult]:
        """Stream the units, log failures and persist scalars before products."""
        outcomes: list[UnitResult] = []
        pending: list[dict] = []
        for outcome in prepared.run():
            for failure in outcome.load_failures:
                logger.warning(
                    "Skipping shot %s in unit %s (load failed: %s)",
                    failure.shot,
                    outcome.group.key,
                    failure.message,
                )
            if outcome.error is not None or outcome.measurement is None:
                logger.error(
                    "Analysis failed for unit %s: %s", outcome.group.key, outcome.error
                )
                continue
            for note in outcome.measurement.notes:
                logger.warning("Unit %s: %s", outcome.group.key, note)
            outcomes.append(outcome)
            pending.extend(prepared.scalar_records(outcome))
        if pending:
            updates = pd.DataFrame(pending)
            self.write_scalar_sidecar(updates)
            self.append_to_sfile(updates)
        return outcomes

    def cleanup(self) -> None:
        """Release the loaded s-file and the display list after a run."""
        self.auxiliary_data = None
        self.display_contents = []
        self.last_plan = None
        logger.debug("[CoreScanAnalyzer] cleanup() complete.")
