"""Run analysis recipes (v3) and supported v2 diagnostics on the core behind the ScanAnalyzer contract."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from geecs_analysis.compat.v2 import UnsupportedRecipe, compile_v2
from geecs_analysis.registry import measure_definition
from geecs_data_utils.shot_files import StackMappingUnavailable
from geecs_schemas.analysis import AnalysisRecipe, WaterfallSummary

from scan_analysis.base import DataUnavailableWarning, ScanAnalyzer
from scan_analysis.core_products import ProductCollector, ProductPlan
from scan_analysis.core_recipe import AnalysisDocument, ScanRecipe, scan_recipe
from scan_analysis.core_scan import PreparedScan, prepare_scan
from scan_analysis.core_sink import (
    ShotStore,
    save_products,
    shot_store_path,
    write_shot_table,
)
from scan_analysis.core_source import source_directory
from scan_analysis.core_workers import effective_workers

logger = logging.getLogger(__name__)

__all__ = ["CoreScanAnalyzer", "core_supports", "write_scalars_into_rows"]


def write_scalars_into_rows(rows: pd.DataFrame, records: list[dict]) -> None:
    """Write each record's scalars, in place, into the rows of its shot.

    The outcome of assigning cell by cell in record order — a later record
    for the same shot wins, a key a record lacks leaves that shot's cell
    alone, a new column is NaN on every other row, and a key whose shots
    match no row still creates its column — written as one aligned
    assignment per column. Cell by cell, a 3600-shot run's 18 scalars took
    ~12 s against a 187-column s-file (worker host, 2026-09-26); this takes tens
    of milliseconds.

    Parameters
    ----------
    rows : pandas.DataFrame
        The scan's s-file rows, keyed by a ``Shotnumber`` column; mutated.
    records : list of dict
        One mapping per shot: ``Shotnumber`` plus the scalars to write.
    """
    shots = rows["Shotnumber"]
    keys = dict.fromkeys(k for r in records for k in r if k != "Shotnumber")
    for key in keys:
        given = {r["Shotnumber"]: r[key] for r in records if key in r}
        hit = shots.isin(list(given))
        if hit.any():
            rows.loc[hit, key] = shots[hit].map(given)
        else:
            # An empty assignment still creates the column, as the cell
            # loop did, with the dtype that value gives it.
            rows.loc[hit, key] = next(iter(given.values()))


def core_supports(document: AnalysisDocument) -> bool:
    """Whether the recipe runs on the core route, decided without any reads.

    An analysis recipe (v3) always does: it has no other route, and one that
    does not bind to the registry fails when the analyzer is built. For a v2
    diagnostic, scan-context backgrounds, unported analyzer kinds and
    unported processing operations all fail compilation with
    ``UnsupportedRecipe``; those recipes keep the legacy wrappers. File
    backgrounds are only declared here; the prepared run loads them later
    through data-utils.
    """
    if isinstance(document, AnalysisRecipe):
        return True
    try:
        compile_v2(document, allow_file_backgrounds=True)
    except UnsupportedRecipe:
        return False
    return True


class CoreScanAnalyzer(ScanAnalyzer):
    """Explicit scan execution of one analysis document on ``geecs_analysis``.

    Honors the contract the task queue, the portal and MCP already call:
    ``run_analysis(scan_tag)`` returns the display files (``None`` when the
    s-file is missing), raises ``DataUnavailableWarning`` when the device
    recorded nothing, persists scalars to the sidecar and the s-file, and
    ``cleanup()`` drops per-scan state. Scan-tag handling, s-file reading and
    scalar persistence are inherited unchanged from :class:`ScanAnalyzer`.

    Deliberate differences from the legacy wrappers: scalars are persisted
    before products are written, so a product write failure — or a result
    the products cannot fold, raised after the persist — never loses them,
    and the output directory is created only when a product is saved.

    The run streams: each outcome's scalars are queued and its measurement
    folded into the products (``ProductCollector``) as it arrives, so a
    camera scan holds one running frame per product however long it is.
    The recipe's ``scan.workers`` asks for a process pool; the count it gets
    is ``core_workers.effective_workers`` — capped by :attr:`worker_cap`
    (a host's override; ``None`` reads the client ``config.ini``) and serial
    for a small run — and is logged with the unit count. Outputs do not
    depend on the count.
    """

    #: A host's cap on the workers any run gets (the portal, the task
    #: queue); ``None`` resolves ``core_workers.host_worker_cap`` at run time.
    worker_cap: Optional[int] = None

    def __init__(self, document: AnalysisDocument, *, id: str, priority: int) -> None:
        self.document = document.model_copy(deep=True)
        # Compiling here surfaces a recipe that does not bind to the registry
        # at construction, before any scan is touched.
        self.spec: ScanRecipe = scan_recipe(self.document)
        super().__init__(device_name=self.spec.device)
        self.id = id
        self.priority = priority
        # scalar_sidecar_path prefers ``id``; keep the legacy fallback name too.
        self._output_name = self.spec.output_name
        self.display_contents: list[str] = []
        #: The products chosen by the last run, for inspection and tests.
        self.last_plan: Optional[ProductPlan] = None

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        document = self.document
        scan_folder = Path(self.scan_directory)
        data_dir = source_directory(self.spec, scan_folder)
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
        # One legacy knob: the waterfall's ``sort_key`` both requests the
        # per-shot waterfall and names the s-file column; the column resolves
        # against the rows refreshed by the s-file merge below, as the wrapper
        # did. The first waterfall summary carries it.
        stack = next(
            (s for s in self.spec.summaries if isinstance(s, WaterfallSummary)), None
        )
        sort_key = stack.sort_key if stack is not None else None
        collector = ProductCollector(
            prepared.prepared.recipe,
            self.auxiliary_data,
            average_before_analysis=prepared.average_before_analysis,
            noscan=self.noscan,
            sort_requested=bool(sort_key),
        )
        self._execute(prepared, collector)
        plan = collector.plan(
            self.auxiliary_data,
            parameter_column=None if self.noscan else self.find_scan_param_column()[0],
            sort_column=self.find_column_for_key(sort_key) if sort_key else None,
            sort_bounds=stack.sort_bounds if stack is not None else None,
            sort_sigma=stack.sort_sigma if stack is not None else 3.0,
        )
        if (plan.singles or plan.summary) and not self.noscan and not sort_key:
            # Figures name the scan by the cleaned ScanInfo string, not the
            # s-file column, exactly as the legacy renderers did (bin singles
            # included, so a one-bin scan is labelled the same way).
            plan = replace(plan, position_label=self.scan_parameter or "")
        self.last_plan = plan
        saved = save_products(plan, self.spec, scan_folder)
        for note in saved.notes:
            logger.warning("%s: %s", self.device_name, note)
        self.display_contents = [str(path) for path in saved.display_files]
        return list(self.display_contents)

    def _execute(self, prepared: PreparedScan, collector: ProductCollector) -> None:
        """Stream the units into the collector; log failures; persist scalars.

        Nothing per unit outlives its iteration but its scalar records and
        what the collector keeps.
        """
        workers = effective_workers(
            self.spec.workers, len(prepared.groups), cap=self.worker_cap
        )
        logger.info(
            "%s: %d units, %d worker%s",
            self.device_name,
            len(prepared.groups),
            workers,
            "" if workers == 1 else "s",
        )
        for service in prepared.prepared.inputs.values():
            # A service that runs one external process per shot (WaveKit)
            # shares the cores among the pool's concurrent shots.
            share = getattr(service, "share_cores", None)
            if callable(share):
                share(workers)
        pending: list[dict] = []
        definition = measure_definition(prepared.prepared.recipe.analysis.measure)
        sidecar = definition.sidecar
        store: ShotStore | None = None
        if (
            definition.shot_store is not None
            and self.spec.save
            and not prepared.average_before_analysis
        ):
            store = ShotStore(
                shot_store_path(
                    self.spec, Path(self.scan_directory), definition.shot_store
                )
            )
        try:
            fold_error = self._stream(
                prepared, collector, workers, pending, sidecar, store
            )
        except BaseException:
            if store is not None:
                self._discard_store(store)
            raise
        if store is not None:
            # The store is a product: like every product write, its final
            # flush and rename must never cost the run its scalars.
            try:
                kept = store.close()
            except OSError as exc:
                logger.warning(
                    "%s: shot store %s not kept (%s); the run's scalars are unaffected",
                    self.device_name,
                    store.path.name,
                    exc,
                )
            else:
                if kept is not None:
                    logger.info(
                        "%s: %d shots stored in %s", self.device_name, store.count, kept
                    )
        if pending:
            updates = pd.DataFrame(pending)
            # The legacy wrapper wrote its scalars into the in-memory rows
            # before persisting, so a waterfall sorted by one of this run's
            # own columns resolves even when the s-file merge is refused.
            write_scalars_into_rows(self.auxiliary_data, pending)
            self.write_scalar_sidecar(updates)
            self.append_to_sfile(updates)
        if fold_error is not None:
            raise fold_error

    def _stream(
        self,
        prepared: PreparedScan,
        collector: ProductCollector,
        workers: int,
        pending: list[dict],
        sidecar: str | None,
        store: ShotStore | None,
    ) -> ValueError | None:
        """The per-outcome loop: fold, persist per shot, queue the scalars.

        Returns the first fold error (raised by the caller once the scalars
        are persisted), or ``None``.
        """
        fold_error: ValueError | None = None
        for outcome in prepared.run(workers=workers):
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
            # A result the products cannot fold (units or axes that disagree,
            # a repeated key) is raised only after the scalars are persisted,
            # as the old sequence planner raised after the s-file merge.
            if fold_error is None:
                try:
                    collector.add(outcome)
                except ValueError as exc:
                    fold_error = exc
            if sidecar is not None:
                self._write_sidecar(prepared, outcome, sidecar)
            if store is not None and len(outcome.loaded_shots) == 1:
                store = self._store_shot(store, outcome.loaded_shots[0], outcome)
            pending.extend(prepared.scalar_records(outcome))
        return fold_error

    @staticmethod
    def _discard_store(store: ShotStore) -> None:
        """Drop a store's part file on the way out of a failed run; never raise."""
        try:
            store.close(keep=False)
        except OSError as exc:
            logger.warning("Shot store %s not discarded: %s", store.path.name, exc)

    def _store_shot(self, store: ShotStore, shot: int, outcome) -> ShotStore | None:
        """Append one shot to the store; a store failure drops the store, not the run.

        Like the sidecar, the store is per single-shot unit only (a bin's
        averaged frame has no one shot). A write or shape failure is
        logged, the partial store is discarded, and the run goes on with
        its scalars and products.
        """
        try:
            store.add(shot, outcome.measurement)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Shot %s: shot store not written (%s); discarding %s",
                shot,
                exc,
                store.path.name,
            )
            self._discard_store(store)
            return None
        return store

    def _write_sidecar(self, prepared: PreparedScan, outcome, name: str) -> None:
        """Write one shot's sidecar table beside its file, as the legacy analyzer did.

        Written whatever ``save`` says (the legacy analyzer wrote it on every
        analyzed shot) and only for a single-shot unit: a bin's averaged
        frame has no one shot to sit beside. A write failure is logged and
        the run goes on; the shot's scalars are unaffected.
        """
        measurement = outcome.measurement
        if not measurement.extras or len(outcome.loaded_shots) != 1:
            return
        shot = outcome.loaded_shots[0]
        try:
            write_shot_table(measurement, prepared.source.references[shot], shot, name)
        except (OSError, ValueError) as exc:
            logger.warning("Shot %s: %s table not written: %s", shot, name, exc)

    def cleanup(self) -> None:
        """Release the loaded s-file and the display list after a run."""
        self.auxiliary_data = None
        self.display_contents = []
        self.last_plan = None
        logger.debug("[CoreScanAnalyzer] cleanup() complete.")
