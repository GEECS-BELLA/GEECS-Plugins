"""Previews drawn with the run's own calls: one frame, or one summary over a few frames.

The editor's preview must be the product image a scan run would write, so
this module makes the run's calls and nothing else: :func:`prepare_document`
is the run's ``prepare_v2`` (frame inputs resolved from the document's
device folder under the scan, as ``core_scan.prepare_scan`` resolves them),
:func:`measure_frame` its per-frame analysis, :func:`preview_frame` the
sink's per-frame draw (``single`` with the document's figure block), and
:func:`preview_summary` the sink's summary draw — the registered kind's
layout over the given measurements, the ``average`` kind over their
noscan average. Nothing here reads a scan's rows, groups shots or writes
a file; the host chooses the frames and what each panel's position and
label mean (the editor previews "the first N shots", one panel per shot).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from geecs_analysis.compat.v2 import analyze_v2
from geecs_analysis.compat.v2_average import average_results
from geecs_analysis.recipe import AnalysisDocument, figure_of, summaries_of
from geecs_analysis.registry import summary_definition
from geecs_analysis.render import RenderError, single

from scan_analysis.core_inputs import PreparedRecipe, prepare_v2
from scan_analysis.core_source import source_directory

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from geecs_analysis.measurement import Measurement

__all__ = [
    "measure_frame",
    "prepare_document",
    "preview_frame",
    "preview_summary",
]


def prepare_document(
    document: AnalysisDocument, *, scan_folder: Optional[Path] = None
) -> PreparedRecipe:
    """Compile the document and load its frame inputs the way a run does.

    With *scan_folder* the recipe's ``{scan_dir}`` is the document's own
    device folder under that scan (``source_directory``); without one the
    placeholder stays literal, as ``prepare_v2`` documents. Raises
    ``UnsupportedRecipe`` for a v2 kind the core does not serve — the
    caller keeps its own route for those.
    """
    data_dir = (
        source_directory(document, scan_folder) if scan_folder is not None else None
    )
    return prepare_v2(document, data_dir=data_dir)


def measure_frame(prepared: PreparedRecipe, array: np.ndarray) -> Measurement:
    """The run's per-frame analysis of one loaded array."""
    return analyze_v2(array, prepared.recipe, inputs=prepared.inputs)


def preview_frame(
    document: AnalysisDocument,
    array: np.ndarray,
    *,
    scan_folder: Optional[Path] = None,
) -> Figure:
    """One frame drawn as the sink draws every shot product: ``single`` + ``figure``."""
    prepared = prepare_document(document, scan_folder=scan_folder)
    return single(measure_frame(prepared, array), figure_of(document))


def preview_summary(
    document: AnalysisDocument,
    arrays: Sequence[np.ndarray],
    positions: Sequence[Optional[float]],
    label: str,
    index: int,
    *,
    scan_folder: Optional[Path] = None,
) -> Figure:
    """The document's ``index``-th summary drawn over *arrays* as the sink draws it.

    A kind that consumes the ordered panels draws one panel per array at
    its position; the ``average`` kind draws the arrays' noscan average
    (``average_results``, the run's convention). Raises ``LookupError``
    when the document has no summary at *index*, ``RenderError`` when the
    frames cannot be summarised (mixed shapes; a kind's own refusal).
    """
    summaries = summaries_of(document)
    if not 0 <= index < len(summaries):
        raise LookupError(
            f"the document has {len(summaries)} summar"
            f"{'y' if len(summaries) == 1 else 'ies'}, no index {index}"
        )
    if not arrays or len(arrays) != len(positions):
        raise ValueError("a summary preview needs one position per frame")
    options = summaries[index]
    definition = summary_definition(options)
    prepared = prepare_document(document, scan_folder=scan_folder)
    results = [measure_frame(prepared, array) for array in arrays]
    figure = figure_of(document)
    if definition.consumes == "average":
        averaged = average_results(results, prepared.recipe, mode="noscan")
        if averaged is None:
            raise RenderError("the frames do not average: their shapes differ")
        return definition.function([averaged], [None], label, options, figure)
    return definition.function(list(results), list(positions), label, options, figure)
