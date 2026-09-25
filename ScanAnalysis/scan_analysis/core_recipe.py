"""One view of either analysis document for the scan host: what a run needs to know.

The core compiles both formats to one numerical recipe; the scan host
additionally needs where the files are, how they are read, how the run
behaves and what to draw. ``scan_recipe`` reads those from a v3
:class:`~geecs_schemas.analysis.AnalysisRecipe` or a v2
:class:`~geecs_schemas.analysis.AnalysisDiagnostic` once, so the source,
runner, planner and sink never ask which format they serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from geecs_analysis.compat.v2 import V2Recipe
from geecs_analysis.recipe import (
    AnalysisDocument,
    compile_document,
    figure_of,
    summaries_of,
)
from geecs_analysis.render.specs import FigureSpec
from geecs_schemas.analysis import AnalysisRecipe

__all__ = ["AnalysisDocument", "ScanRecipe", "scan_recipe"]


@dataclass(frozen=True)
class ScanRecipe:
    """A run's compiled recipe plus the host-side facts, format-independent.

    ``device`` names the product files and the auxiliary-data prefix;
    ``folder`` is the data subfolder under the scan (the device unless the
    document says otherwise); ``output_name`` selects the analyzer directory
    and prefixes the s-file columns, ``scalar_suffix`` ends them.
    ``line_loading_json`` is the trace reader's configuration (``None`` for
    cameras). ``summaries`` are the schema option models the sink resolves
    against the summary registry.
    """

    recipe: V2Recipe
    device: str
    folder: str
    output_name: str
    scalar_suffix: str
    file_tail: Optional[str]
    prefer_stack: bool
    line_loading_json: Optional[str]
    average_frames_first: bool
    save: bool
    priority: int
    figure: FigureSpec
    summaries: tuple

    @property
    def line(self) -> bool:
        """Whether the frames are traces."""
        return self.recipe.input_kind == "line"


def scan_recipe(document: AnalysisDocument) -> ScanRecipe:
    """Compile the document (declaring file backgrounds) and read the run facts."""
    recipe = compile_document(document, allow_file_backgrounds=True)
    loading = document.line_loading
    if isinstance(document, AnalysisRecipe):
        # What only the format knows: where file discovery and the runtime
        # switches live. Everything named the same way is read below.
        source = document.input
        file_tail, prefer_stack = source.file_tail, source.format == "device_hdf5"
        scalar_suffix = document.scalar_suffix
        average_first = document.scan.average_frames_first
    else:
        file_tail = document.scan.file_tail
        prefer_stack = document.scan.data_format == "device_hdf5"
        scalar_suffix = document.metric_suffix
        average_first = document.scan.mode == "per_bin"
    return ScanRecipe(
        recipe=recipe,
        device=document.device,
        folder=document.data_folder,
        output_name=document.effective_output_name,
        scalar_suffix=scalar_suffix or "",
        file_tail=file_tail,
        prefer_stack=prefer_stack,
        line_loading_json=loading.model_dump_json() if loading is not None else None,
        average_frames_first=average_first,
        save=document.scan.save,
        priority=document.scan.priority,
        figure=figure_of(document),
        summaries=summaries_of(document),
    )
