"""Bind an analysis recipe (format v3) to the registry and compile it to run.

The schema types the recipe's frame; this module binds its ``steps`` and
``measure`` to the builtin registry, checks that each can process the
input's frames, and produces the same in-memory compiled recipe the v2
adapter produces, so both document formats run through one evaluator
(``compat.v2.analyze_v2``) and a converted recipe is identical to its
source by construction. Reading a v2 diagnostic alongside is the
transition's convenience: consumers call :func:`compile_document`,
:func:`figure_of` and :func:`summaries_of` and never ask which format they
hold. No file access and no numerical imports happen here.
"""

from __future__ import annotations

from typing import Union

from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe
from geecs_schemas.analysis.recipe import LineInput
from pydantic import ValidationError

from geecs_analysis.compat.v2 import FileBackground, V2Recipe, compile_v2
from geecs_analysis.registry import definition, measure_definition
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.specs import Analysis
from geecs_analysis.steps.roi import RoiSpec

AnalysisDocument = Union[AnalysisDiagnostic, AnalysisRecipe]


class RecipeError(ValueError):
    """A recipe names a step, measure, parameter or binding the core cannot run."""


def compile_recipe(
    recipe: AnalysisRecipe, *, allow_file_backgrounds: bool = False
) -> V2Recipe:
    """Bind the recipe's vocabulary to the registry and compile it, without I/O.

    Unknown step or measure names, unknown parameters, steps or a measure
    that do not process the input's frames, and frame bindings the steps do
    not use (or use without declaring) are refused with :class:`RecipeError`.
    Frame inputs require a source host that loads them
    (``allow_file_backgrounds=True``); the live optimizer has none.
    """
    try:
        analysis = Analysis.model_validate(
            {
                "steps": [step.model_dump() for step in recipe.steps],
                "measure": recipe.measure.model_dump(),
            }
        )
    except ValidationError as exc:
        raise RecipeError(f"recipe does not bind to the registry: {exc}") from exc
    ndim = type(recipe.input).ndim
    for spec in analysis.steps:
        if ndim not in definition(spec).ndim:
            raise RecipeError(
                f"step {spec.step!r} does not process {recipe.input.kind} frames"
            )
    if ndim not in measure_definition(analysis.measure).ndim:
        raise RecipeError(
            f"measure {analysis.measure.kind!r} does not measure "
            f"{recipe.input.kind} frames"
        )
    required = {
        getattr(spec, definition(spec).input_field)
        for spec in analysis.steps
        if definition(spec).input_field is not None
    }
    declared = set(recipe.inputs)
    if required - declared:
        raise RecipeError(
            f"steps bind frame inputs the recipe does not declare: {sorted(required - declared)}"
        )
    if declared - required:
        raise RecipeError(
            f"recipe declares frame inputs no step uses: {sorted(declared - required)}"
        )
    if required and not allow_file_backgrounds:
        raise RecipeError("frame inputs need a source host that loads them")
    if required and isinstance(recipe.input, LineInput):
        raise RecipeError("frame inputs are loaded for camera recipes only")
    common = dict(
        analysis=analysis,
        input_kind=recipe.input.kind,
        device=recipe.device,
        output_name=recipe.effective_output_name,
        metric_suffix=recipe.scalar_suffix,
    )
    if isinstance(recipe.input, LineInput):
        source = recipe.input
        return V2Recipe(
            **common,
            storage_dtype=source.storage_dtype,
            x_scale=source.x_scale,
            y_scale=source.y_scale,
            x_unit=source.x_unit,
            y_unit=source.y_unit,
            label=source.label,
        )
    origin = (0, 0)
    for spec in analysis.steps:
        # Beam coordinates start where the first crop starts, as the pure
        # Frame API's axes do; no inactive section can shift them.
        if isinstance(spec, RoiSpec) and spec.units == "index":
            origin = tuple(int(lo or 0) for lo, _ in spec.bounds)
            break
    return V2Recipe(
        **common,
        camera_origin=origin,
        file_backgrounds=tuple(
            FileBackground(key, binding.path, binding.fallback_level)
            for key, binding in recipe.inputs.items()
        ),
    )


def compile_document(
    document: AnalysisDocument, *, allow_file_backgrounds: bool = False
) -> V2Recipe:
    """Compile either document format to the one in-memory recipe."""
    if isinstance(document, AnalysisRecipe):
        return compile_recipe(document, allow_file_backgrounds=allow_file_backgrounds)
    return compile_v2(document, allow_file_backgrounds=allow_file_backgrounds)


def is_line(document: AnalysisDocument) -> bool:
    """Whether the document's frames are traces (1D)."""
    if isinstance(document, AnalysisRecipe):
        return isinstance(document.input, LineInput)
    return document.image_kind == "line"


def figure_of(document: AnalysisDocument) -> FigureSpec:
    """The per-frame draw: the recipe's ``figure``, or the v2 renderer's translation."""
    if isinstance(document, AnalysisRecipe):
        return FigureSpec.model_validate(document.figure.model_dump())
    from geecs_analysis.compat.v2_render import figure_v2

    return figure_v2(document.scan.renderer, line=is_line(document))


def summaries_of(document: AnalysisDocument) -> tuple:
    """The summary kinds to draw: as declared, or the v2 renderer's fixed pair."""
    if isinstance(document, AnalysisRecipe):
        return tuple(document.summaries)
    from geecs_analysis.compat.v2_render import summaries_v2

    return summaries_v2(document.scan.renderer, line=is_line(document))


__all__ = [
    "AnalysisDocument",
    "RecipeError",
    "compile_document",
    "compile_recipe",
    "figure_of",
    "is_line",
    "summaries_of",
]
