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

from geecs_analysis import summaries as _summaries  # noqa: F401 -- registers the summary kinds
from geecs_analysis.compat.v2 import (
    FileBackground,
    ScanBackground,
    TableInput,
    UnsupportedRecipe,
    V2Recipe,
    compile_v2,
    trace_channels,
)
from geecs_analysis.pipeline import measure_input, measure_tables, shot_scalar_columns
from geecs_analysis.registry import (
    definition,
    definitions,
    measure_definition,
    measure_definitions,
    summary_definitions,
)
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.specs import Analysis
from geecs_analysis.steps.roi import RoiSpec

#: The published recipe reference: one card per step, measure and summary,
#: addressed as ``#step-<name>``, ``#measure-<kind>`` and ``#summary-<kind>``.
#: The project's public docs, not a facility value.
RECIPE_REFERENCE_URL = (
    "https://geecs-plugins.readthedocs.io/en/latest/sites/analysis_recipes/index.html"
)

AnalysisDocument = Union[AnalysisDiagnostic, AnalysisRecipe]


class RecipeError(ValueError):
    """A recipe names a step, measure, parameter or binding the core cannot run."""


def compile_recipe(
    recipe: AnalysisRecipe, *, allow_file_backgrounds: bool = False
) -> V2Recipe:
    """Bind the recipe's vocabulary to the registry and compile it, without I/O.

    Unknown step or measure names, unknown parameters, steps or a measure
    that do not process the input's frames, and bindings the steps and the
    measure do not use (or use without declaring) are refused with
    :class:`RecipeError` — so is an input declared as a table but bound as
    a frame, or the reverse.
    Frame and table inputs require a source host that loads them
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
    required = _step_inputs(analysis)
    required |= _reference_input(recipe, analysis, required)
    tables = _table_inputs(analysis, required)
    required |= tables
    _check_declarations(recipe, required, tables)
    if required and not allow_file_backgrounds:
        raise RecipeError("frame and table inputs need a source host that loads them")
    if (required - tables) and isinstance(recipe.input, LineInput):
        raise RecipeError("frame inputs are loaded for camera recipes only")
    if shot_scalar_columns(analysis.measure) and recipe.scan.average_frames_first:
        raise RecipeError(
            "the measure reads the scan's recorded scalars per shot; "
            "scan.average_frames_first averages the frames before it sees a shot"
        )
    common = dict(
        analysis=analysis,
        input_kind=recipe.input.kind,
        device=recipe.device,
        output_name=recipe.effective_output_name,
        metric_suffix=recipe.scalar_suffix,
    )
    if isinstance(recipe.input, LineInput):
        return _line_recipe(recipe.input, analysis, common)
    return _camera_recipe(recipe, analysis, common)


def _line_recipe(source: LineInput, analysis: Analysis, common: dict) -> V2Recipe:
    """The compiled line recipe: the source's scaling and the measure's channels."""
    try:
        channels = trace_channels(
            analysis.measure, source.loading.auxiliary_columns, analysis.steps
        )
    except UnsupportedRecipe as exc:
        raise RecipeError(str(exc)) from exc
    if channels and source.siblings:
        raise RecipeError("a stitched line input carries no auxiliary channels")
    return V2Recipe(
        **common,
        channels=channels,
        storage_dtype=source.storage_dtype,
        x_scale=source.x_scale,
        y_scale=source.y_scale,
        x_unit=source.x_unit,
        y_unit=source.y_unit,
        label=source.label,
    )


def _camera_recipe(
    recipe: AnalysisRecipe, analysis: Analysis, common: dict
) -> V2Recipe:
    """The compiled camera recipe: the crop origin and the host's input requests."""
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
            if binding.path is not None
        ),
        scan_backgrounds=tuple(
            ScanBackground(
                key,
                binding.from_scan.scan,
                binding.from_scan.statistic,
                binding.from_scan.percentile,
            )
            for key, binding in recipe.inputs.items()
            if binding.from_scan is not None
        ),
        tables=tuple(
            TableInput(key, binding.table)
            for key, binding in recipe.inputs.items()
            if binding.table is not None
        ),
    )


def _step_inputs(analysis: Analysis) -> set[str]:
    """The frame keys the steps bind."""
    return {
        getattr(spec, definition(spec).input_field)
        for spec in analysis.steps
        if definition(spec).input_field is not None
    }


def _reference_input(
    recipe: AnalysisRecipe, analysis: Analysis, required: set[str]
) -> set[str]:
    """The measure's comparison frame, if any: its own key, never a step's."""
    reference = measure_input(analysis.measure)
    if reference is None:
        return set()
    if reference in required:
        # The reference goes through the steps: a step subtracting the
        # same frame would leave the measure comparing against zeros.
        raise RecipeError(
            f"frame input {reference!r} is bound by a step and by the measure; "
            "declare the measure's comparison frame under its own name"
        )
    binding = recipe.inputs.get(reference)
    if binding is not None and binding.fallback_level is not None:
        raise RecipeError(
            f"frame input {reference!r} is the measure's comparison frame; a "
            "fallback_level (a constant for a background step) cannot stand in for it"
        )
    return {reference}


def _table_inputs(analysis: Analysis, frames: set[str]) -> set[str]:
    """The keys the measure binds as tables: one per field, none of them a frame."""
    keys = list(measure_tables(analysis.measure).values())
    if len(set(keys)) != len(keys):
        raise RecipeError(
            "the measure binds one input under two table fields: "
            f"{sorted({key for key in keys if keys.count(key) > 1})}"
        )
    tables = set(keys)
    if tables & frames:
        raise RecipeError(
            f"inputs bound both as a frame and as a table: {sorted(tables & frames)}"
        )
    return tables


def _check_declarations(
    recipe: AnalysisRecipe, required: set[str], tables: set[str]
) -> None:
    """Every bound key is declared, every declared key is used, each as its kind."""
    declared = set(recipe.inputs)
    if required - declared:
        raise RecipeError(
            "steps or the measure bind inputs the recipe does not declare: "
            f"{sorted(required - declared)}"
        )
    if declared - required:
        raise RecipeError(
            "recipe declares inputs no step or measure uses: "
            f"{sorted(declared - required)}"
        )
    for key, binding in recipe.inputs.items():
        if (binding.table is not None) != (key in tables):
            raise RecipeError(
                f"input {key!r} is declared as a "
                f"{'table' if binding.table is not None else 'frame'} but bound as a "
                f"{'frame' if binding.table is not None else 'table'}"
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
    return document.input_kind == "line"


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


def recipe_schema() -> dict:
    """The recipe's JSON Schema with ``steps`` and ``measure`` bound to the registry.

    The document's own schema types a step as an open reference (a registered
    name plus parameters as written) because the numerical vocabulary is the
    core's, not the schema package's. A form needs that vocabulary listed, so
    this replaces the reference types with the registry's discriminated unions
    (:class:`geecs_analysis.specs.Analysis`): one variant per registered step
    and measure, its parameters typed and described. Every step, measure and
    summary variant carries ``x-ndim``, the frame dimensionalities it
    processes, so a form can say which fit the input, and ``x-docs``, its
    card in the published reference (:data:`RECIPE_REFERENCE_URL`). Every
    measure variant also carries ``x-scalars``, its ``scalar_docs``: what
    each scalar it can write means. Validation is not
    changed by this: a document still validates as :class:`AnalysisRecipe`
    and then binds through :func:`compile_recipe`.
    """
    from geecs_analysis.specs import Analysis

    schema = AnalysisRecipe.model_json_schema()
    bound = Analysis.model_json_schema()
    defs = {**schema.get("$defs", {}), **bound.get("$defs", {})}
    defs.pop("StepRef", None)
    defs.pop("MeasureRef", None)
    for role, field, items in (
        ("step", "step", definitions()),
        ("measure", "kind", measure_definitions()),
        ("summary", "kind", summary_definitions()),
    ):
        for item in items:
            variant = defs[item.spec.__name__]
            name = item.spec.model_fields[field].default
            variant["x-ndim"] = sorted(item.ndim)
            variant["x-docs"] = f"{RECIPE_REFERENCE_URL}#{role}-{name}"
    for item in measure_definitions():
        defs[item.spec.__name__]["x-scalars"] = dict(item.spec.scalar_docs)
    properties = dict(schema["properties"])
    properties["steps"] = {
        **properties["steps"],
        "items": bound["properties"]["steps"]["items"],
    }
    properties["measure"] = {
        **bound["properties"]["measure"],
        "description": properties["measure"].get("description", ""),
    }
    return {**schema, "$defs": defs, "properties": properties}


__all__ = [
    "RECIPE_REFERENCE_URL",
    "AnalysisDocument",
    "RecipeError",
    "compile_document",
    "compile_recipe",
    "figure_of",
    "is_line",
    "recipe_schema",
    "summaries_of",
]
