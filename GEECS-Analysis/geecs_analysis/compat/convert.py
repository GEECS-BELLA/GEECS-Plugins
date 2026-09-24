"""Convert a v2 diagnostic the core serves into a v3 recipe, once, losslessly.

The conversion is built on the v2 adapter's compile output: the steps and
measure are the compiled specs written back as registry references, the
input and naming are the compiled recipe's conventions, and the renderer
options become the ``figure`` and ``summaries`` blocks. The result is
recompiled and compared with the source's compilation, so a converted
recipe runs identically by construction; anything the v3 shape does not
carry is reported in ``notes`` rather than dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisRecipe
from geecs_schemas.analysis.processing_1d import Line1DConfig
from geecs_schemas.analysis.renderer import RendererOptions

from geecs_analysis.compat.v2 import compile_v2
from geecs_analysis.recipe import compile_recipe


@dataclass(frozen=True)
class Conversion:
    """The converted recipe and what the v3 shape does not carry from the source."""

    recipe: AnalysisRecipe
    notes: tuple[str, ...]


def _tidy(value):
    """Write whole-number floats as ints so the YAML reads as it was written."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_tidy(item) for item in value]
    if isinstance(value, dict):
        return {key: _tidy(item) for key, item in value.items()}
    return value


def _spec_mapping(spec, tag: str) -> dict:
    """A registered spec as a recipe reference: the tag first, non-defaults after."""
    return {
        tag: getattr(spec, tag),
        **_tidy(spec.model_dump(mode="json", exclude_defaults=True, exclude={tag})),
    }


def _figure_v2(options: RendererOptions, *, line: bool, notes: list[str]) -> dict:
    """Only what the v2 document set, plus the image palette's zero floor."""
    figure: dict = {}
    if not line:
        imshow: dict = {}
        mode = options.colormap_mode or "sequential"
        if mode == "sequential":
            # The v2 image palette always started at zero; the pure draw would
            # autoscale from the data minimum, so the floor is written out.
            imshow["vmin"] = 0
        elif mode == "diverging":
            notes.append(
                "renderer.colormap_mode 'diverging' has no static v3 form "
                "(figure.imshow autoscales); dropped"
            )
        elif options.vmin is not None:
            imshow["vmin"] = options.vmin
        if options.vmax is not None and mode != "diverging":
            imshow["vmax"] = options.vmax
        if options.cmap is not None:
            imshow["cmap"] = options.cmap
        figure["imshow"] = imshow
    axes = {k: getattr(options, k) for k in ("xlabel", "ylabel") if getattr(options, k)}
    if axes:
        figure["axes"] = axes
    if options.colorbar_label:
        figure["colorbar"] = {"label": options.colorbar_label}
    # The v2 renderers drew every figure at 150 dpi; the core default is the
    # preview's 110, so the resolution is written out (editable, like vmin).
    fig = {"dpi": options.dpi if options.dpi is not None else 150}
    if options.figsize_inches is not None:
        fig["figsize"] = [options.figsize_inches, options.figsize_inches]
        notes.append(
            "renderer.figsize_inches became figure.fig.figsize (a square canvas)"
        )
    if fig:
        figure["fig"] = fig
    if options.duration is not None:
        notes.append("renderer.duration (animation) dropped: no animation kind")
    if options.mode is not None:
        notes.append(
            f"renderer.mode {options.mode!r} dropped: summaries are listed kinds"
        )
    return figure


def _summaries_v2(options: RendererOptions, *, line: bool) -> list[dict]:
    if line:
        stack: dict = {"kind": "waterfall"}
        for field, key in (
            ("waterfall_sort_key", "sort_key"),
            ("waterfall_sort_sigma", "sort_sigma"),
            ("waterfall_sort_bounds", "sort_bounds"),
            ("waterfall_even_y_spacing", "even_spacing"),
            ("colormap_mode", "scale"),
            ("cmap", "cmap"),
            ("vmin", "vmin"),
            ("vmax", "vmax"),
        ):
            value = getattr(options, field)
            if value is not None:
                stack[key] = list(value) if isinstance(value, tuple) else value
        return [stack, {"kind": "average"}]
    # The v2 grid drew 6x6 inch panels; the core default is smaller, so the
    # panel size is written out too.
    grid: dict = {"kind": "image_grid", "panel_size": list(options.figsize or (6, 6))}
    return [grid, {"kind": "average"}]


def to_v3(document: AnalysisDiagnostic) -> Conversion:
    """Convert one v2 diagnostic the core serves; ``UnsupportedRecipe`` otherwise.

    Raises
    ------
    geecs_analysis.compat.v2.UnsupportedRecipe
        The recipe needs a capability the core has not ported; it stays v2.
    """
    compiled = compile_v2(document, allow_file_backgrounds=True)
    notes: list[str] = []
    config = document.image
    line = isinstance(config, Line1DConfig)
    scan = document.scan
    source: dict = {"kind": "line" if line else "camera"}
    if scan.device is not None:
        source["folder"] = scan.device
    if scan.file_tail is not None:
        source["file_tail"] = scan.file_tail
    if scan.data_format is not None:
        source["format"] = scan.data_format
    if line:
        source["loading"] = config.data_loading.model_dump(
            mode="json", exclude_defaults=True
        )
        for key, value, default in (
            ("x_scale", compiled.x_scale, 1.0),
            ("y_scale", compiled.y_scale, 1.0),
            ("x_unit", compiled.x_unit, ""),
            ("y_unit", compiled.y_unit, ""),
            ("label", compiled.label, ""),
            ("storage_dtype", compiled.storage_dtype, "float32"),
        ):
            if value != default:
                source[key] = value
    else:
        if "bit_depth" in config.model_fields_set:
            notes.append("image.bit_depth dropped: the core does not use it")
    if scan.gdoc_slot is not None:
        notes.append("scan.gdoc_slot dropped (retired)")

    raw: dict = {"schema_version": 3, "device": document.name}
    if document.output_name is not None:
        raw["output_name"] = document.output_name
    if document.metric_suffix is not None:
        raw["scalar_suffix"] = document.metric_suffix
    # The human notes survive: the document's own description, then the
    # image section's when it says something different.
    notes_text = [
        text
        for text in (document.description, getattr(config, "description", None))
        if text
    ]
    if notes_text:
        raw["description"] = "; ".join(dict.fromkeys(notes_text))
    if getattr(config, "metadata", None):
        raw["metadata"] = dict(config.metadata)
    raw["input"] = source
    if compiled.file_backgrounds:
        raw["inputs"] = {
            request.key: {
                "path": request.path,
                "fallback_level": request.fallback_level,
            }
            for request in compiled.file_backgrounds
        }
    raw["steps"] = [_spec_mapping(spec, "step") for spec in compiled.analysis.steps]
    raw["measure"] = _spec_mapping(compiled.analysis.measure, "kind")
    runtime: dict = {}
    if scan.priority != 100:
        runtime["priority"] = scan.priority
    if scan.mode == "per_bin":
        runtime["average_frames_first"] = True
    if not scan.save:
        runtime["save"] = False
    if runtime:
        raw["scan"] = runtime
    figure = _figure_v2(scan.renderer, line=line, notes=notes)
    if figure:
        raw["figure"] = figure
    raw["summaries"] = _summaries_v2(scan.renderer, line=line)

    recipe = AnalysisRecipe.model_validate(_tidy(raw))
    recipe._source_id = document.source_id
    check = compile_recipe(recipe, allow_file_backgrounds=True)
    if check.camera_origin != compiled.camera_origin:
        notes.append(
            f"beam coordinates started at {compiled.camera_origin} (an inactive "
            f"v2 roi section's origin); the recipe starts them at {check.camera_origin}"
        )
        check = replace(check, camera_origin=compiled.camera_origin)
    if check != compiled:
        raise AssertionError(
            "converted recipe does not compile to its source: "
            f"{check!r} != {compiled!r}"
        )
    return Conversion(recipe, tuple(notes))


__all__ = ["Conversion", "to_v3"]
