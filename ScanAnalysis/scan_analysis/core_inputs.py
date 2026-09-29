"""Prepare the pure v2 core's file inputs through shared data-utils readers.

This is source orchestration, shared by explicit scan runs and portal previews.
Numerical execution stays in geecs-analysis; no files are written here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from geecs_analysis.compat.v2 import UnsupportedRecipe, V2Recipe
from geecs_analysis.pipeline import bind_inputs
from geecs_analysis.registry import measure_definition
from geecs_analysis.recipe import AnalysisDocument, compile_document
from geecs_analysis.steps.background_constant import BackgroundConstantSpec
from geecs_analysis.steps.background_frame import BackgroundFrameSpec
from geecs_data_utils.frames import Frame
from geecs_data_utils.io.images import read_imaq_image

from scan_analysis.core_services import services_for

logger = logging.getLogger(__name__)


class ServicesNotRequested(UnsupportedRecipe):
    """The recipe's measure needs a service and this caller did not ask for one.

    A measure with a service (the FROG retrieval, the WaveKit engine) starts
    an external program per frame. Scan runs and the editor's explicit previews ask for it; a
    per-request view (the portal's shot browser) does not, and — this being
    an ``UnsupportedRecipe`` — keeps the route it had before the port, which
    refuses such a kind outright.
    """


@dataclass(frozen=True)
class PreparedRecipe:
    """An immutable compiled recipe, its loaded frame bindings and its services.

    ``inputs`` holds the frames the steps bind and, when the measure names a
    service (``core_services``), the collaborator built for it. Pickles (a
    pooled run sends it to each worker once): the bound inputs travel as a
    plain mapping and are rebound as a read-only view.
    """

    recipe: V2Recipe
    inputs: Mapping[str, object]

    def __getstate__(self) -> dict:
        """Pickle the bindings as a plain dict (a proxy view cannot be)."""
        return {"recipe": self.recipe, "inputs": dict(self.inputs)}

    def __setstate__(self, state: dict) -> None:
        """Restore the read-only binding view."""
        object.__setattr__(self, "recipe", state["recipe"])
        object.__setattr__(
            self,
            "inputs",
            bind_inputs(
                state["recipe"].analysis.steps,
                dict(state["inputs"]),
                measure=state["recipe"].analysis.measure,
            ),
        )


class ScanContextRequired(UnsupportedRecipe):
    """A scan background needs the scan it comes from, or a computed cache.

    A context-free preview (no scan folder), or a per-request view whose
    background no run has computed yet, refuses the recipe this way — an
    ``UnsupportedRecipe``, so the caller keeps its old route — rather than
    read a whole scan or write a file to draw one frame.
    """


def prepare_v2(
    document: AnalysisDocument,
    *,
    data_dir: Path | None = None,
    services: bool = True,
    compute_scan_backgrounds: bool = True,
) -> PreparedRecipe:
    """Compile either document before reading inputs; load its frame inputs.

    ``data_dir`` is the device data directory, matching the old scan wrapper's
    ``{scan_dir}`` substitution. A context-free preview leaves the placeholder
    literal, as before. Load/float-conversion failures select the request's
    fallback constant and log a warning; a request without one (a v3 recipe
    that says so) makes the failure an error. Successfully loaded malformed
    geometry raises instead of silently selecting the constant. Each distinct
    background is loaded once for this prepared run, including repeated
    pipeline steps. A measure's service (the FROG retriever, the WaveKit
    engine) is built here from this host's config — and, for WaveKit, from
    the scan's own capture stack — so a host that cannot provide it fails
    now, before any shot is read; ``services=False`` refuses such a recipe
    with :class:`ServicesNotRequested` instead, before any file is read.

    A scan background (``scan.background_source``, a recipe's ``from_scan``
    input) is computed from its scan's frames by ``core_backgrounds`` — and
    cached — before any shot is read; ``compute_scan_backgrounds=False``
    uses only an existing cache and refuses the recipe
    (``ScanContextRequired``) otherwise.
    """
    recipe = compile_document(document, allow_file_backgrounds=True)
    service = measure_definition(recipe.analysis.measure).service
    if service is not None and not services:
        raise ServicesNotRequested(
            f"the {recipe.analysis.measure.kind!r} measure runs the {service!r} "
            "service (an external program per frame); this caller did not ask "
            "for services"
        )
    inputs = {}
    fallbacks = {}
    for request in recipe.file_backgrounds:
        path = request.path
        if data_dir is not None:
            path = path.replace("{scan_dir}", str(data_dir))
        try:
            background = read_imaq_image(Path(path)).astype(np.float64)
        except Exception as exc:
            if request.fallback_level is None:
                raise
            # Legacy catches reader/float conversion failures, but shape
            # validation happens after loading and must remain a hard error.
            logger.warning(
                "Failed to load background from %s: %s. Falling back to constant_level=%s.",
                path,
                exc,
                request.fallback_level,
            )
            fallbacks[request.key] = request.fallback_level
        else:
            inputs[request.key] = Frame.from_array(background)
    if fallbacks:
        steps = tuple(
            BackgroundConstantSpec(level=fallbacks[spec.source])
            if isinstance(spec, BackgroundFrameSpec) and spec.source in fallbacks
            else spec
            for spec in recipe.analysis.steps
        )
        recipe = replace(
            recipe, analysis=recipe.analysis.model_copy(update={"steps": steps})
        )
    for request in recipe.scan_backgrounds:
        inputs[request.key] = Frame.from_array(
            _scan_background(document, request, data_dir, compute_scan_backgrounds)
        )
    inputs.update(services_for(recipe.analysis.measure, data_dir=data_dir))
    return PreparedRecipe(
        recipe,
        bind_inputs(recipe.analysis.steps, inputs, measure=recipe.analysis.measure),
    )


def _scan_background(document, request, data_dir, compute):
    """One scan-background request, resolved against the run's device folder."""
    from scan_analysis.core_backgrounds import resolve_scan_background
    from scan_analysis.core_recipe import scan_recipe

    if data_dir is None:
        raise ScanContextRequired(
            "A scan background needs the scan it comes from (no scan folder given)"
        )
    spec = scan_recipe(document)
    try:
        return resolve_scan_background(
            request,
            data_dir=Path(data_dir),
            device=spec.device,
            file_tail=spec.file_tail,
            prefer_stack=spec.prefer_stack,
            compute=compute,
        )
    except LookupError as exc:
        raise ScanContextRequired(str(exc)) from exc
