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
from geecs_analysis.compat.v2 import V2Recipe
from geecs_analysis.pipeline import bind_inputs
from geecs_analysis.recipe import AnalysisDocument, compile_document
from geecs_analysis.steps.background_constant import BackgroundConstantSpec
from geecs_analysis.steps.background_frame import BackgroundFrameSpec
from geecs_data_utils.frames import Frame
from geecs_data_utils.io.images import read_imaq_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedRecipe:
    """An immutable compiled recipe and its already-loaded frame bindings."""

    recipe: V2Recipe
    inputs: Mapping[str, Frame]


def prepare_v2(
    document: AnalysisDocument, *, data_dir: Path | None = None
) -> PreparedRecipe:
    """Compile either document before reading inputs; load its frame inputs.

    ``data_dir`` is the device data directory, matching the old scan wrapper's
    ``{scan_dir}`` substitution. A context-free preview leaves the placeholder
    literal, as before. Load/float-conversion failures select the request's
    fallback constant and log a warning; a request without one (a v3 recipe
    that says so) makes the failure an error. Successfully loaded malformed
    geometry raises instead of silently selecting the constant. Each distinct
    background is loaded once for this prepared run, including repeated
    pipeline steps.
    """
    recipe = compile_document(document, allow_file_backgrounds=True)
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
    return PreparedRecipe(recipe, bind_inputs(recipe.analysis.steps, inputs))
