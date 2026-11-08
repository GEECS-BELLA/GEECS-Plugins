"""Read any document of the analyzers tree as the kind it declares.

One tree holds three documents: the
:class:`~geecs_schemas.analysis.recipe.AnalysisRecipe` for every analysis,
the :class:`~geecs_schemas.analysis.himg.HimgJob` for the ``.himg``
data-management jobs, and the
:class:`~geecs_schemas.analysis.scalar.ScalarJob` for s-file columns
computed from other columns.  Every loader dispatches here so a file's kind is
read once, from what it declares (a top-level ``kind`` is a job), and
never guessed from its other keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from geecs_schemas.analysis.himg import HimgJob
from geecs_schemas.analysis.recipe import AnalysisRecipe
from geecs_schemas.analysis.scalar import ScalarJob

#: What a loader hands back: one of the three documents of the tree.
AnalysisDocument = Union[AnalysisRecipe, HimgJob, ScalarJob]


def load_analysis_document(data: Mapping[str, object]) -> AnalysisDocument:
    """Validate a raw document as the kind it declares.

    Parameters
    ----------
    data : Mapping
        The raw YAML mapping.

    Returns
    -------
    AnalysisRecipe, HimgJob or ScalarJob
        A top-level ``kind`` starting ``scalar`` validates as a scalar-columns
        job, any other top-level ``kind`` as a ``.himg`` job (a recipe has
        none); anything else validates as a recipe, which refuses another
        ``schema_version`` and the retired v2 layout with its own message.

    Raises
    ------
    pydantic.ValidationError
        When the document does not validate as its declared kind.
    """
    if str(data.get("kind", "")).startswith("scalar"):
        # a misspelt scalar kind fails naming scalar_columns, not the .himg kinds
        return ScalarJob.model_validate(data)
    if "kind" in data:
        return HimgJob.model_validate(data)
    return AnalysisRecipe.model_validate(data)


__all__ = ["AnalysisDocument", "load_analysis_document"]
