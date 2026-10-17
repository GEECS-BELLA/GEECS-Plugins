"""Read any document of the analyzers tree as the kind it declares.

One tree holds two documents: the
:class:`~geecs_schemas.analysis.recipe.AnalysisRecipe` for every analysis
and the :class:`~geecs_schemas.analysis.himg.HimgJob` for the ``.himg``
data-management jobs.  Every loader dispatches here so a file's kind is
read once, from what it declares (a top-level ``kind`` is a job), and
never guessed from its other keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from geecs_schemas.analysis.himg import HimgJob
from geecs_schemas.analysis.recipe import AnalysisRecipe

#: What a loader hands back: one of the two documents of the tree.
AnalysisDocument = Union[AnalysisRecipe, HimgJob]


def load_analysis_document(data: Mapping[str, object]) -> AnalysisDocument:
    """Validate a raw document as the kind it declares.

    Parameters
    ----------
    data : Mapping
        The raw YAML mapping.

    Returns
    -------
    AnalysisRecipe or HimgJob
        A top-level ``kind`` validates as a ``.himg`` job (a recipe has
        none); anything else validates as a recipe, which refuses another
        ``schema_version`` and the retired v2 layout with its own message.

    Raises
    ------
    pydantic.ValidationError
        When the document does not validate as its declared kind.
    """
    if "kind" in data:
        return HimgJob.model_validate(data)
    return AnalysisRecipe.model_validate(data)


__all__ = ["AnalysisDocument", "load_analysis_document"]
