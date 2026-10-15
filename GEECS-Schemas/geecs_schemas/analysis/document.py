"""Read any document of the analyzers tree as the kind it declares.

One tree holds three documents: the v3
:class:`~geecs_schemas.analysis.recipe.AnalysisRecipe` for every recipe
the analysis core serves, the v2
:class:`~geecs_schemas.analysis.diagnostic.AnalysisDiagnostic` for the
analyzer kinds it has not ported, and the
:class:`~geecs_schemas.analysis.himg.HimgJob` for the ``.himg``
data-management jobs.  Every loader dispatches here so a file's kind is
read once, from what it declares — a top-level ``kind`` (a job), else
``schema_version`` — and never guessed from its other keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from geecs_schemas._base import declared_schema_version
from geecs_schemas.analysis.diagnostic import AnalysisDiagnostic
from geecs_schemas.analysis.himg import HimgJob
from geecs_schemas.analysis.recipe import CURRENT_RECIPE_VERSION, AnalysisRecipe

#: What a loader hands back: one of the three documents of the tree.
AnalysisDocument = Union[AnalysisDiagnostic, AnalysisRecipe, HimgJob]


def load_analysis_document(data: Mapping[str, object]) -> AnalysisDocument:
    """Validate a raw document as the kind it declares.

    Parameters
    ----------
    data : Mapping
        The raw YAML mapping.

    Returns
    -------
    AnalysisRecipe or AnalysisDiagnostic or HimgJob
        A top-level ``kind`` validates as a ``.himg`` job (neither analysis
        format has one); otherwise ``schema_version: 3`` validates as a
        recipe and anything else goes to the v2 diagnostic, which refuses
        the pre-v2 layout itself.

    Raises
    ------
    pydantic.ValidationError
        When the document does not validate as its declared format.
    """
    if "kind" in data:
        return HimgJob.model_validate(data)
    if declared_schema_version(data) == CURRENT_RECIPE_VERSION:
        return AnalysisRecipe.model_validate(data)
    return AnalysisDiagnostic.model_validate(data)


__all__ = ["AnalysisDocument", "load_analysis_document"]
