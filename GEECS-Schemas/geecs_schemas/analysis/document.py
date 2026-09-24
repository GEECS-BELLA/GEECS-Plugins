"""Read either analysis document by its declared format version.

One tree holds both formats while the corpus converts: the v3
:class:`~geecs_schemas.analysis.recipe.AnalysisRecipe` for every recipe
the analysis core serves and the v2
:class:`~geecs_schemas.analysis.diagnostic.AnalysisDiagnostic` for the
analyzer kinds it has not ported.  Every loader dispatches here so a
file's format is read once, from ``schema_version``, and never guessed
from its keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from geecs_schemas._base import declared_schema_version
from geecs_schemas.analysis.diagnostic import AnalysisDiagnostic
from geecs_schemas.analysis.recipe import CURRENT_RECIPE_VERSION, AnalysisRecipe

#: What a loader hands back: one of the two document formats.
AnalysisDocument = Union[AnalysisDiagnostic, AnalysisRecipe]


def load_analysis_document(data: Mapping[str, object]) -> AnalysisDocument:
    """Validate a raw document as the format its ``schema_version`` declares.

    Parameters
    ----------
    data : Mapping
        The raw YAML mapping.

    Returns
    -------
    AnalysisRecipe or AnalysisDiagnostic
        ``schema_version: 3`` validates as a recipe; anything else goes to
        the v2 diagnostic, which refuses the pre-v2 layout itself.

    Raises
    ------
    pydantic.ValidationError
        When the document does not validate as its declared format.
    """
    if declared_schema_version(data) == CURRENT_RECIPE_VERSION:
        return AnalysisRecipe.model_validate(data)
    return AnalysisDiagnostic.model_validate(data)


__all__ = ["AnalysisDocument", "load_analysis_document"]
