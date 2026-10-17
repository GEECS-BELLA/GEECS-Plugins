"""ResolvedDiagnosticConfig — the group loader's wrapper around a recipe.

The documents themselves (``AnalysisRecipe``, ``AnalysisGroup``) are tested
in GEECS-Schemas.
"""

from __future__ import annotations

import pytest
from geecs_schemas.analysis import AnalysisRecipe
from pydantic import ValidationError

from scan_analysis.config import ResolvedDiagnosticConfig


class TestResolvedDiagnosticConfig:
    def _recipe(self, **scan) -> AnalysisRecipe:
        return AnalysisRecipe.model_validate(
            {
                "device": "UC_Test",
                "input": {"kind": "camera"},
                "measure": {"kind": "beam"},
                "scan": scan,
            }
        )

    def test_pairs_a_recipe_with_its_id_and_priority(self):
        resolved = ResolvedDiagnosticConfig(
            id="UC_Test_file", priority=7, diagnostic=self._recipe(priority=50)
        )
        assert resolved.enabled is True
        assert resolved.priority == 7
        assert resolved.diagnostic.scan.priority == 50

    def test_priority_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ResolvedDiagnosticConfig(id="x", priority=-1, diagnostic=self._recipe())

    def test_unknown_field_refused(self):
        with pytest.raises(ValidationError):
            ResolvedDiagnosticConfig(
                id="x", priority=1, diagnostic=self._recipe(), extra_field=True
            )
