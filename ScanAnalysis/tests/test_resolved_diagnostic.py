"""ResolvedDiagnosticConfig — the group loader's wrapper around a diagnostic.

The documents themselves (``AnalysisDiagnostic``, ``ScanRuntime``,
``BackgroundSource``, ``AnalysisGroup``) are tested in
``GEECS-Schemas/tests/test_analysis_diagnostic.py``.
"""

from __future__ import annotations

import pytest
from geecs_schemas.analysis import AnalysisDiagnostic
from pydantic import ValidationError

from scan_analysis.config import ResolvedDiagnosticConfig


class TestResolvedDiagnosticConfig:
    def _diag(self, **scan) -> AnalysisDiagnostic:
        return AnalysisDiagnostic(
            name="UC_Test",
            analyzer={"kind": "beam"},
            image={"type": "camera", "bit_depth": 16},
            scan=scan,
        )

    def test_pairs_a_diagnostic_with_its_id_and_priority(self):
        resolved = ResolvedDiagnosticConfig(
            id="UC_Test_file", priority=7, diagnostic=self._diag(priority=50)
        )
        assert resolved.enabled is True
        assert resolved.priority == 7
        assert resolved.diagnostic.scan.priority == 50

    def test_priority_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ResolvedDiagnosticConfig(id="x", priority=-1, diagnostic=self._diag())

    def test_unknown_field_refused(self):
        with pytest.raises(ValidationError):
            ResolvedDiagnosticConfig(
                id="x", priority=1, diagnostic=self._diag(), extra_field=True
            )
