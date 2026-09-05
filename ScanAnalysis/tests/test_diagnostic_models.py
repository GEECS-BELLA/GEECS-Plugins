"""The scan-side config models moved to GEECS-Schemas; pin the aliases and the loader wrapper.

The document models (``AnalysisDiagnostic``, ``ScanRuntime``,
``BackgroundSource``, ``AnalysisGroup``) are tested in
``GEECS-Schemas/tests/test_analysis_diagnostic.py``.  What ScanAnalysis
still owns is the transitional names and :class:`ResolvedDiagnosticConfig`.
"""

from __future__ import annotations

import pytest
from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisGroup,
    BackgroundSource,
    ScanRuntime,
)
from pydantic import ValidationError

from scan_analysis.config import (
    AnalysisGroupConfig,
    DiagnosticAnalysisConfig,
    ResolvedDiagnosticConfig,
    ScanRuntimeConfig,
)


def test_transitional_aliases_point_at_the_schema_models():
    assert ScanRuntimeConfig is ScanRuntime
    assert AnalysisGroupConfig is AnalysisGroup
    assert DiagnosticAnalysisConfig is AnalysisDiagnostic


def test_scan_runtime_defaults_through_the_alias():
    cfg = ScanRuntimeConfig()
    assert cfg.priority == 100
    assert cfg.mode == "per_shot"
    assert cfg.save is True
    assert cfg.gdoc_slot is None
    assert cfg.renderer.as_kwargs() == {}
    assert cfg.background_source is None


def test_background_source_still_requires_exactly_one_variant():
    with pytest.raises(ValidationError, match="exactly one source"):
        BackgroundSource()
    with pytest.raises(ValidationError, match="exactly one source"):
        BackgroundSource(scan_number=1, autodetect={})


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
