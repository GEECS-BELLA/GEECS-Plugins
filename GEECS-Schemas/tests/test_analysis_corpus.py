"""Corpus walk for the analysis-config documents (integration; skips without the sibling checkout).

Every diagnostic under ``scan_analysis_configs/analyzers/<namespace>/`` and
every group under ``scan_analysis_configs/groups/`` in the sibling
GEECS-Plugins-Configs checkout must validate as the format its
``schema_version`` declares: the v3 recipe for what the analysis core
serves, the v2 diagnostic for the rest (the corpus was regenerated in v2
once for 0.19.0; the core-served recipes convert to v3 on the configs
branch ``analysis-recipe-v3``, so either mix is a valid checkout).

Every namespace is walked, ``UNCLASSIFIED`` included: the legacy flat
camera configs that used to live there were deleted with the corpus
regeneration, and what remains are real v2 diagnostics. A stray file the
schema refuses fails the walk unless :data:`KNOWN_INVALID` names it (the
fix belongs in the configs repo); the dict is empty since the regeneration.
"""

from __future__ import annotations

import pytest
import yaml

from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisGroup,
    AnalysisRecipe,
    load_analysis_document,
)


#: namespace/stem → why the v2 schema refuses it (fix belongs in the configs
#: repo). Empty since the corpus was regenerated in v2 (branch
#: analysis-config-v2 there, 2026-09-05) — U_FROG_Beam's ignored FROG keys
#: went with it.
KNOWN_INVALID: dict[str, str] = {}


def diagnostics(configs):
    root = configs / "scan_analysis_configs" / "analyzers"
    return sorted(root.glob("*/*.y*ml"))


def groups(configs):
    return sorted((configs / "scan_analysis_configs" / "groups").glob("*/*.y*ml"))


@pytest.mark.integration
class TestAnalysisCorpus:
    def test_every_diagnostic_validates_as_its_declared_format(self, configs_repo):
        failures = {}
        lifted = 0
        for path in diagnostics(configs_repo):
            key = f"{path.parent.name}/{path.stem}"
            try:
                diag = load_analysis_document(yaml.safe_load(path.read_text()))
            except Exception as exc:  # noqa: BLE001 — collected and reported below
                failures[key] = str(exc).splitlines()[0]
                continue
            model = (
                AnalysisRecipe
                if isinstance(diag, AnalysisRecipe)
                else AnalysisDiagnostic
            )
            assert diag.schema_version == (3 if model is AnalysisRecipe else 2)
            # stable through a dump: the editor's save path
            assert model.model_validate(diag.model_dump(mode="json")) == diag
            lifted += 1
        unexpected = {k: v for k, v in failures.items() if k not in KNOWN_INVALID}
        assert not unexpected, unexpected
        missing = set(KNOWN_INVALID) - set(failures)
        assert not missing, f"KNOWN_INVALID entries now validate — drop them: {missing}"
        assert lifted >= 40

    def test_every_group_validates(self, configs_repo):
        for path in groups(configs_repo):
            group = AnalysisGroup.model_validate(yaml.safe_load(path.read_text()))
            assert group.schema_version == 1, path
