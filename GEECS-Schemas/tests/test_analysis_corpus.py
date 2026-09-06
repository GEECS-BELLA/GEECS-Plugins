"""Corpus walk for the analysis-config documents (integration; skips without the sibling checkout).

Every diagnostic under ``scan_analysis_configs/analyzers/<namespace>/`` and
every group under ``scan_analysis_configs/groups/`` in the sibling
GEECS-Plugins-Configs checkout must validate as v2 — directly once the
corpus is regenerated, through the one-shot converter while it is still v1.

Two documented exceptions:

* ``analyzers/UNCLASSIFIED/`` holds legacy *flat camera configs* (no
  ``image_analyzer``, no ``image:`` wrapper) that the 2026 unified-config
  migration copied over without a scan pairing.  They never loaded as
  diagnostics and are slated for deletion, not conversion — the walk skips
  the folder.
* :data:`KNOWN_INVALID` lists files the v2 schema refuses on purpose (the
  fix belongs in the configs repo); empty since the corpus regeneration.
"""

from __future__ import annotations

import pytest
import yaml

from geecs_schemas.analysis import AnalysisDiagnostic, AnalysisGroup
from geecs_schemas.convert.analysis_diagnostics import convert_v1_diagnostic


#: namespace/stem → why the v2 schema refuses it (fix belongs in the configs
#: repo). Empty since the corpus was regenerated in v2 (branch
#: analysis-config-v2 there, 2026-09-05) — U_FROG_Beam's ignored FROG keys
#: went with it.
KNOWN_INVALID: dict[str, str] = {}

SKIPPED_NAMESPACES = {"UNCLASSIFIED"}


def diagnostics(configs):
    root = configs / "scan_analysis_configs" / "analyzers"
    return sorted(
        path
        for path in root.glob("*/*.y*ml")
        if path.parent.name not in SKIPPED_NAMESPACES
    )


def groups(configs):
    return sorted((configs / "scan_analysis_configs" / "groups").glob("*/*.y*ml"))


@pytest.mark.integration
class TestAnalysisCorpus:
    def test_every_diagnostic_converts_to_v2(self, configs_repo):
        failures = {}
        lifted = 0
        for path in diagnostics(configs_repo):
            key = f"{path.parent.name}/{path.stem}"
            try:
                diag = AnalysisDiagnostic.model_validate(
                    convert_v1_diagnostic(yaml.safe_load(path.read_text()))
                )
            except Exception as exc:  # noqa: BLE001 — collected and reported below
                failures[key] = str(exc).splitlines()[0]
                continue
            assert diag.schema_version == 2
            # stable through a dump: the editor's save path
            assert (
                AnalysisDiagnostic.model_validate(diag.model_dump(mode="json")) == diag
            )
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
