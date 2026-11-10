"""A scratch configs tree (one experiment, the analysis root) and an app over it."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.settings import Settings

#: path below the configs root -> YAML text
FILES = {
    "scanner_configs/experiments/Bench/shot_control_configurations/laser_off.yaml": """\
schema_version: 2
name: laser_off
states:
  SCAN:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '4.0'}
  STANDBY:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '0.5'}
""",
    "scanner_configs/experiments/Bench/shot_control_configurations/spare.yml": "schema_version: 2\nname: spare\n",
    # a bare OFF: key, which YAML 1.1 reads as a boolean (the model maps it back)
    "scanner_configs/experiments/Bench/shot_control_configurations/bare_off.yaml": """\
schema_version: 2
name: bare_off
states:
  OFF:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '0.5'}
  SCAN:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '4.0'}
""",
    # the legacy dialect (no schema_version), which the reader converts
    "scanner_configs/experiments/Bench/shot_control_configurations/legacy.yaml": """\
device: U_DG645
variables:
  Amplitude.Ch AB: {'OFF': '0.5', SCAN: '4.0', STANDBY: '0.5'}
""",
    "scanner_configs/experiments/Bench/setups/low_power.yaml": """\
name: low_power
steps:
  - {do: set, device: U_PS, variable: Current, value: 1.5}
  - {do: wait, seconds: 2}
""",
    "scanner_configs/experiments/Bench/setups/renamed.yaml": "name: other\nsteps:\n  - {do: wait, seconds: 1}\n",
    "scanner_configs/experiments/Bench/setups/broken.yaml": "name: [unclosed\n",
    "scanner_configs/experiments/Bench/archiver/archive_policy.yaml": "include_setpoints: true\n",
    "scan_analysis_configs/analyzers/Ns/compact.yaml": "kind: himg_compact\ndevice: CamA\n",
    "scan_analysis_configs/analyzers/Ns/recipe.yaml": "device: CamB\n",
    "scan_analysis_configs/groups/base.yaml": "name: base\nanalyzers: []\n",
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "scanner_configs/experiments/Empty").mkdir()
    return tmp_path


@pytest.fixture
def client(root: Path) -> TestClient:
    settings = Settings(configs_root=root, experiment="Bench", cors_origins=("http://surface.test",))  # fmt: skip
    with TestClient(create_app(settings)) as c:
        yield c
