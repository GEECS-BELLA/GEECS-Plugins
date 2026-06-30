"""Session-wide test fixtures for ImageAnalysis.

Initialises the unified analysis config base directory once per test session so
that string-based YAML config lookups work on machines that have the configs
repo mounted.  On CI (where the configs repo is absent) the fixture is a
no-op — integration tests skip themselves via individual path checks.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_analysis_config():
    try:
        from geecs_data_utils.scan_data import ScanPaths
        from geecs_data_utils.config_roots import scan_analysis_config

        scan_analysis_config.set_base_dir(
            ScanPaths.paths_config.scan_analysis_configs_path
        )
    except Exception:
        pass
