"""Pin the light-package-import contract (PEP 562 lazy device re-exports).

Every queue client depends on it: ``import geecs_bluesky`` /
``import geecs_bluesky.qs_client`` must
not pull the heavy device stack (ophyd-async/aioca) or the queueserver
api — a client that only submits scans imports light.  Runs in a
subprocess so the assertion is immune to whatever the pytest process has
already imported.
"""

from __future__ import annotations

import subprocess
import sys

_LIGHT_IMPORT_CODE = """
import sys

import geecs_bluesky

for heavy in ("aioca", "ophyd_async", "bluesky_queueserver_api"):
    assert heavy not in sys.modules, f"package import pulled {heavy}"

import geecs_bluesky.qs_client
from geecs_bluesky.qs_client import make_queue_client  # noqa: F401

# The log-marker re-export must stay light too — resolving the NAME, not
# just importing the package (a PEP 562 lazy shim once passed the import
# leg while attribute access dragged in bluesky + aioca; the constant
# now lives in the import-light log_markers module).
from geecs_bluesky.qs_client import FAILED_MOVE_LOG_PREFIX

assert FAILED_MOVE_LOG_PREFIX.startswith("FAILED MOVE")

for heavy in ("aioca", "ophyd_async", "bluesky_queueserver_api", "bluesky"):
    assert heavy not in sys.modules, f"qs_client import pulled {heavy}"

# The scanner's other light imports go through package __init__s.
import geecs_bluesky.actions.steps  # noqa: F401
import geecs_bluesky.optimization.events  # noqa: F401

for heavy in ("aioca", "ophyd_async", "bluesky", "xopt"):
    assert heavy not in sys.modules, f"scanner light imports pulled {heavy}"

# The lazy re-exports still resolve (this leg MAY pull the device stack).
from geecs_bluesky import CaMotor

assert CaMotor.__name__ == "CaMotor"
"""


def test_package_and_qs_client_import_light() -> None:
    """The package and qs_client import without the heavy device stack."""
    result = subprocess.run(
        [sys.executable, "-c", _LIGHT_IMPORT_CODE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
