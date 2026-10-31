"""An experiment instance's units are rows of their own in ``/fleet-status``.

``<unit>@<Instance>.service`` matches the same ``geecs-*`` listing as the main
experiment's unit; without the instance on the role the two would collapse
into one role ("2 processes for this role") and the proof of a second
experiment — both instance sets visible — could not be read off the table.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="fleet_status.sh and its test need bash",
)

_spec = importlib.util.spec_from_file_location(
    "fleet_status_sh_tests", Path(__file__).with_name("test_fleet_status_sh.py")
)
assert _spec and _spec.loader
_sh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sh)

UNITS = (
    "geecs-qserver.service",
    "geecs-qserver@PulsedWire.service",
    "geecs-scanner.service",
    "geecs-scanner@PulsedWire.service",
)


def test_instance_units_carry_their_instance_on_the_role(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    listing = "\\n".join(f"{u} loaded active running" for u in UNITS)
    (bin_dir / "systemctl").write_text(
        f"""#!/usr/bin/env bash
case "$*" in
    *list-units*) [ -z "${{*##*--user*}}" ] && exit 0; printf '{listing}\\n'; exit 0 ;;
    *"ActiveState"*) echo active ;;
    *"SubState"*) echo running ;;
    *"MainPID"*) echo 0 ;;
    *) exit 0 ;;
esac
"""
    )
    (bin_dir / "ss").write_text("#!/usr/bin/env bash\nexit 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    r = subprocess.run(
        ["bash", "-s"],
        input=_sh.remote_snippet(),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    roles = {
        line.split("\t", 1)[0]
        for line in r.stdout.splitlines()
        if line.startswith("role=")
    }
    assert "role=Queueserver RE Manager" in roles
    assert "role=Queueserver RE Manager @PulsedWire" in roles
    assert "role=GEECS Scanner" in roles
    assert "role=GEECS Scanner @PulsedWire" in roles
