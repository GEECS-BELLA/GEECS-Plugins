"""Shared fixtures: a synthetic share that mimics the real folder layout."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

DAY = date(2026, 9, 11)

#: A completed scan, as the Bluesky path writes it.
SUCCESS_INI = """[Scan Info]
Scan No = {n}
ScanStartInfo = "807 phase 1 acceptance: preset through the manager"
Scan Parameter = "U_S1H Current"
Start = -1.0
End = 1.0
Step size = 0.5
Shots per step = 2
ScanEndInfo = "success"
Background = false
ScanMode = "standard"
Scanner = "bluesky"
Plan = "scan"
Trigger profile = "HTU-NoGas"
"""

#: A real failure, transcribed from Scan006 on 2026-09-11.
FAILED_INI = """[Scan Info]
Scan No = {n}
ScanStartInfo = "806 acceptance: plugin camera count"
Scan Parameter = "Shotnumber"
Start = 0.0
End = 0.0
Step size = 0.0
Shots per step = 5
ScanEndInfo = "fail: <AsyncStatus, device: uc_amp4_ir_input, errored: TimeoutError(\
"uc_amp4_ir_input-hdf-capture didn't match True in 10.0s, last value False")>"
Background = false
ScanMode = "noscan"
Scanner = "bluesky"
Plan = "count"
Trigger profile = "HTU-NoGas"
"""


@pytest.fixture
def make_run(tmp_path: Path):
    """Return a builder for a day of N identical successful scans."""

    def build(count: int, day_folder: str = "26_0911") -> Path:
        scans = tmp_path / "Undulator" / "Y2026" / "09-Sep" / day_folder / "scans"
        scans.mkdir(parents=True, exist_ok=True)
        for n in range(1, count + 1):
            folder = scans / f"Scan{n:03d}"
            folder.mkdir()
            (folder / f"ScanInfoScan{n:03d}.ini").write_text(SUCCESS_INI.format(n=n))
        return tmp_path

    return build


@pytest.fixture
def share(tmp_path: Path) -> Path:
    """Build a share with one successful, one failed and one bare scan."""
    scans = tmp_path / "Undulator" / "Y2026" / "09-Sep" / "26_0911" / "scans"
    scans.mkdir(parents=True)

    ok = scans / "Scan001"
    (ok / "UC_Amp4_IR_input").mkdir(parents=True)
    (ok / "ScanInfoScan001.ini").write_text(SUCCESS_INI.format(n=1))
    (ok / "ScanDataScan001.txt").write_text("shotnumber\n1\n")
    (ok / "scan.log").write_text("")

    bad = scans / "Scan006"
    (bad / "UC_Amp4_IR_input").mkdir(parents=True)
    (bad / "ScanInfoScan006.ini").write_text(FAILED_INI.format(n=6))
    (bad / "scan.log").write_text("")

    # Development churn: a folder with only a log, no ScanInfo at all.
    bare = scans / "Scan031"
    bare.mkdir()
    (bare / "scan.log").write_text("")

    # Not a scan; must be ignored.
    (scans / "notes.txt").write_text("ignore me")

    return tmp_path
