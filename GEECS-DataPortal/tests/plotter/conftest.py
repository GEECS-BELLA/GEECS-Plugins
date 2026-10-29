"""A fake data share: one day, a stepped scan and a no-scan, tiny s-files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from geecs_portal.plotter.app import create_app

EXPERIMENT = "Testexp"
DAY = date(2026, 10, 2)
SCAN_VARIABLE = "U_Jet Position"
#: A Master Control column carrying a LabVIEW alias.
ALIASED = "U_Gas Pressure Alias:Gas (psi)"


def write_scan(
    root: Path, number: int, frame: pd.DataFrame, parameter: str = "Shotnumber"
) -> Path:
    """Write ``scans/ScanNNN/`` (with its ScanInfo) and ``analysis/sN.txt``."""
    day = root / EXPERIMENT / "Y2026" / "10-Oct" / "26_1002"
    folder = day / "scans" / f"Scan{number:03d}"
    folder.mkdir(parents=True)
    (folder / f"ScanInfoScan{number:03d}.ini").write_text(
        "[Scan Info]\n"
        f'Scan No = "{number}"\n'
        f'Scan Parameter = "{parameter}"\n'
        'Start = "1"\nEnd = "2"\nStep size = "1"\nShots per step = "2"\n'
    )
    (day / "analysis").mkdir(exist_ok=True)
    frame.to_csv(day / "analysis" / f"s{number}.txt", sep="\t", index=False)
    return folder


@pytest.fixture
def share(tmp_path: Path) -> Path:
    """Scan001 steps the jet over two positions; Scan002 is a no-scan."""
    write_scan(
        tmp_path,
        1,
        pd.DataFrame(
            {
                "Shotnumber": [1, 2, 3, 4],
                "Bin #": [1, 1, 2, 2],
                SCAN_VARIABLE: [1.0, 1.0, 2.0, 2.0],
                "U_ICT charge": [10.0, 12.0, float("nan"), 30.0],
                ALIASED: [1.5, 1.5, 2.5, 2.5],
            }
        ),
        parameter=SCAN_VARIABLE,
    )
    write_scan(
        tmp_path,
        2,
        pd.DataFrame({"Shotnumber": [1, 2], "U_ICT charge": [5.0, 6.0]}),
    )
    return tmp_path


@pytest.fixture
def settings_dir(tmp_path: Path) -> Path:
    """The app's settings folder: never the real user data folder."""
    return tmp_path / "plotter-settings"


@pytest.fixture
def client(share: Path, settings_dir: Path) -> TestClient:
    """The app over the fake share."""
    return TestClient(
        create_app(EXPERIMENT, base_directory=share, settings_dir=settings_dir)
    )


def post(client: TestClient, path: str = "/", **fields: object) -> str:
    """POST the page form (day 2026-10-02 unless given) and follow to the page.

    The viewer is the set's own owner unless ``owner`` says otherwise.
    """
    form = {"day": "2026-10-02", "set": "default", **fields}
    form.setdefault("owner", form["set"])
    return client.post(path, data={k: str(v) for k, v in form.items()}).text
