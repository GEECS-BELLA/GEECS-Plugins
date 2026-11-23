"""A fake data share: one day, a stepped scan and a no-scan, tiny s-files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from geecs_portal.plotter.app import create_app

EXPERIMENT = "Testexp"
DAY = date(2026, 10, 2)
SCAN_VARIABLE = "U_Jet Position"
#: A Master Control column carrying a LabVIEW alias.
ALIASED = "U_Gas Pressure Alias:Gas (psi)"
#: Scan001's camera (the ``camera`` fixture).
CAM = "UC_Cam"


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
    """The app's settings folder: never this machine's own state folder."""
    return tmp_path / "plotter-settings"


class FakeConfigurator:
    """The configurator as the portal sees it, in memory.

    A save writes ``<stem>.yaml`` into the shared folder at once, as a
    commit followed by the share's fast-forward would, and records who made
    it; a create over a file that is there is refused, as the real one is.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.saves: list[tuple[str, str]] = []  # (stem, operator)

    def save(self, stem: str, document: dict, *, operator: str, overwrite: bool) -> str:
        from geecs_portal.plotter.shared_sets import taken

        path = self.folder / f"{stem}.yaml"
        if path.exists() and not overwrite:
            raise taken(stem)
        self.folder.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.saves.append((stem, operator))
        return f"committed {stem} as {operator}'s"


@pytest.fixture
def configs_root(tmp_path: Path) -> Path:
    """The configs repo's root, with no shared sets yet."""
    return tmp_path / "configs"


@pytest.fixture
def shared_dir(configs_root: Path) -> Path:
    """Where the experiment's shared sets are read from."""
    from geecs_portal.plotter.shared_sets import shared_folder

    return shared_folder(configs_root, EXPERIMENT)


@pytest.fixture
def configurator(shared_dir: Path) -> FakeConfigurator:
    """The configurator the app shares sets through."""
    return FakeConfigurator(shared_dir)


@pytest.fixture
def client(
    share: Path,
    settings_dir: Path,
    configs_root: Path,
    configurator: FakeConfigurator,
) -> TestClient:
    """The app over the fake share, sharing sets through the fake configurator."""
    return TestClient(
        create_app(
            EXPERIMENT,
            base_directory=share,
            settings_dir=settings_dir,
            configs_root=configs_root,
            set_saver=configurator,
        )
    )


def post(client: TestClient, path: str = "/", **fields: object) -> str:
    """POST the page form (day 2026-10-02 unless given) and follow to the page.

    The viewer is the set's own owner unless ``owner`` says otherwise.
    """
    form = {"day": "2026-10-02", "set": "default", **fields}
    form.setdefault("owner", form["set"])
    return client.post(path, data={k: str(v) for k, v in form.items()}).text


@pytest.fixture
def camera(share: Path) -> Path:
    """Scan001's camera and vendor folders; shot 3 has no file."""
    folder = share / EXPERIMENT / "Y2026" / "10-Oct" / "26_1002" / "scans" / "Scan001"
    cam = folder / CAM
    cam.mkdir()
    for shot in (1, 2, 4):
        frame = np.full((4, 6), 10 * shot, dtype=np.uint16)
        frame[0, 0] = 1000  # a bright corner, so the window is not flat
        Image.fromarray(frame).save(cam / f"Scan001_{CAM}_{shot:03d}.png")
    haso = folder / "U_Haso"
    haso.mkdir()
    (haso / "Scan001_U_Haso_001.himg").write_bytes(b"proprietary")
    return folder
