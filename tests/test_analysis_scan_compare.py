"""The scan comparison harness runs both factory routes on private copies and diffs them."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "analysis_scan_compare",
    Path(__file__).resolve().parents[1] / "scripts/analysis_scan_compare.py",
)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)

pytest.importorskip("scan_analysis")
pytest.importorskip("geecs_analysis")


def _archive(base: Path) -> tuple[Path, Path]:
    """A completed beam scan in the GEECS layout plus its diagnostic YAML."""
    from geecs_data_utils import ScanPaths, ScanTag

    tag = ScanTag(year=2026, month=1, day=1, number=7, experiment="Test")
    scan = ScanPaths.get_scan_folder_path(tag=tag, base_directory=base)
    device = scan / "Camera"
    device.mkdir(parents=True)  # fixture acquisition, not analysis
    (scan / "ScanInfoScan007.ini").write_text(
        '[Scan Info]\nScan No = "7"\nScan Parameter = "U_Motor:Position"\n'
    )
    yy, xx = np.mgrid[:24, :24]
    for shot in range(1, 7):
        image = 200 * np.exp(-((xx - 6 - shot) ** 2 + (yy - 12) ** 2) / 8) + 5
        np.save(device / f"Scan007_Camera_{shot:03d}.npy", image)
    analysis = scan.parent.parent / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        {
            "Shotnumber": range(1, 7),
            "Bin #": [1, 1, 2, 2, 3, 3],
            "U_Motor Position Alias:motor": [1.0, 1.05, 2.0, 2.05, 3.0, 3.05],
        }
    ).to_csv(analysis / "s7.txt", sep="\t", index=False)
    (analysis / "Scan007").mkdir()
    (analysis / "Scan007" / "unrelated.txt").write_text("must not be copied\n")
    diagnostic = base / "Camera.yaml"
    diagnostic.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "name": "Camera",
                "analyzer": {"kind": "beam"},
                "image": {"type": "camera", "pipeline": []},
                "scan": {"file_tail": ".npy", "renderer": {"dpi": 30}},
            }
        )
    )
    return scan, diagnostic


def test_both_routes_match_on_a_synthetic_scan_and_the_source_is_untouched(
    tmp_path, capsys
):
    scan, diagnostic = _archive(tmp_path / "share")
    before = sorted(
        p.relative_to(tmp_path / "share") for p in (tmp_path / "share").rglob("*")
    )
    output = tmp_path / "compare"
    rc = harness.main(
        ["--diagnostic", str(diagnostic), "--scan", str(scan), "--output", str(output)]
    )
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert printed.rstrip().endswith("MATCH")
    assert (
        sorted(
            p.relative_to(tmp_path / "share") for p in (tmp_path / "share").rglob("*")
        )
        == before
    )
    for route in ("legacy", "core"):
        tree = output / route
        assert list(tree.rglob("*_averaged_image_grid.png"))
        assert not list(tree.rglob("unrelated.txt"))


def test_overrides_reach_the_document_and_a_dirty_output_is_refused(tmp_path):
    scan, diagnostic = _archive(tmp_path / "share")
    from image_analysis.config import load_diagnostic

    document = harness._apply_overrides(
        load_diagnostic(diagnostic), ["scan.data_format=per_shot_files"]
    )
    assert document.scan.data_format == "per_shot_files"
    with pytest.raises(SystemExit, match="scan.<field>"):
        harness._apply_overrides(document, ["image.type=line"])
    output = tmp_path / "compare"
    output.mkdir()
    (output / "stale").touch()
    with pytest.raises(SystemExit, match="must be empty"):
        harness.main(
            [
                "--diagnostic",
                str(diagnostic),
                "--scan",
                str(scan),
                "--output",
                str(output),
            ]
        )


def test_compare_reports_every_kind_of_difference():
    frame = pd.DataFrame({"Shotnumber": [1, 2], "x": [1.0, 2.0]})
    legacy = {
        "a.h5": ("h5", "image", np.ones((2, 2)), np.dtype("float64")),
        "b.h5": ("h5", "image", np.ones((2, 2)), np.dtype("float64")),
        "Device_average_processed.h5": (
            "h5",
            "image",
            np.ones((2, 2)),
            np.dtype("float64"),
        ),
        "s7.txt": ("table", frame),
        "only_legacy.png": ("bytes", 10),
    }
    core = {
        "a.h5": ("h5", "image", np.ones((2, 2)), np.dtype("float64")),
        "b.h5": ("h5", "image", np.ones((2, 2)) + 1e-9, np.dtype("float64")),
        "Device_average_processed.h5": (
            "h5",
            "image",
            np.ones((2, 2)) * (1 + 1e-16),
            np.dtype("float64"),
        ),
        "s7.txt": ("table", frame.assign(x=[1.0, 3.0])),
        "only_core.png": ("bytes", 10),
    }
    problems = harness.compare(legacy, core, average_ulps=4)
    assert [
        p.split(":")[0].split(" ")[-1] for p in problems if p.startswith("only")
    ] == ["only_legacy.png", "only_core.png"]
    assert any(p.startswith("b.h5: arrays differ") for p in problems)
    assert not any(p.startswith("Device_average_processed.h5") for p in problems)
    assert any(p.startswith("s7.txt:") for p in problems)
    assert harness.compare(legacy, legacy, average_ulps=4) == []
