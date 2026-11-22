"""A rerun owns its declared columns on its rows: what it computed, NaN and failure included."""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geecs_analysis.engine import stream
from geecs_data_utils import ScanPaths, ScanTag
from geecs_schemas.analysis import AnalysisRecipe

import scan_analysis.base as base
from scan_analysis.core_analyzer import CoreScanAnalyzer

TAG = ScanTag(year=2026, month=1, day=3, number=1, experiment="Test")
SHOTS = 6
#: A row the scanner recorded but whose shot has no file: outside every run.
UNMAPPED = 7
#: Columns no run of the ``Diag`` recipe owns: the scanner's, another
#: analyzer's, and one under the same output name that the measure never emits.
FOREIGN = {
    "U_Laser Energy": [float(n) for n in range(1, UNMAPPED + 1)],
    "Other_x_CoM": [10.0 * n for n in range(1, UNMAPPED + 1)],
    "Diag_operator_note": [0.5] * UNMAPPED,
}
#: Stands for an old run's value on the unmapped row.
OLD = 4.2


def recipe(*, per_bin: bool = False, save: bool = False) -> AnalysisRecipe:
    scan: dict = {"save": save}
    if per_bin:
        scan["average_frames_first"] = True
    return AnalysisRecipe.model_validate(
        {
            "device": "Camera",
            "output_name": "Diag",
            "input": {"kind": "camera", "file_tail": ".npy"},
            "measure": {"kind": "beam"},
            "scan": scan,
            "figure": {"fig": {"dpi": 20}},
            "summaries": [{"kind": "image_grid"}],
        }
    )


@pytest.fixture
def scan(tmp_path, monkeypatch) -> Path:
    """Six camera shots in three bins, a seventh row with no file, foreign columns."""
    folder = ScanPaths.get_scan_folder_path(tag=TAG, base_directory=tmp_path)
    device = folder / "Camera"
    device.mkdir(parents=True)  # fixture acquisition
    (folder / "ScanInfoScan001.ini").write_text(
        '[Scan Info]\nScan No = "1"\nScan Parameter = "U_Motor:Position"\n'
        'Start = "1"\nEnd = "3"\nStep size = "1"\nShots per step = "2"\n'
    )
    yy, xx = np.mgrid[:16, :16]
    for shot in range(1, SHOTS + 1):
        image = 100 * np.exp(-((xx - 4 - shot) ** 2 + (yy - 8) ** 2) / 4) + 2
        np.save(device / f"Scan001_Camera_{shot:03d}.npy", image)
    rows = pd.DataFrame(
        {
            "Shotnumber": range(1, UNMAPPED + 1),
            "Bin #": [1, 1, 2, 2, 3, 3, 3],
            "U_Motor Position Alias:motor": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0],
            **FOREIGN,
        }
    )
    analysis = folder.parent.parent / "analysis"
    analysis.mkdir()
    rows.to_csv(analysis / "s1.txt", sep="\t", index=False)
    monkeypatch.setattr(base, "ScanPaths", partial(ScanPaths, base_directory=tmp_path))
    return folder


def sfile(scan: Path) -> pd.DataFrame:
    return pd.read_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t")


def sidecar(scan: Path) -> pd.DataFrame:
    path = scan.parent.parent / "analysis" / "Scan001" / "Scan001_Diag.txt"
    return pd.read_csv(path, sep="\t")


def set_cells(scan: Path, column: str, shots: dict[int, float]) -> None:
    rows = sfile(scan)
    for shot, value in shots.items():
        rows.loc[rows["Shotnumber"] == shot, column] = value
    rows.to_csv(scan.parent.parent / "analysis" / "s1.txt", sep="\t", index=False)


def run(doc: AnalysisRecipe) -> CoreScanAnalyzer:
    analyzer = CoreScanAnalyzer(doc, id="Diag", priority=1)
    analyzer.run_analysis(TAG)
    return analyzer


def analysis_fails_for(monkeypatch, shots=None, *, nan_for=()) -> None:
    """Make the measure raise on ``shots`` (all when None), or return NaNs for ``nan_for``."""
    real = stream.analyze_compiled

    def analyze(raw, recipe, *, shot=None, inputs=None):
        number = None if shot is None else shot.shot_number
        if shots is None or number in shots:
            raise RuntimeError("measure refused the frame")
        measurement = real(raw, recipe, shot=shot, inputs=inputs)
        if number in nan_for:
            scalars = {key: float("nan") for key in measurement.scalars}
            measurement = dataclasses.replace(measurement, scalars=scalars)
        return measurement

    monkeypatch.setattr(stream, "analyze_compiled", analyze)


def corrupt(scan: Path, shot: int) -> None:
    """The shot's file exists but cannot be read: a load failure."""
    (scan / "Camera" / f"Scan001_Camera_{shot:03d}.npy").write_bytes(b"not an array")


def diag_columns(rows: pd.DataFrame) -> list[str]:
    return [c for c in rows.columns if c.startswith("Diag_") and c in OWNED]


#: Columns the beam measure wrote on the first run (filled by ``first_run``).
OWNED: set[str] = set()


@pytest.fixture
def first_run(scan) -> pd.DataFrame:
    """A clean run, then an old value planted on the unmapped row."""
    run(recipe())
    OWNED.clear()
    OWNED.update(sidecar(scan).columns.drop("Shotnumber"))
    set_cells(scan, "Diag_x_CoM", {UNMAPPED: OLD})
    return sfile(scan)


def assert_foreign_cells_kept(rows: pd.DataFrame) -> None:
    for column, values in FOREIGN.items():
        assert rows[column].tolist() == values, column


def test_all_units_failing_clears_the_declared_columns(scan, first_run, monkeypatch):
    assert first_run.loc[:5, "Diag_x_CoM"].notna().all()
    analysis_fails_for(monkeypatch)
    run(recipe())
    rows = sfile(scan).set_index("Shotnumber")
    owned = diag_columns(rows)
    assert owned and set(owned) == OWNED
    assert rows.loc[1:SHOTS, owned].isna().all().all()
    # The shot outside the run keeps its old value; nothing else moved.
    assert rows.loc[UNMAPPED, "Diag_x_CoM"] == OLD
    assert list(rows.reset_index().columns) == list(first_run.columns)
    assert_foreign_cells_kept(rows.reset_index())


def test_a_failed_first_run_creates_no_all_nan_columns(scan, monkeypatch):
    before = sfile(scan)
    analysis_fails_for(monkeypatch)
    run(recipe())
    after = sfile(scan)
    assert list(after.columns) == list(before.columns)
    pd.testing.assert_frame_equal(after, before)


def test_a_declared_column_added_on_disk_mid_run_is_cleared(scan, monkeypatch):
    """Whether a declared column exists is decided under the lock, not from the snapshot."""
    real_fail = RuntimeError("measure refused the frame")
    planted = []

    def analyze(raw, recipe, *, shot=None, inputs=None):
        if not planted:
            # Another writer adds a column this recipe declares, after the
            # analyzer loaded its in-memory rows.
            set_cells(scan, "Diag_x_CoM", {n: OLD for n in range(1, UNMAPPED + 1)})
            planted.append(True)
        raise real_fail

    monkeypatch.setattr(stream, "analyze_compiled", analyze)
    run(recipe())
    rows = sfile(scan).set_index("Shotnumber")
    assert rows.loc[1:SHOTS, "Diag_x_CoM"].isna().all()
    assert rows.loc[UNMAPPED, "Diag_x_CoM"] == OLD
    side = sidecar(scan).set_index("Shotnumber")
    assert list(side.columns) == ["Diag_x_CoM"]
    pd.testing.assert_frame_equal(
        side, rows.loc[side.index, side.columns], check_dtype=False
    )


def test_a_run_that_wrote_nothing_leaves_no_earlier_sidecar_values(scan, monkeypatch):
    """An all-failed run on an s-file without the columns replaces the old sidecar."""
    run(recipe())
    assert sidecar(scan)["Diag_x_CoM"].notna().all()
    # The s-file is regenerated without this recipe's columns.
    rows, owned = sfile(scan), set(sidecar(scan).columns) - {"Shotnumber"}
    rows.drop(columns=list(owned)).to_csv(
        scan.parent.parent / "analysis" / "s1.txt", sep="\t", index=False
    )
    analysis_fails_for(monkeypatch)
    run(recipe())
    side = sidecar(scan)
    assert list(side.columns) == ["Shotnumber"]
    assert side["Shotnumber"].tolist() == list(range(1, SHOTS + 1))


def test_mixed_success_and_failure_write_the_same_rows_to_sidecar_and_sfile(
    scan, first_run, monkeypatch
):
    analysis_fails_for(monkeypatch, {2, 5}, nan_for={4})
    run(recipe())
    rows = sfile(scan).set_index("Shotnumber")
    side = sidecar(scan).set_index("Shotnumber")
    assert list(side.index) == list(range(1, SHOTS + 1))
    pd.testing.assert_frame_equal(
        side, rows.loc[side.index, side.columns], check_dtype=False
    )
    assert rows.loc[[2, 4, 5], "Diag_x_CoM"].isna().all()  # failed, NaN, failed
    assert rows.loc[[1, 3, 6], "Diag_x_CoM"].notna().all()
    assert rows.loc[UNMAPPED, "Diag_x_CoM"] == OLD
    assert_foreign_cells_kept(rows.reset_index())


def test_a_shot_that_fails_to_load_keeps_its_cells(scan, first_run, monkeypatch):
    corrupt(scan, 3)
    analysis_fails_for(monkeypatch, {5})
    run(recipe())
    rows = sfile(scan).set_index("Shotnumber")
    old = first_run.set_index("Shotnumber")
    assert rows.loc[3, "Diag_x_CoM"] == old.loc[3, "Diag_x_CoM"]
    assert np.isnan(rows.loc[5, "Diag_x_CoM"])
    assert 3 not in sidecar(scan)["Shotnumber"].tolist()


def test_a_bin_stamps_its_values_on_every_member_including_a_load_failure(
    scan, monkeypatch
):
    """Per-bin: a bin's values land on all its rows, a member that failed to load too."""
    corrupt(scan, 2)
    run(recipe(per_bin=True))
    rows = sfile(scan).set_index("Shotnumber")
    assert rows.loc[1, "Diag_x_CoM"] == rows.loc[2, "Diag_x_CoM"]
    assert rows.loc[3, "Diag_x_CoM"] == rows.loc[4, "Diag_x_CoM"]
    assert rows.loc[1, "Diag_x_CoM"] != rows.loc[3, "Diag_x_CoM"]
    # Bin 3's shot 7 has no file: a member all the same, stamped with its bin.
    assert rows.loc[5, "Diag_x_CoM"] == rows.loc[UNMAPPED, "Diag_x_CoM"]
    pinned = rows.loc[1, "Diag_x_CoM"]
    # A rerun where bin 2 fails after loading clears its members, keeps bin 1.
    real = stream.analyze_compiled

    def analyze(raw, recipe, *, shot=None, inputs=None):
        if abs(float(raw.mean()) - bin_two_mean) < 1e-9:
            raise RuntimeError("measure refused the bin")
        return real(raw, recipe, shot=shot, inputs=inputs)

    frames = [np.load(scan / "Camera" / f"Scan001_Camera_{n:03d}.npy") for n in (3, 4)]
    bin_two_mean = float(np.mean(frames, axis=0).mean())
    monkeypatch.setattr(stream, "analyze_compiled", analyze)
    run(recipe(per_bin=True))
    rows = sfile(scan).set_index("Shotnumber")
    assert rows.loc[[3, 4], "Diag_x_CoM"].isna().all()
    assert rows.loc[1, "Diag_x_CoM"] == rows.loc[2, "Diag_x_CoM"] == pinned


# ---- RunOutcome: every failure the run continues past is counted

#: The beam measure's scalars (the NaN shot emits each once).
OWNED_BEAM_SCALARS = CoreScanAnalyzer(
    recipe(), id="Diag", priority=1
).spec.recipe.analysis.measure.emitted_scalars()


def test_a_clean_run_counts_every_unit_and_nothing_else(scan):
    analyzer = run(recipe())
    assert analyzer.outcome.model_dump() == {
        "units_attempted": SHOTS,
        "units_succeeded": SHOTS,
        "units_failed": 0,
        "shots_load_failed": 0,
        "nan_values": 0,
        "product_write_failures": 0,
        "scalar_write_failures": 0,
        "fold_errors": 0,
    }


def test_failures_load_failures_and_nans_are_counted(scan, monkeypatch):
    corrupt(scan, 3)
    analysis_fails_for(monkeypatch, {2, 5}, nan_for={4})
    outcome = run(recipe()).outcome
    assert outcome.units_attempted == SHOTS
    # 2 and 5 failed analysis; 3 had nothing loadable.
    assert (outcome.units_succeeded, outcome.units_failed) == (3, 3)
    assert outcome.shots_load_failed == 1
    assert outcome.nan_values == len(OWNED_BEAM_SCALARS)


@pytest.mark.parametrize("refused", ["sfile", "sidecar"])
def test_a_refused_scalar_write_is_counted(scan, monkeypatch, refused):
    """A failed scalar write can never read as a clean success."""
    if refused == "sfile":
        monkeypatch.setattr(base, "merge_sfile", lambda *args, **kwargs: None)
    else:
        monkeypatch.setattr(CoreScanAnalyzer, "scalar_sidecar_path", lambda self: None)
    outcome = run(recipe()).outcome
    assert outcome.units_succeeded == SHOTS
    assert outcome.scalar_write_failures == 1


def test_a_skipped_figure_is_a_product_write_failure(scan, monkeypatch):
    from geecs_analysis.render import RenderError

    from scan_analysis import core_sink

    def refuse(*args, **kwargs):
        raise RenderError("cannot draw")

    monkeypatch.setattr(core_sink, "draw_product", refuse)
    outcome = run(recipe(save=True)).outcome
    assert outcome.product_write_failures == 3  # one figure per bin


def test_the_small_scan_figure_gate_is_not_a_failure(scan, monkeypatch):
    """Two successful units draw no figures: expected, never counted."""
    analysis_fails_for(monkeypatch, {1, 2, 3, 4})
    analyzer = run(recipe(save=True))
    assert any("more than two successful units" in n for n in analyzer.last_plan.notes)
    assert analyzer.outcome.product_write_failures == 0
    assert analyzer.outcome.units_succeeded == 2


def test_a_stale_outcome_never_survives_into_the_next_run(scan, monkeypatch):
    analyzer = CoreScanAnalyzer(recipe(), id="Diag", priority=1)
    with monkeypatch.context() as patch:
        analysis_fails_for(patch, {1})
        analyzer.run_analysis(TAG)
    analyzer.cleanup()  # every host's duty after a run; the outcome outlives it
    assert analyzer.outcome.units_failed == 1
    analyzer.run_analysis(TAG)  # a clean run on the same analyzer: fresh counts
    analyzer.cleanup()
    assert analyzer.outcome.units_attempted == SHOTS
    assert analyzer.outcome.units_failed == 0
    # A run that returns early (no s-file) leaves no outcome at all.
    (scan.parent.parent / "analysis" / "s1.txt").unlink()
    assert analyzer.run_analysis(TAG) is None
    assert analyzer.outcome is None
    # Nor does one that refuses before any unit runs (no device data).
    pd.DataFrame({"Shotnumber": [1], "Bin #": [1]}).to_csv(
        scan.parent.parent / "analysis" / "s1.txt", sep="\t", index=False
    )
    analyzer.run_analysis(TAG)
    analyzer.cleanup()
    assert analyzer.outcome is not None
    for path in (scan / "Camera").iterdir():
        path.unlink()
    with pytest.raises(base.DataUnavailableWarning):
        analyzer.run_analysis(TAG)
    assert analyzer.outcome == base.RunOutcome()


def _counts(**given: int) -> dict:
    """A clean six-unit run's counts, then *given*."""
    return {"units_attempted": 6, "units_succeeded": 6, **given}


@pytest.mark.parametrize(
    ("counts", "state", "degraded"),
    [
        pytest.param(_counts(), "done", False, id="clean"),
        pytest.param(
            _counts(units_attempted=0, units_succeeded=0), "no_data", False, id="none"
        ),
        pytest.param(
            _counts(units_succeeded=0, units_failed=6), "failed", False, id="all-failed"
        ),
        pytest.param(
            _counts(units_succeeded=5, units_failed=1), "done", True, id="unit-failed"
        ),
        pytest.param(_counts(shots_load_failed=1), "done", True, id="load-failed"),
        pytest.param(
            _counts(product_write_failures=1), "done", True, id="product-not-written"
        ),
        pytest.param(
            _counts(scalar_write_failures=1), "done", True, id="scalar-write-failed"
        ),
        # The small-scan figure gate draws nothing and counts nothing.
        pytest.param(
            _counts(units_attempted=2, units_succeeded=2), "done", False, id="small"
        ),
        pytest.param(_counts(nan_values=18), "done", False, id="nan-only"),
    ],
)
def test_the_state_table(counts, state, degraded):
    outcome = base.RunOutcome(**counts)
    verdict = outcome.verdict()
    assert (verdict.state, verdict.degraded) == (state, degraded)
    assert (verdict.message is not None) == (state != "done")
    report = outcome.report()
    assert report == {
        "counts": outcome.model_dump(),
        "state": state,
        "degraded": degraded,
        "message": verdict.message,
    }
    assert set(base.DEGRADING_COUNTS) <= set(base.RunOutcome.model_fields)
