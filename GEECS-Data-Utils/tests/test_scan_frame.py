"""Hermetic tests for the s-file reader and the union-with-provenance frame.

Synthetic s-files in tmp trees + hand-built RunDetails — no network, no
data root, no config.ini.
"""

from __future__ import annotations

import pandas as pd
import pytest

from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan
from geecs_data_utils.scan_frame import (
    PROVENANCE_RUN,
    PROVENANCE_SFILE,
    scan_frame,
)
from geecs_data_utils.tiled_catalog import RunDetail, summary_from_metadata


def _detail(frame) -> RunDetail:
    start = {"uid": "u", "time": 1.0, "scan_number": 2, "experiment": "Undulator"}
    return RunDetail(
        summary=summary_from_metadata("u", start, {"exit_status": "success"}),
        start_doc=start,
        stop_doc={},
        data=frame,
    )


def _event_frame(n=3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scan_event_index": list(range(1, n + 1)),  # 1-based by schema
            "cam-fwhm": [10.0 * i for i in range(1, n + 1)],
        }
    )


def _write_sfile(tmp_path, rows, name="s2.txt", columns=("Shotnumber", "U_ICT charge")):
    day = tmp_path / "26_0829"
    scans = day / "scans" / "Scan002"
    scans.mkdir(parents=True, exist_ok=True)
    analysis = day / "analysis"
    analysis.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(columns))
    frame.to_csv(analysis / name, sep="\t", index=False)
    return scans, analysis / name


class TestSfilePath:
    def test_convention_is_day_analysis_unpadded(self, tmp_path):
        scans, sfile = _write_sfile(tmp_path, [[1, 5.0]])
        assert sfile_path_for_scan(scans) == sfile
        assert sfile.name == "s2.txt"  # unpadded, per the LabVIEW convention

    def test_non_canonical_folder_raises(self, tmp_path):
        with pytest.raises(ValueError):
            sfile_path_for_scan(tmp_path / "scratch" / "ScanX")
        with pytest.raises(ValueError):
            sfile_path_for_scan(tmp_path / "notscans" / "Scan002")

    def test_read_sfile_headers_verbatim(self, tmp_path):
        _, sfile = _write_sfile(tmp_path, [[1, 5.0], [2, 6.5]])
        frame = read_sfile(sfile)
        assert list(frame.columns) == ["Shotnumber", "U_ICT charge"]
        assert frame["U_ICT charge"].tolist() == [5.0, 6.5]


class TestScanFrame:
    def test_union_joins_on_shot_identity_one_based(self, tmp_path):
        # THE join contract: Shotnumber == scan_event_index, both 1-based
        # (no off-by-one — shot 2's s-file value pairs with event row 2).
        scans, _ = _write_sfile(tmp_path, [[1, 5.0], [2, 6.5], [3, 7.0]])
        result = scan_frame(_detail(_event_frame(3)), scans)
        frame = result.frame
        row2 = frame[frame["scan_event_index"] == 2].iloc[0]
        assert row2["cam-fwhm"] == 20.0
        assert row2["U_ICT charge"] == 6.5
        assert result.provenance["cam-fwhm"] == PROVENANCE_RUN
        assert result.provenance["U_ICT charge"] == PROVENANCE_SFILE
        assert result.columns(PROVENANCE_SFILE) == ["Shotnumber", "U_ICT charge"]

    def test_outer_join_keeps_one_sided_shots(self, tmp_path):
        # s-file has a shot the event table lacks (and vice versa):
        # both survive, NaN on the absent side.
        scans, _ = _write_sfile(tmp_path, [[2, 6.5], [4, 9.0]])
        frame = scan_frame(_detail(_event_frame(3)), scans).frame
        assert len(frame) == 4  # shots 1,2,3 (events) ∪ 2,4 (s-file)
        shot4 = frame[frame["Shotnumber"] == 4].iloc[0]
        assert pd.isna(shot4["cam-fwhm"])

    def test_name_collision_suffixes_the_sfile_column(self, tmp_path):
        scans, _ = _write_sfile(
            tmp_path, [[1, 1.0], [2, 2.0]], columns=("Shotnumber", "cam-fwhm")
        )
        result = scan_frame(_detail(_event_frame(2)), scans)
        assert result.provenance["cam-fwhm"] == PROVENANCE_RUN
        assert result.provenance["cam-fwhm (s-file)"] == PROVENANCE_SFILE
        assert result.frame["cam-fwhm"].tolist() == [10.0, 20.0]

    def test_run_only_when_sfile_absent(self, tmp_path):
        scans = tmp_path / "26_0829" / "scans" / "Scan002"
        scans.mkdir(parents=True)
        result = scan_frame(_detail(_event_frame(2)), scans)
        assert set(result.provenance.values()) == {PROVENANCE_RUN}
        assert len(result.frame) == 2

    def test_sfile_only_when_no_event_data(self, tmp_path):
        scans, _ = _write_sfile(tmp_path, [[1, 5.0]])
        result = scan_frame(_detail(None), scans)
        assert set(result.provenance.values()) == {PROVENANCE_SFILE}

    def test_both_absent_is_empty(self):
        result = scan_frame(None, None)
        assert result.frame.empty
        assert result.provenance == {}

    def test_parses_but_corrupt_shot_key_degrades_to_run_only(self, tmp_path):
        # An s-file that PARSES but has a torn key cell (stray text /
        # non-integral float) must degrade like binary garbage — never
        # raise out of scan_frame.
        for bad in ("EOS", 2.5):
            scans, _ = _write_sfile(tmp_path, [[1, 5.0], [bad, 6.0]])
            result = scan_frame(_detail(_event_frame(2)), scans)
            assert set(result.provenance.values()) == {PROVENANCE_RUN}
            assert len(result.frame) == 2

    def test_duplicate_shotnumber_keeps_first_one_row_per_shot(self, tmp_path):
        # Producers write unique 1..N; corruption/hand edits must not
        # silently multiply rows — keep-first (the shared join doctrine).
        scans, _ = _write_sfile(tmp_path, [[1, 5.0], [2, 6.5], [2, 9.9]])
        result = scan_frame(_detail(_event_frame(3)), scans)
        frame = result.frame
        assert len(frame) == 3  # one row per shot, not four
        assert frame[frame["Shotnumber"] == 2]["U_ICT charge"].tolist() == [6.5]

    def test_event_shotnumber_collision_still_joins_on_event_index(self, tmp_path):
        # An event table that itself carries "Shotnumber": the s-file's
        # key column is renamed, but the join must still align
        # scan_event_index with the s-file's ORIGINAL Shotnumber values.
        scans, _ = _write_sfile(tmp_path, [[1, 5.0], [2, 6.5]])
        event = _event_frame(2)
        event["Shotnumber"] = [101, 102]  # decoy event column
        result = scan_frame(_detail(event), scans)
        frame = result.frame
        assert result.provenance["Shotnumber"] == PROVENANCE_RUN
        assert result.provenance["Shotnumber (s-file)"] == PROVENANCE_SFILE
        row2 = frame[frame["scan_event_index"] == 2].iloc[0]
        assert row2["U_ICT charge"] == 6.5  # joined on the s-file key, not the decoy
        assert row2["Shotnumber"] == 102

    def test_corrupt_sfile_degrades_to_run_only(self, tmp_path):
        scans, sfile = _write_sfile(tmp_path, [[1, 5.0]])
        sfile.write_bytes(b"\x00\x01 not a tsv \x02")
        result = scan_frame(_detail(_event_frame(1)), scans)
        # a corrupt s-file must never sink the run side
        assert PROVENANCE_RUN in set(result.provenance.values())

    def test_read_only_tree_untouched(self, tmp_path):
        scans, _ = _write_sfile(tmp_path, [[1, 5.0]])
        before = sorted(str(p) for p in tmp_path.rglob("*"))
        scan_frame(_detail(_event_frame(1)), scans)
        assert sorted(str(p) for p in tmp_path.rglob("*")) == before


class TestScanDataDelegation:
    def test_load_scalars_uses_the_shared_reader(self, monkeypatch, tmp_path):
        # ScanData.load_scalars must go through data.sfile (one home) —
        # pinned by spying the shared reader.
        import geecs_data_utils.data.sfile as sfile_mod
        from geecs_data_utils.scan_data import ScanData

        calls = {}
        real = sfile_mod.read_sfile

        def spy(path):
            calls["path"] = path
            return real(path)

        # load_scalars imports the names function-locally from data.sfile,
        # so patching the module attribute intercepts the call.
        monkeypatch.setattr(sfile_mod, "read_sfile", spy)

        day = tmp_path / "Undulator" / "Y2026" / "08-Aug" / "26_0829"
        scans = day / "scans" / "Scan002"
        scans.mkdir(parents=True)
        (day / "analysis").mkdir()
        pd.DataFrame({"Shotnumber": [1], "v": [2.0]}).to_csv(
            day / "analysis" / "s2.txt", sep="\t", index=False
        )
        from geecs_data_utils.scan_paths import ScanPaths

        sd = ScanData(paths=ScanPaths(folder=scans))
        before = sorted(str(p) for p in tmp_path.rglob("*"))
        sd.load_scalars(append_paths=False)
        assert calls["path"] == day / "analysis" / "s2.txt"
        assert sd.data_frame is not None and len(sd.data_frame) == 1
        # The old path derived the s-file via get_analysis_folder(),
        # which CREATES analysis/ScanNNN — reading must no longer mkdir.
        after = sorted(str(p) for p in tmp_path.rglob("*"))
        assert after == before
