"""Compaction and restore of a converted ``.himg`` folder (io/himg_compact.py), the CLI, the worker."""

from __future__ import annotations

import json
import logging
import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from geecs_data_utils.himg_cli import main as himg_main
from geecs_data_utils.io import himg_compact as compact_module
from geecs_data_utils.io.himg import himg_bytes
from geecs_data_utils.io.himg_compact import (
    MANIFEST_NAME,
    HimgCompactReport,
    HimgFolderActive,
    HimgRestoreReport,
    HimgSourceChanged,
    HimgStackIncomplete,
    NoHimgStack,
    compact_himg_folder,
    manifest_path_for,
    read_manifest,
    restore_himg_folder,
)
from geecs_data_utils.data.sfile import run_closed_evidence, scan_data_txt_path_for
from geecs_data_utils.io.himg_stack import (
    HimgSourcesDeleted,
    HimgStackError,
    HimgStackExists,
    HimgVerificationFailed,
    NoHimgFiles,
    convert_himg_folder,
    list_himg_files,
    part_path_for,
    read_source_bytes,
    stack_header,
    stack_path_for,
    verify_himg_stack,
)
from geecs_data_utils.io.himg_worker import run_himg_job, run_job_in_process

DEVICE = "U_HasoLift"
STAMPS = [3873135602.613, 3873135603.611, 3873135604.615]
HEIGHT, WIDTH = 5, 7
OLD = time.time() - 3600  # an hour ago: past the age guard


def _himg(seed: int) -> bytes:
    blob = bytes([seed % 256]) * 20 + b"\x01\x02"
    header = b"\x00" + struct.pack("<4I", 2, WIDTH, HEIGHT, len(blob)) + blob
    rng = np.random.default_rng(seed)
    return himg_bytes(header, rng.integers(0, 16, (HEIGHT, WIDTH), dtype=np.uint16))


def build_scan(root: Path, *, closed: bool = True, convert: bool = True) -> Path:
    """``…/scans/Scan012/U_HasoLift`` with three native shots (old mtimes) and a stack."""
    scan = root / "Undulator" / "Y2026" / "03-Mar" / "26_0310" / "scans" / "Scan012"
    device_dir = scan / DEVICE
    device_dir.mkdir(parents=True)
    for seed, stamp in enumerate(STAMPS):
        path = device_dir / f"{DEVICE}_{stamp:.3f}.himg"
        path.write_bytes(_himg(seed))
        os.utime(path, (OLD, OLD))
        (device_dir / f"{DEVICE}_{stamp:.3f}.has").write_bytes(b"sidecar")
    (scan / "ScanInfoScan012.ini").write_text('[Scan Info]\nScan No = "12"\n')
    if closed:
        (scan / "ScanDataScan012.txt").write_text("Shotnumber\n1\n2\n3\n")
    (scan / "other_device").mkdir()
    (scan / "other_device" / "note.txt").write_text("untouched")
    if convert:
        convert_himg_folder(device_dir)
    return device_dir


def snapshot(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def sources(device_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in list_himg_files(device_dir)}


class TestCompact:
    def test_verifies_deletes_and_leaves_a_manifest(self, tmp_path):
        device_dir = build_scan(tmp_path)
        stack = stack_path_for(device_dir)
        stack_before = stack.read_bytes()
        originals = sources(device_dir)
        phases: list[tuple[int, int, str]] = []

        report = compact_himg_folder(device_dir, progress=lambda *p: phases.append(p))

        assert isinstance(report, HimgCompactReport)
        assert report.files_deleted == 3 and report.frames == 3
        assert report.bytes_freed == sum(len(b) for b in originals.values())
        assert report.stack_bytes == len(stack_before)
        assert report.already_compacted is False
        assert list_himg_files(device_dir) == []
        # The stack is not rewritten: its per-shot headers (the haso
        # service's sensor header) are exactly what the converter wrote.
        assert stack.read_bytes() == stack_before
        assert (
            stack_header(stack) == originals[min(originals)][:39]
        )  # 17 + 22-byte blob
        # The record says what went, and where it lives.
        manifest = read_manifest(device_dir)
        assert manifest["format"] == "geecs-himg-manifest"
        assert manifest["stack"] == stack.name and manifest["frames"] == 3
        assert [f["name"] for f in manifest["files"]] == sorted(originals)
        assert manifest["source_bytes"] == report.bytes_freed
        assert report.manifest_path == manifest_path_for(device_dir)
        # Files and GB before/after are in the one-line label.
        summary = report.summary()
        assert "3 .himg files verified against U_HasoLift.h5 and deleted" in summary
        assert "4 files /" in summary and "-> 1 file /" in summary
        assert "freed" in summary
        # Every frame was verified before the first deletion.
        verifying = [p for p in phases if p[2] == "verifying"]
        deleting = [p for p in phases if p[2] == "deleting"]
        assert verifying == [
            (1, 3, "verifying"),
            (2, 3, "verifying"),
            (3, 3, "verifying"),
        ]
        assert deleting == [(1, 3, "deleting"), (2, 3, "deleting"), (3, 3, "deleting")]
        assert phases.index(verifying[-1]) < phases.index(deleting[0])

    def test_touches_only_the_device_folder(self, tmp_path):
        device_dir = build_scan(tmp_path)
        before = snapshot(tmp_path)
        compact_himg_folder(device_dir)
        after = snapshot(tmp_path)
        outside_before = {
            p: b for p, b in before.items() if device_dir not in p.parents
        }
        outside_after = {p: b for p, b in after.items() if device_dir not in p.parents}
        assert outside_after == outside_before
        # Inside: the .himg files went, the sidecars and the stack stayed,
        # the manifest arrived; no directory anywhere.
        inside = sorted(p.name for p in device_dir.iterdir())
        assert inside == sorted(
            [f"{DEVICE}.h5", MANIFEST_NAME] + [f"{DEVICE}_{s:.3f}.has" for s in STAMPS]
        )
        assert all(not p.is_dir() for p in device_dir.iterdir())

    def test_a_young_file_means_the_scan_may_still_be_running(self, tmp_path):
        device_dir = build_scan(tmp_path)
        young = list_himg_files(device_dir)[-1]
        now = time.time()
        os.utime(young, (now, now))
        before = snapshot(device_dir)
        with pytest.raises(HimgFolderActive, match="may still be running"):
            compact_himg_folder(device_dir)
        assert snapshot(device_dir) == before
        # The guard is the age, not the clock: a shorter window admits it.
        compact_himg_folder(device_dir, min_age=0.0)
        assert list_himg_files(device_dir) == []

    def test_no_closed_run_evidence_refuses(self, tmp_path):
        device_dir = build_scan(tmp_path, closed=False)
        assert run_closed_evidence(device_dir.parent) is None
        before = snapshot(device_dir)
        with pytest.raises(HimgFolderActive, match="no evidence the scan closed"):
            compact_himg_folder(device_dir)
        assert snapshot(device_dir) == before
        # The analysis s-file is evidence too (the legacy layout).
        sfile = device_dir.parents[2] / "analysis" / "s12.txt"
        sfile.parent.mkdir()
        sfile.write_text("Shotnumber\n1\n")
        assert run_closed_evidence(device_dir.parent) == sfile
        compact_himg_folder(device_dir)
        assert list_himg_files(device_dir) == []

    def test_the_shell_escape_hatch_for_a_dead_scan(self, tmp_path):
        device_dir = build_scan(tmp_path, closed=False)
        compact_himg_folder(device_dir, require_closed=False)
        assert list_himg_files(device_dir) == []

    def test_a_file_the_stack_lacks_refuses(self, tmp_path):
        device_dir = build_scan(tmp_path)
        late = device_dir / f"{DEVICE}_3873135605.617.himg"
        late.write_bytes(_himg(9))
        os.utime(late, (OLD, OLD))
        before = snapshot(device_dir)
        with pytest.raises(HimgStackIncomplete, match="have no frame in U_HasoLift.h5"):
            compact_himg_folder(device_dir)
        assert snapshot(device_dir) == before

    def test_a_mismatch_deletes_nothing(self, tmp_path):
        device_dir = build_scan(tmp_path)
        victim = list_himg_files(device_dir)[1]
        data = bytearray(victim.read_bytes())
        data[-1] ^= 0xFF
        victim.write_bytes(bytes(data))
        os.utime(victim, (OLD, OLD))
        before = snapshot(device_dir)
        with pytest.raises(HimgSourceChanged, match="nothing was deleted") as info:
            compact_himg_folder(device_dir)
        assert info.value.changed == (victim.name,)
        assert "the stack is intact" in str(info.value)
        assert snapshot(device_dir) == before
        assert not manifest_path_for(device_dir).exists()

    def test_no_stack_or_a_foreign_stack_refuses(self, tmp_path):
        device_dir = build_scan(tmp_path, convert=False)
        with pytest.raises(NoHimgStack, match="has no stack"):
            compact_himg_folder(device_dir)
        assert len(list_himg_files(device_dir)) == 3
        import h5py

        with h5py.File(stack_path_for(device_dir), "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((3, HEIGHT, WIDTH)))
        with pytest.raises(NoHimgStack, match="not a .himg stack"):
            compact_himg_folder(device_dir)
        assert len(list_himg_files(device_dir)) == 3

    def test_a_part_file_means_another_writer_owns_the_folder(self, tmp_path):
        device_dir = build_scan(tmp_path)
        part_path_for(device_dir).write_bytes(b"")
        with pytest.raises(HimgStackError, match="conversion is in progress"):
            compact_himg_folder(device_dir)
        assert len(list_himg_files(device_dir)) == 3

    def test_already_compacted_is_a_report_not_an_error(self, tmp_path):
        device_dir = build_scan(tmp_path)
        compact_himg_folder(device_dir)
        again = compact_himg_folder(device_dir)
        assert again.already_compacted and again.files_deleted == 0
        assert "already compacted" in again.summary()
        # No files and no manifest: nothing to do, said as such.
        manifest_path_for(device_dir).unlink()
        with pytest.raises(NoHimgFiles, match="nothing to compact"):
            compact_himg_folder(device_dir)

    def test_an_interrupted_deletion_pass_is_finished_by_the_next_run(self, tmp_path):
        device_dir = build_scan(tmp_path)
        originals = sources(device_dir)
        # Simulate a run that wrote the manifest and died after one unlink.
        compact_himg_folder(device_dir)
        restore_himg_folder(device_dir)
        for path in list_himg_files(device_dir):
            os.utime(path, (OLD, OLD))
        stack = stack_path_for(device_dir)
        compact_module._write_manifest(
            device_dir, stack, *_provenance_of(stack), compacted=time.time()
        )
        list_himg_files(device_dir)[0].unlink()
        report = compact_himg_folder(device_dir)
        assert report.files_deleted == 2 and report.frames == 3
        assert list_himg_files(device_dir) == []
        assert restore_himg_folder(device_dir).files_restored == 3
        assert sources(device_dir) == originals


def _provenance_of(stack: Path):
    from geecs_data_utils.io.scan_stack import open_stack, read_stack_timestamps

    with open_stack(stack) as f:
        names, digests, sizes = compact_module._provenance(f)
    return names, digests, sizes, [float(s) for s in read_stack_timestamps(stack)]


class TestRestore:
    def test_rebuilds_every_file_byte_for_byte_and_drops_the_manifest(self, tmp_path):
        device_dir = build_scan(tmp_path)
        originals = sources(device_dir)
        compact_himg_folder(device_dir)
        phases: list[tuple[int, int, str]] = []

        report = restore_himg_folder(device_dir, progress=lambda *p: phases.append(p))

        assert isinstance(report, HimgRestoreReport)
        assert report.files_restored == 3 and report.files_present == 0
        assert report.bytes_written == sum(len(b) for b in originals.values())
        assert sources(device_dir) == originals
        assert not manifest_path_for(device_dir).exists()
        assert not any(p.suffix == ".part" for p in device_dir.iterdir())
        assert phases == [(i, 3, "restoring") for i in (1, 2, 3)]
        assert verify_himg_stack(stack_path_for(device_dir), against_files=True).ok
        summary = report.summary()
        assert "3 .himg files restored from U_HasoLift.h5, verified" in summary
        assert "-> 4 files /" in summary

    def test_compact_then_restore_round_trips_the_folder(self, tmp_path):
        device_dir = build_scan(tmp_path)
        before = snapshot(device_dir)
        compact_himg_folder(device_dir)
        restore_himg_folder(device_dir)
        assert snapshot(device_dir) == before

    def test_files_already_there_are_kept_when_identical(self, tmp_path):
        device_dir = build_scan(tmp_path)
        originals = sources(device_dir)
        report = restore_himg_folder(device_dir)  # nothing was compacted
        assert report.files_restored == 0 and report.files_present == 3
        assert "3 already present" in report.summary()
        assert sources(device_dir) == originals

    def test_a_differing_file_on_disk_stops_the_restore(self, tmp_path):
        device_dir = build_scan(tmp_path)
        compact_himg_folder(device_dir)
        name = f"{DEVICE}_{STAMPS[1]:.3f}.himg"
        (device_dir / name).write_bytes(b"someone else's file")
        with pytest.raises(HimgSourceChanged, match="nothing overwritten"):
            restore_himg_folder(device_dir)
        assert (device_dir / name).read_bytes() == b"someone else's file"
        assert manifest_path_for(device_dir).exists()  # still compacted

    def test_a_corrupt_frame_stops_the_restore(self, tmp_path, monkeypatch):
        device_dir = build_scan(tmp_path)
        compact_himg_folder(device_dir)
        real = compact_module._rebuild

        def corrupt(f, index):
            data = bytearray(real(f, index))
            if index == 2:
                data[-1] ^= 0xFF
            return bytes(data)

        monkeypatch.setattr(compact_module, "_rebuild", corrupt)
        with pytest.raises(HimgVerificationFailed, match="restore stopped"):
            restore_himg_folder(device_dir)
        assert len(list_himg_files(device_dir)) == 2  # the good ones, verified
        assert manifest_path_for(device_dir).exists()

    def test_restore_needs_a_stack(self, tmp_path):
        device_dir = build_scan(tmp_path, convert=False)
        with pytest.raises(NoHimgStack):
            restore_himg_folder(device_dir)


class TestReadSourceBytes:
    def test_reads_the_whole_file(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"\x00\x01" * 1000)
        assert read_source_bytes(path) == b"\x00\x01" * 1000


class TestCli:
    def test_compact_and_restore_a_scan_folder(self, tmp_path, capsys):
        device_dir = build_scan(tmp_path)
        originals = sources(device_dir)
        scan = device_dir.parent
        assert himg_main(["compact", str(scan)]) == 0
        out = capsys.readouterr().out
        assert "3 .himg files verified against U_HasoLift.h5 and deleted" in out
        assert list_himg_files(device_dir) == []
        assert himg_main(["restore", str(scan), "--device", DEVICE]) == 0
        assert "3 .himg files restored" in capsys.readouterr().out
        assert sources(device_dir) == originals

    def test_compact_refusals_are_failures(self, tmp_path, capsys):
        device_dir = build_scan(tmp_path, closed=False)
        assert himg_main(["compact", str(device_dir)]) == 1
        assert (
            "FAILED — U_HasoLift: no evidence the scan closed"
            in capsys.readouterr().out
        )
        assert len(list_himg_files(device_dir)) == 3
        assert himg_main(["compact", str(device_dir), "--assume-closed"]) == 0
        assert list_himg_files(device_dir) == []

    def test_nothing_converted_is_an_error(self, tmp_path, capsys):
        device_dir = build_scan(tmp_path, convert=False)
        assert himg_main(["compact", str(device_dir.parent)]) == 1
        assert "no converted device folders" in capsys.readouterr().err


class TestWorker:
    """The out-of-process runner: reports, errors, progress and logs come back intact."""

    def test_convert_compact_restore_through_a_child_process(self, tmp_path, caplog):
        device_dir = build_scan(tmp_path, convert=False)
        originals = sources(device_dir)
        events: list[tuple[int, int, str]] = []
        caplog.set_level(logging.INFO)

        report = run_himg_job(
            {"command": "convert", "device_dir": str(device_dir), "device": DEVICE},
            progress=lambda *e: events.append(e),
        )
        assert type(report).__name__ == "HimgStackReport"
        assert report.stack_path == stack_path_for(device_dir) and report.verified
        assert (3, 3, "writing") in events and (3, 3, "verifying") in events
        # The child's log records arrive under their own logger names.
        assert any(
            r.name == "geecs_data_utils.io.himg_stack" and "3 frames" in r.getMessage()
            for r in caplog.records
        )

        events.clear()
        compacted = run_himg_job(
            {"command": "compact", "device_dir": str(device_dir)},
            progress=lambda *e: events.append(e),
        )
        assert isinstance(compacted, HimgCompactReport)
        assert compacted.files_deleted == 3
        assert compacted.manifest_path == manifest_path_for(device_dir)
        assert events[-1] == (3, 3, "deleting")

        restored = run_himg_job({"command": "restore", "device_dir": str(device_dir)})
        assert isinstance(restored, HimgRestoreReport) and restored.files_restored == 3
        assert sources(device_dir) == originals

        checked = run_himg_job(
            {"command": "verify", "device_dir": str(device_dir), "against_files": True}
        )
        assert checked.ok and checked.frames == 3

    def test_the_jobs_errors_come_back_as_themselves(self, tmp_path):
        device_dir = build_scan(tmp_path)
        with pytest.raises(HimgStackExists) as info:
            run_himg_job({"command": "convert", "device_dir": str(device_dir)})
        assert info.value.stack_path == stack_path_for(device_dir)
        with pytest.raises(HimgFolderActive, match="may still be running"):
            run_himg_job(
                {"command": "compact", "device_dir": str(device_dir), "min_age": 1e9}
            )
        with pytest.raises(HimgStackError, match="unknown himg job"):
            run_himg_job({"command": "explode", "device_dir": str(device_dir)})

    def test_in_process_body_is_the_same_dispatch(self, tmp_path):
        device_dir = build_scan(tmp_path)
        report = run_job_in_process(
            {"command": "compact", "device_dir": str(device_dir)}
        )
        assert report.files_deleted == 3
        with pytest.raises(ValueError, match="unknown himg job"):
            run_job_in_process({"command": "nope", "device_dir": str(device_dir)})

    def test_a_child_that_dies_without_a_report_is_an_error(self, tmp_path):
        """A child that prints noise and exits: no report, so an error naming the tail."""
        import subprocess
        import sys
        from unittest import mock

        device_dir = build_scan(tmp_path)
        script = "import sys; print('boom, not json'); sys.exit(3)"
        real_popen = subprocess.Popen

        def fake_popen(command, **kwargs):
            return real_popen([sys.executable, "-c", script], **kwargs)

        with mock.patch.object(subprocess, "Popen", fake_popen):
            with pytest.raises(
                HimgStackError, match="exited with 3 and no report"
            ) as info:
                run_himg_job({"command": "restore", "device_dir": str(device_dir)})
        assert "boom, not json" in str(info.value)


def test_manifest_is_json_a_human_can_read(tmp_path):
    device_dir = build_scan(tmp_path)
    compact_himg_folder(device_dir)
    text = manifest_path_for(device_dir).read_text()
    record = json.loads(text)
    assert text.count("\n") > 10  # indented, one field per line
    assert record["compacted_iso"].endswith("+00:00")
    assert record["files"][0]["sha256"] and record["files"][0]["size"] > 0
    assert record["files"][0]["acq_timestamp"] == pytest.approx(
        STAMPS[0] - 2082844800.0
    )


class TestCompactedFolderIsNotReconverted:
    """The stack of a compacted folder is the only copy: overwrite refuses."""

    def test_convert_overwrite_refuses_and_keeps_every_frame(self, tmp_path):
        device_dir = build_scan(tmp_path)
        compact_himg_folder(device_dir)
        stack = stack_path_for(device_dir)
        before = stack.read_bytes()
        # A stray file lands after the compaction (a self-triggered frame).
        late = device_dir / f"{DEVICE}_3873135605.617.himg"
        late.write_bytes(_himg(9))
        os.utime(late, (OLD, OLD))
        with pytest.raises(HimgStackIncomplete, match="restore the folder first"):
            compact_himg_folder(device_dir)
        with pytest.raises(HimgSourcesDeleted, match="restore it") as info:
            convert_himg_folder(device_dir, overwrite=True)
        assert len(info.value.missing) == 3
        assert stack.read_bytes() == before  # the three frames are still there
        assert verify_himg_stack(stack).frames == 3
        # The way back: restore, then reconvert with overwrite, then compact.
        restore_himg_folder(device_dir)
        assert convert_himg_folder(device_dir, overwrite=True).frames == 4
        for path in list_himg_files(device_dir):  # restored files are young
            os.utime(path, (OLD, OLD))
        assert compact_himg_folder(device_dir).files_deleted == 4

    def test_a_damaged_stack_is_told_apart_from_a_changed_file(
        self, tmp_path, monkeypatch
    ):
        device_dir = build_scan(tmp_path)
        stack = stack_path_for(device_dir)
        real = compact_module.verify_himg_stack

        def damaged(*args, **kwargs):
            report = real(*args, **kwargs)
            return type(report)(
                stack_path=report.stack_path,
                frames=report.frames,
                mismatches=("U_HasoLift_3873135602.613.himg",),
            )

        monkeypatch.setattr(compact_module, "verify_himg_stack", damaged)
        with pytest.raises(HimgVerificationFailed, match="did not rebuild"):
            compact_himg_folder(device_dir)
        assert len(list_himg_files(device_dir)) == 3
        assert stack.read_bytes()  # untouched

    def test_a_rerun_keeps_the_first_compaction_stamp(self, tmp_path):
        device_dir = build_scan(tmp_path)
        compact_himg_folder(device_dir)
        first = read_manifest(device_dir)
        assert first["updated"] is None
        # An interrupted pass: the manifest is there, one file came back.
        restore_himg_folder(device_dir)
        stack = stack_path_for(device_dir)
        compact_module._write_manifest(
            device_dir, stack, *_provenance_of(stack), compacted=first["compacted"]
        )
        for path in list_himg_files(device_dir):
            os.utime(path, (OLD, OLD))
        time.sleep(0.01)
        compact_himg_folder(device_dir)
        again = read_manifest(device_dir)
        assert again["compacted"] == first["compacted"]
        assert again["updated"] is not None and again["updated"] > first["compacted"]


class TestRunClosedEvidence:
    def test_the_scan_data_table_then_the_sfile(self, tmp_path):
        scan = (
            tmp_path
            / "Undulator"
            / "Y2026"
            / "03-Mar"
            / "26_0310"
            / "scans"
            / "Scan012"
        )
        scan.mkdir(parents=True)
        assert scan_data_txt_path_for(scan) == scan / "ScanDataScan012.txt"
        assert run_closed_evidence(scan) is None
        sfile = (
            tmp_path
            / "Undulator"
            / "Y2026"
            / "03-Mar"
            / "26_0310"
            / "analysis"
            / "s12.txt"
        )
        sfile.parent.mkdir()
        sfile.write_text("Shotnumber\n")
        assert run_closed_evidence(scan) == sfile
        (scan / "ScanDataScan012.txt").write_text("Shotnumber\n")
        assert run_closed_evidence(scan) == scan / "ScanDataScan012.txt"
        # A folder that is not a canonical scans/ScanNNN path has only the table.
        odd = tmp_path / "somewhere" / "Scan012"
        odd.mkdir(parents=True)
        assert run_closed_evidence(odd) is None
        (odd / "ScanDataScan012.txt").write_text("x")
        assert run_closed_evidence(odd) == odd / "ScanDataScan012.txt"


class TestVerifyCli:
    def test_verify_knows_a_compacted_folder(self, tmp_path, capsys):
        device_dir = build_scan(tmp_path)
        compact_himg_folder(device_dir)
        assert himg_main(["verify", str(device_dir)]) == 0
        assert "3 frames verified" in capsys.readouterr().out
        assert himg_main(["verify", str(device_dir.parent)]) == 0
        assert "3 frames verified" in capsys.readouterr().out
        # against the files: the deleted sources are reported as missing, honestly
        assert himg_main(["verify", str(device_dir), "--against-files"]) == 1
        assert "3 source file(s) missing" in capsys.readouterr().out
