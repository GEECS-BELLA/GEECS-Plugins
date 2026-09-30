"""The ``.himg`` → capture-stack converter (io/himg_stack.py) and its CLI."""

from __future__ import annotations

import hashlib
import struct

import h5py
import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.himg_cli import main as himg_main
from geecs_data_utils.io.himg import himg_bytes, parse_himg
from geecs_data_utils.io.himg_stack import (
    HEADER_DATASET,
    SOURCE_NAME_DATASET,
    SOURCE_SHA256_DATASET,
    SOURCE_SIZE_DATASET,
    HimgSource,
    HimgStackError,
    HimgStackExists,
    HimgStampsUnavailable,
    HimgVerificationFailed,
    NoHimgFiles,
    convert_himg_folder,
    himg_sources,
    stack_path_for,
    stamp_attribute_name,
    verify_himg_stack,
    stack_header,
    write_himg_stack,
)
from geecs_data_utils.io.scan_stack import (
    FRAMES_DATASET,
    LABVIEW_EPOCH_OFFSET,
    ShotRef,
    find_stack_file,
    frame_index_for_acq_timestamp,
    is_stack_file,
    open_stack,
    parse_attribute_name,
    read_shot,
    read_stack_attributes,
    read_stack_timestamps,
    stack_content_kind,
    stack_scalar_variables,
)
from geecs_data_utils.shot_files import map_shot_files

DEVICE = "U_HasoLift"
STAMPS = [3873135602.613, 3873135603.611, 3873135604.615]  # LabVIEW s, ~1 Hz
HEIGHT, WIDTH = 5, 7


def _pixels(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 16, (HEIGHT, WIDTH), dtype=np.uint16)


def _header(seed: int) -> bytes:
    blob = (
        bytes([seed % 256]) * 20 + b"\x01\x02"
    )  # a per-shot byte, like the real stamp
    return b"\x00" + struct.pack("<4I", 2, WIDTH, HEIGHT, len(blob)) + blob


def _write(path, seed: int) -> bytes:
    data = himg_bytes(_header(seed), _pixels(seed))
    path.write_bytes(data)
    return data


def native_folder(root, device=DEVICE, stamps=STAMPS):
    """A device folder of natively named files, written out of stamp order."""
    device_dir = root / device
    device_dir.mkdir(parents=True)
    files = {}
    for seed, stamp in reversed(list(enumerate(stamps))):
        path = device_dir / f"{device}_{stamp:.3f}.himg"
        files[path.name] = _write(path, seed)
    (device_dir / "Thumbs.db").write_bytes(b"not a shot")
    return device_dir, files


def legacy_folder(root, device=DEVICE, stamps=STAMPS, scan="Scan012"):
    """A device folder of legacy shot-numbered files plus the scan's rows."""
    device_dir = root / device
    device_dir.mkdir(parents=True)
    files = {}
    for seed, _ in enumerate(stamps):
        path = device_dir / f"{scan}_{device}_{seed + 1:03d}.himg"
        files[path.name] = _write(path, seed)
        (device_dir / f"{scan}_{device}_{seed + 1:03d}_raw.has").write_bytes(b"x")
    rows = pd.DataFrame(
        {
            "Shotnumber": range(1, len(stamps) + 1),
            "Bin #": [1] * len(stamps),
            f"{device} acq_timestamp": stamps,
        }
    )
    return device_dir, files, rows


class TestSources:
    def test_native_names_carry_their_stamp_and_sort_by_it(self, tmp_path):
        device_dir, _ = native_folder(tmp_path)
        sources = himg_sources(device_dir)
        assert [s.acq_timestamp for s in sources] == STAMPS
        assert [s.shot_number for s in sources] == [None] * 3
        assert [s.path.name for s in sources] == [
            f"{DEVICE}_{stamp:.3f}.himg" for stamp in STAMPS
        ]

    def test_legacy_names_take_the_rows_stamp(self, tmp_path):
        device_dir, _, rows = legacy_folder(tmp_path)
        sources = himg_sources(device_dir, rows=rows)
        assert [(s.shot_number, s.acq_timestamp) for s in sources] == list(
            zip([1, 2, 3], STAMPS)
        )

    @pytest.mark.parametrize(
        "spelling", [f"{DEVICE}:acq_timestamp", "u_hasolift-acq_timestamp"]
    )
    def test_the_stamp_column_is_found_by_the_shared_rule(self, tmp_path, spelling):
        device_dir, _, rows = legacy_folder(tmp_path)
        rows = rows.rename(columns={f"{DEVICE} acq_timestamp": spelling})
        assert [s.acq_timestamp for s in himg_sources(device_dir, rows=rows)] == STAMPS

    def test_legacy_names_without_stamps_are_refused(self, tmp_path):
        device_dir, _, rows = legacy_folder(tmp_path)
        with pytest.raises(HimgStampsUnavailable, match="need the scan's scalar rows"):
            himg_sources(device_dir)
        with pytest.raises(HimgStampsUnavailable, match="no U_HasoLift acq_timestamp"):
            himg_sources(device_dir, rows=rows[["Shotnumber"]])
        rows.loc[1, f"{DEVICE} acq_timestamp"] = float("nan")
        with pytest.raises(HimgStampsUnavailable, match="shot 2 has no finite"):
            himg_sources(device_dir, rows=rows)

    def test_a_foreign_name_is_refused(self, tmp_path):
        device_dir, _ = native_folder(tmp_path)
        (device_dir / "reference.himg").write_bytes(b"x")
        with pytest.raises(HimgStackError, match="reference.himg: neither"):
            himg_sources(device_dir)

    def test_missing_folder_lists_nothing(self, tmp_path):
        assert himg_sources(tmp_path / "absent") == []
        assert not (tmp_path / "absent").exists()


class TestWriteAndRead:
    def test_the_stack_is_the_layout_every_reader_prefers(self, tmp_path):
        device_dir, files = native_folder(tmp_path)
        report = convert_himg_folder(device_dir)
        stack = report.stack_path
        assert stack == stack_path_for(device_dir) == device_dir / f"{DEVICE}.h5"
        assert report.frames == 3 and report.verified is True
        assert report.source_bytes == sum(len(d) for d in files.values())
        assert report.stack_bytes == stack.stat().st_size > 0
        assert "3 frames" in report.summary() and "verified" in report.summary()

        assert find_stack_file(device_dir) == stack and is_stack_file(stack)
        assert stack_content_kind(stack) == "image"
        np.testing.assert_array_equal(
            read_stack_timestamps(stack, labview_epoch=True), STAMPS
        )
        np.testing.assert_array_equal(
            read_stack_timestamps(stack), np.array(STAMPS) - LABVIEW_EPOCH_OFFSET
        )
        for index, stamp in enumerate(STAMPS):
            assert frame_index_for_acq_timestamp(stack, stamp) == index
            np.testing.assert_array_equal(read_shot(stack, index), _pixels(index))
        attributes = read_stack_attributes(stack)
        assert set(attributes) == {stamp_attribute_name(DEVICE)}
        assert parse_attribute_name(stamp_attribute_name(DEVICE)) == (
            "u_hasolift",
            "himg",
            "frame_acq_timestamp",
        )
        assert stack_scalar_variables(stack) == {}
        with open_stack(stack) as f:
            assert bool(f.attrs["finalized"]) is True
            assert f.attrs["device"] == DEVICE and f.attrs["source_format"] == "himg"
            assert f[FRAMES_DATASET].dtype == np.uint16
            assert f[FRAMES_DATASET].chunks == (1, HEIGHT, WIDTH)
            assert f[FRAMES_DATASET].compression == "gzip"
            names = list(f[SOURCE_NAME_DATASET].asstr()[:])
            digests = list(f[SOURCE_SHA256_DATASET].asstr()[:])
            sizes = list(f[SOURCE_SIZE_DATASET][:])
            headers = f[HEADER_DATASET][:]
        assert names == [f"{DEVICE}_{stamp:.3f}.himg" for stamp in STAMPS]
        assert digests == [hashlib.sha256(files[n]).hexdigest() for n in names]
        assert sizes == [len(files[n]) for n in names]
        assert [bytes(h) for h in headers] == [_header(i) for i in range(3)]

    def test_every_frame_rebuilds_its_source_byte_for_byte(self, tmp_path):
        device_dir, files = native_folder(tmp_path)
        stack = convert_himg_folder(device_dir).stack_path
        with open_stack(stack) as f:
            for index, name in enumerate(f[SOURCE_NAME_DATASET].asstr()[:]):
                rebuilt = himg_bytes(
                    f[HEADER_DATASET][index].tobytes(), f[FRAMES_DATASET][index]
                )
                assert rebuilt == files[str(name)]
        check = verify_himg_stack(stack, against_files=True)
        assert check.ok and check.frames == 3 and "verified" in check.summary()

    def test_the_sources_are_untouched(self, tmp_path):
        device_dir, files = native_folder(tmp_path)
        before = {p.name: p.read_bytes() for p in device_dir.iterdir()}
        convert_himg_folder(device_dir)
        after = {
            p.name: p.read_bytes() for p in device_dir.iterdir() if p.suffix != ".h5"
        }
        assert after == before
        assert sorted(p.name for p in device_dir.iterdir()) == sorted(
            [*before, f"{DEVICE}.h5"]
        )

    def test_legacy_folder_converts_and_the_shot_mapper_prefers_the_stack(
        self, tmp_path
    ):
        device_dir, _, rows = legacy_folder(tmp_path)
        stack = convert_himg_folder(device_dir, rows=rows).stack_path
        mapped = map_shot_files(
            device_dir, rows, device=DEVICE, file_tail=".himg", prefer_stack=True
        )
        assert {shot: (str(ref), ref.shot_index) for shot, ref in mapped.items()} == {
            1: (str(stack), 0),
            2: (str(stack), 1),
            3: (str(stack), 2),
        }
        assert all(isinstance(ref, ShotRef) for ref in mapped.values())
        np.testing.assert_array_equal(read_shot(mapped[2]), _pixels(1))

    def test_a_device_name_other_than_the_folder(self, tmp_path):
        device_dir, _, rows = legacy_folder(tmp_path, device="U_HasoLift-Raw")
        rows = rows.rename(
            columns={"U_HasoLift-Raw acq_timestamp": "U_HasoLift acq_timestamp"}
        )
        stack = convert_himg_folder(
            device_dir, rows=rows, device="U_HasoLift"
        ).stack_path
        assert stack == device_dir / "U_HasoLift-Raw.h5"
        assert set(read_stack_attributes(stack)) == {stamp_attribute_name("U_HasoLift")}


class TestRefusals:
    def test_no_himg_files_is_its_own_error(self, tmp_path):
        empty = tmp_path / DEVICE
        empty.mkdir()
        (empty / "note.txt").write_text("nothing here")
        with pytest.raises(NoHimgFiles):
            convert_himg_folder(empty)
        assert sorted(p.name for p in empty.iterdir()) == ["note.txt"]

    def test_a_missing_folder_is_never_created(self, tmp_path):
        missing = tmp_path / "scans" / "Scan012" / DEVICE
        with pytest.raises(NoHimgFiles):
            convert_himg_folder(missing)
        with pytest.raises(HimgStackError, match="not an existing directory"):
            write_himg_stack(missing, [HimgSource(missing / "x.himg", 1.0)])
        assert not (tmp_path / "scans").exists()

    def test_an_existing_stack_is_refused_unless_overwritten(self, tmp_path):
        device_dir, _ = native_folder(tmp_path)
        first = convert_himg_folder(device_dir).stack_path
        created = first.stat().st_mtime_ns
        with pytest.raises(HimgStackExists) as excinfo:
            convert_himg_folder(device_dir)
        assert excinfo.value.stack_path == first
        assert first.stat().st_mtime_ns == created
        again = convert_himg_folder(device_dir, overwrite=True)
        assert again.stack_path == first and verify_himg_stack(first).ok

    def test_a_bad_source_leaves_nothing_behind(self, tmp_path):
        device_dir, _ = native_folder(tmp_path)
        (device_dir / f"{DEVICE}_3873135605.000.himg").write_bytes(b"\x00truncated")
        with pytest.raises(HimgStackError, match="3873135605.000.himg"):
            convert_himg_folder(device_dir)
        assert not any(p.suffix in {".h5", ".part"} for p in device_dir.iterdir())

    def test_frames_of_two_shapes_are_refused(self, tmp_path):
        device_dir, _ = native_folder(tmp_path)
        other = b"\x00" + struct.pack("<4I", 2, 3, 3, 0)
        (device_dir / f"{DEVICE}_3873135605.000.himg").write_bytes(
            himg_bytes(other, np.zeros((3, 3), np.uint16))
        )
        with pytest.raises(HimgStackError, match="differs from the first frame"):
            convert_himg_folder(device_dir)
        assert find_stack_file(device_dir) is None

    def test_a_part_file_means_another_conversion_owns_the_folder(self, tmp_path):
        """Two writers never race to the rename: the second refuses, by name."""
        device_dir, _ = native_folder(tmp_path)
        part = device_dir / f"{DEVICE}.h5.part"
        part.write_bytes(b"another writer, or one that died")
        with pytest.raises(HimgStackError, match="in progress"):
            convert_himg_folder(device_dir)
        assert part.read_bytes() == b"another writer, or one that died"
        assert find_stack_file(device_dir) is None
        # overwrite takes the folder over: the dead part is discarded.
        assert convert_himg_folder(device_dir, overwrite=True).verified is True
        assert not part.exists()

    def test_a_stack_that_fails_verification_is_removed(self, tmp_path, monkeypatch):
        import geecs_data_utils.io.himg_stack as module

        device_dir, _ = native_folder(tmp_path)
        real_verify = module.verify_himg_stack

        def corrupt_then_verify(stack_path, **kwargs):
            with h5py.File(stack_path, "a") as f:
                f[FRAMES_DATASET][1] = np.full((HEIGHT, WIDTH), 9, np.uint16)
            return real_verify(stack_path, **kwargs)

        monkeypatch.setattr(module, "verify_himg_stack", corrupt_then_verify)
        with pytest.raises(HimgVerificationFailed) as excinfo:
            convert_himg_folder(device_dir)
        assert excinfo.value.mismatches == (f"{DEVICE}_{STAMPS[1]:.3f}.himg",)
        assert find_stack_file(device_dir) is None


class TestVerify:
    def test_tampered_frame_and_missing_source_are_reported(self, tmp_path):
        device_dir, _ = native_folder(tmp_path)
        stack = convert_himg_folder(device_dir).stack_path
        with h5py.File(stack, "a") as f:
            f[FRAMES_DATASET][2] = np.zeros((HEIGHT, WIDTH), np.uint16)
        (device_dir / f"{DEVICE}_{STAMPS[0]:.3f}.himg").unlink()
        check = verify_himg_stack(stack, against_files=True)
        assert not check.ok
        assert check.mismatches == (f"{DEVICE}_{STAMPS[2]:.3f}.himg",)
        assert check.missing == (f"{DEVICE}_{STAMPS[0]:.3f}.himg",)
        assert "1 of 3 frames mismatched, 1 source" in check.summary()

    def test_a_foreign_stack_is_refused(self, tmp_path):
        stack = tmp_path / "UC_Cam.h5"
        with h5py.File(stack, "w") as f:
            f.create_dataset(FRAMES_DATASET, data=np.zeros((1, 2, 2), np.uint16))
        with pytest.raises(HimgStackError, match="not a .himg stack"):
            verify_himg_stack(stack)


class TestCli:
    @pytest.fixture
    def scan(self, tmp_path):
        scan_folder = tmp_path / "Undulator" / "Y2026" / "03-Mar" / "26_0310"
        scan_folder = scan_folder / "scans" / "Scan012"
        device_dir, files, rows = legacy_folder(scan_folder)
        rows.to_csv(scan_folder / "ScanDataScan012.txt", sep="\t", index=False)
        other, _ = native_folder(scan_folder, device="U_HasoLift2")
        (scan_folder / "UC_Cam").mkdir()
        (scan_folder / "UC_Cam" / "UC_Cam_3873135602.613.png").write_bytes(b"png")
        return scan_folder

    def test_convert_walks_every_himg_device_of_a_scan(self, scan, capsys):
        assert himg_main(["convert", str(scan)]) == 0
        out = capsys.readouterr().out
        assert "U_HasoLift: U_HasoLift.h5: 3 frames" in out
        assert "U_HasoLift2: U_HasoLift2.h5: 3 frames" in out
        assert "UC_Cam" not in out
        assert verify_himg_stack(scan / DEVICE / f"{DEVICE}.h5", against_files=True).ok
        assert verify_himg_stack(scan / "U_HasoLift2" / "U_HasoLift2.h5").ok
        assert not (scan / "UC_Cam" / "UC_Cam.h5").exists()

    def test_convert_one_device_then_refuse_then_overwrite(self, scan, capsys):
        device_dir = scan / DEVICE
        assert himg_main(["convert", str(device_dir)]) == 0
        assert himg_main(["convert", str(scan), "--device", DEVICE]) == 1
        assert "already exists" in capsys.readouterr().out
        assert not (scan / "U_HasoLift2" / "U_HasoLift2.h5").exists()
        assert himg_main(["convert", str(scan), "--device", DEVICE, "--overwrite"]) == 0
        assert himg_main(["convert", str(scan), "--device", "Nope"]) == 1

    def test_nothing_to_do_is_an_error_not_a_silent_success(self, tmp_path, capsys):
        (tmp_path / "UC_Cam").mkdir()
        assert himg_main(["convert", str(tmp_path)]) == 1
        assert himg_main(["verify", str(tmp_path)]) == 1
        assert capsys.readouterr().err.count("no .himg device folders") == 2

    def test_verify_reports_each_stack(self, scan, capsys):
        assert himg_main(["verify", str(scan)]) == 1  # no stacks yet
        assert "no stack" in capsys.readouterr().out
        himg_main(["convert", str(scan)])
        assert himg_main(["verify", str(scan), "--against-files"]) == 0
        assert himg_main(["verify", str(scan / DEVICE / f"{DEVICE}.h5")]) == 0
        (scan / DEVICE / "Scan012_U_HasoLift_002.himg").unlink()
        assert himg_main(["verify", str(scan / DEVICE), "--against-files"]) == 1
        assert "1 source file(s) missing" in capsys.readouterr().out


class TestStackHeader:
    def test_any_row_of_the_provenance_group_is_a_header(self, tmp_path):
        device_dir, files = native_folder(tmp_path)
        report = convert_himg_folder(device_dir)
        first = stack_header(report.stack_path)
        assert first == _header(0)
        assert stack_header(report.stack_path, 2) == _header(2)
        # Any header of the sensor rebuilds a readable file from any frame.
        header, pixels = parse_himg(files[f"{DEVICE}_{STAMPS[1]:.3f}.himg"])
        assert parse_himg(himg_bytes(first, pixels))[1].tolist() == pixels.tolist()
        with pytest.raises(HimgStackError, match="out of range"):
            stack_header(report.stack_path, 3)

    def test_a_plain_stack_has_no_header(self, tmp_path):
        stack = tmp_path / "Cam.h5"
        with h5py.File(stack, "w") as f:
            f.create_dataset(FRAMES_DATASET, data=np.zeros((1, 2, 2), dtype="u2"))
        with pytest.raises(HimgStackError, match="no /entry/instrument/himg/header"):
            stack_header(stack)
