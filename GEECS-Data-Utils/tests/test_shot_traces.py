"""A scan's trace devices row by row (io/shot_traces.py).

Native scope files (the NI-HWS waveform HDF5 the LeCroy VI saves as
``ScanNNN_<device>_SSS.dat``) are found, read one channel at a time, and
a bin's traces average only on a shared x axis. A row the device missed
gives no trace, never a neighbour's.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.io import shot_traces
from geecs_data_utils.io.array1d import Data1DResult

DEVICE = "SCOPE-AA-LeCroy"


def write_scope(path, channels, dt=1e-9, start=0.0):
    """An NI-HWS waveform file: per channel a time axis and scaled counts.

    Each channel's volts are ``counts * 0.5 + 1`` (its ``scale_coef``).
    """
    with h5py.File(path, "w") as f:
        axes = f.create_group("wfm_group0/axes")
        traces = f.create_group("wfm_group0/traces")
        for i, counts in enumerate(channels):
            x = axes.create_group(f"axis{2 * i}")
            x.attrs["increment"] = dt
            x.attrs["start"] = start
            y = axes.create_group(f"axis{2 * i + 1}")
            y.create_dataset("data_vector/data", data=np.asarray(counts, dtype=float))
            y.create_dataset("scale_coef", data=np.array([1.0, 0.5]))
            traces.create_group(f"trace{i}").attrs["name"] = np.bytes_(f"Channel {i}")
    return path


@pytest.fixture()
def scan_folder(tmp_path):
    """A canonical scan folder with three scope shots, two channels each."""
    folder = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1103" / "scans" / "Scan040"
    device = folder / DEVICE
    device.mkdir(parents=True)
    for shot in (1, 2, 3):
        write_scope(
            device / f"Scan040_{DEVICE}_{shot:03d}.dat",
            [np.full(5, shot), np.arange(5) * shot],
        )
    (folder / "CAM-X").mkdir()
    return folder


def test_a_scope_folder_is_a_trace_device(scan_folder):
    probe = shot_traces.trace_device(scan_folder, DEVICE)
    assert probe.kind == "native"
    assert probe.ext == "dat"
    assert probe.channels == ("Channel 0", "Channel 1")


def test_a_camera_or_unknown_name_is_not(scan_folder):
    assert shot_traces.trace_device(scan_folder, "CAM-X").kind == ""
    assert shot_traces.trace_device(scan_folder, "nope").kind == ""


def test_a_dat_file_that_is_not_a_waveform_is_not(tmp_path):
    folder = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1103" / "scans" / "Scan001"
    (folder / "DAQ").mkdir(parents=True)
    (folder / "DAQ" / "Scan001_DAQ_001.dat").write_bytes(b"not hdf5")
    assert shot_traces.trace_device(folder, "DAQ").kind == ""


def test_one_row_reads_its_own_file_and_channel(scan_folder):
    found = shot_traces.load_row_trace(scan_folder, None, DEVICE, 2, channel=1)
    assert found.kind == "native"
    np.testing.assert_allclose(found.result.data[:, 0], np.arange(5) * 1e-9)
    np.testing.assert_allclose(found.result.data[:, 1], np.arange(5) * 2 * 0.5 + 1)
    assert found.result.x_units == "s"


def test_a_channel_past_the_scope_is_refused(scan_folder):
    found = shot_traces.load_row_trace(scan_folder, None, DEVICE, 1, channel=5)
    assert found.result is None
    assert "no channel 5" in found.reason


def test_a_missed_shot_is_never_a_neighbours_trace(scan_folder):
    frame = pd.DataFrame(
        {f"{DEVICE} acq_timestamp": [1.0, np.nan, 3.0], "Shotnumber": [1, 2, 3]}
    )
    found = shot_traces.load_row_trace(scan_folder, frame, DEVICE, 2)
    assert found.result is None
    assert found.reason == "the device missed this shot"


def test_a_row_past_the_table_is_refused(scan_folder):
    frame = pd.DataFrame({"Shotnumber": [1, 2]})
    assert shot_traces.load_row_trace(scan_folder, frame, DEVICE, 3).result is None


def test_a_missing_file_is_refused(scan_folder):
    (scan_folder / DEVICE / f"Scan040_{DEVICE}_003.dat").unlink()
    found = shot_traces.load_row_trace(scan_folder, None, DEVICE, 3)
    assert found.result is None
    assert found.reason == "file not found"


def test_a_file_with_no_waveform_says_so(scan_folder):
    # The scope's stub when it was not armed: an HDF5 file with nothing in it.
    h5py.File(scan_folder / DEVICE / f"Scan040_{DEVICE}_002.dat", "w").close()
    found = shot_traces.load_row_trace(scan_folder, None, DEVICE, 2)
    assert found.result is None
    assert found.reason == "the file holds no trace"


def test_an_empty_first_file_does_not_hide_the_scope(scan_folder):
    # an unarmed shot's stub sorting first: the probe reads past it
    h5py.File(scan_folder / DEVICE / f"Scan040_{DEVICE}_001.dat", "w").close()
    found = shot_traces.trace_device(scan_folder, DEVICE)
    assert found.kind == "native"
    assert found.channels


def test_a_bin_averages_sample_by_sample(scan_folder):
    found = shot_traces.average_trace_rows(scan_folder, None, DEVICE, (1, 2, 3))
    np.testing.assert_allclose(found.result.data[:, 1], np.full(5, 2 * 0.5 + 1))
    np.testing.assert_allclose(found.result.data[:, 0], np.arange(5) * 1e-9)


def test_a_bin_leaves_out_a_shot_with_no_file(scan_folder):
    (scan_folder / DEVICE / f"Scan040_{DEVICE}_003.dat").unlink()
    found = shot_traces.average_trace_rows(scan_folder, None, DEVICE, (1, 2, 3))
    np.testing.assert_allclose(found.result.data[:, 1], np.full(5, 1.5 * 0.5 + 1))


def test_a_bin_whose_axes_differ_has_no_average(scan_folder):
    write_scope(
        scan_folder / DEVICE / f"Scan040_{DEVICE}_002.dat",
        [np.full(5, 2), np.arange(5)],
        dt=2e-9,
    )
    found = shot_traces.average_trace_rows(scan_folder, None, DEVICE, (1, 2))
    assert found.result is None
    assert "x axes differ" in found.reason


def _trace(x, y):
    return Data1DResult(data=np.column_stack((x, y)), x_units="s")


def test_average_traces_needs_the_same_length():
    assert (
        shot_traces.average_traces(
            [_trace([0, 1], [1, 1]), _trace([0, 1, 2], [1, 1, 1])]
        )
        is None
    )


def test_average_traces_skips_a_nan_sample():
    mean = shot_traces.average_traces(
        [_trace([0, 1], [1.0, np.nan]), _trace([0, 1], [3.0, 4.0])]
    )
    np.testing.assert_allclose(mean.data[:, 1], [2.0, 4.0])
    assert mean.x_units == "s"


def test_average_traces_tolerates_float_rounding_of_the_axis():
    x = np.arange(4) * 5e-11
    mean = shot_traces.average_traces(
        [_trace(x, [1] * 4), _trace(x * (1 + 1e-13), [3] * 4)]
    )
    np.testing.assert_allclose(mean.data[:, 1], [2] * 4)


def test_reading_leaves_the_tree_untouched(scan_folder):
    before = sorted(p.relative_to(scan_folder) for p in scan_folder.rglob("*"))
    shot_traces.average_trace_rows(scan_folder, None, DEVICE, (1, 2, 3, 9))
    shot_traces.load_row_trace(scan_folder, None, "nope", 1)
    assert sorted(p.relative_to(scan_folder) for p in scan_folder.rglob("*")) == before


def test_a_pooled_bin_averages_shots_from_two_scans(scan_folder, tmp_path):
    other = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1104" / "scans" / "Scan002"
    (other / DEVICE).mkdir(parents=True)
    write_scope(
        other / DEVICE / f"Scan002_{DEVICE}_001.dat", [np.full(5, 7.0), np.arange(5)]
    )
    parts = [
        shot_traces.ScanRows(scan_folder, None, (1,)),
        shot_traces.ScanRows(other, None, (1,)),
    ]
    found = shot_traces.average_trace_scan_rows(parts, DEVICE)
    # Shot 1 of each: counts 1 and 7, volts 1.5 and 4.5 → 3.0.
    np.testing.assert_allclose(found.result.data[:, 1], np.full(5, 3.0))


SPECTRUM = "SPEC-PL1-Thorlabs"


@pytest.fixture()
def spectra(tmp_path):
    """A scan folder of per-shot spectrometer text files (shot 2 missed).

    Each is what the Thorlabs VI writes: two tab-separated columns,
    wavelength and counts, no header, CRLF line ends.
    """
    folder = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1112" / "scans" / "Scan007"
    device = folder / SPECTRUM
    device.mkdir(parents=True)
    for shot in (1, 3):
        lines = [f"{500 + i * 0.5:.3f}\t{shot * i:.3f}" for i in range(4)]
        (device / f"Scan007_{SPECTRUM}_{shot:03d}.txt").write_bytes(
            ("\r\n".join(lines) + "\r\n").encode()
        )
    return folder


def test_a_folder_of_text_columns_is_a_trace_device(spectra):
    # Before, the folder probe ignored .txt and fell back to "png", so the
    # Images page asked for a picture and said "file not found".
    probe = shot_traces.trace_device(spectra, SPECTRUM)
    assert (probe.kind, probe.content, probe.ext, probe.channels) == (
        "native",
        "lineout",
        "txt",
        ("",),
    )


def test_a_text_shot_reads_wavelength_against_counts(spectra):
    found = shot_traces.load_row_trace(spectra, None, SPECTRUM, 3)
    assert found.kind == "native"
    np.testing.assert_allclose(found.result.data[:, 0], [500, 500.5, 501, 501.5])
    np.testing.assert_allclose(found.result.data[:, 1], [0, 3, 6, 9])


def test_a_text_shot_with_no_file_is_refused(spectra):
    assert shot_traces.load_row_trace(spectra, None, SPECTRUM, 2).kind == "missing"


def test_a_text_file_that_is_not_columns_is_not_a_trace(tmp_path):
    folder = tmp_path / "Expt" / "Y2026" / "11-Nov" / "26_1112" / "scans" / "Scan001"
    (folder / "NOTES").mkdir(parents=True)
    (folder / "NOTES" / "Scan001_NOTES_001.txt").write_text("aligned the beam\n")
    (folder / "ONECOL").mkdir()
    (folder / "ONECOL" / "Scan001_ONECOL_001.txt").write_text("1\n2\n3\n")
    assert shot_traces.trace_device(folder, "NOTES").kind == ""
    assert shot_traces.trace_device(folder, "ONECOL").kind == ""
