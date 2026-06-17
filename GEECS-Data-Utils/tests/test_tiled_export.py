"""Tests for the Tiled→legacy-scalar-file exporter (pure transform layer)."""

from __future__ import annotations

import io

import pandas as pd

from geecs_data_utils.tiled_export import build_legacy_scalar_dataframe


def _primary_df() -> pd.DataFrame:
    """A synthetic primary stream: data + companion + row-identity columns."""
    return pd.DataFrame(
        {
            "bin_number": [1, 1, 2, 2],
            "shot_index_in_bin": [1, 2, 1, 2],
            "scan_event_index": [1, 2, 3, 4],
            "wavemeter-wavelength_nm": [800.1, 800.2, 800.3, 800.4],
            "jet_x-position": [4.0, 4.0, 5.0, 5.0],
            # companion columns that must be dropped:
            "wavemeter-acq_timestamp": [1.0, 2.0, 3.0, 4.0],
            "wavemeter-shot_id": [1, 2, 3, 4],
            "wavemeter-valid": [True, True, True, True],
        }
    )


def _start_doc() -> dict:
    return {
        "scan_number": 12,
        "scan_folder": "/data/Undulator/.../scans/Scan012",
        "geecs_scalar_headers": {
            # ordered: jet_x first to assert header-map order is preserved
            "jet_x-position": "U_ESP_JetXYZ Position.Axis 1",
            "wavemeter-wavelength_nm": "UC_Wavemeter Wavelength (nm)",
        },
    }


def test_columns_and_order() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    assert list(df.columns) == [
        "Bin #",
        "scan",
        "U_ESP_JetXYZ Position.Axis 1",
        "UC_Wavemeter Wavelength (nm)",
        "Shotnumber",
    ]


def test_companion_columns_dropped() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    for bad in ("wavemeter-acq_timestamp", "wavemeter-shot_id", "wavemeter-valid"):
        assert bad not in df.columns


def test_no_elapsed_time_column() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    assert "Elapsed Time" not in df.columns


def test_row_identity_values() -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    assert list(df["Bin #"]) == [1, 1, 2, 2]
    assert list(df["scan"]) == [12, 12, 12, 12]
    assert list(df["Shotnumber"]) == [1, 2, 3, 4]


def test_tsv_roundtrip(tmp_path) -> None:
    df = build_legacy_scalar_dataframe(_start_doc(), _primary_df())
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    reloaded = pd.read_csv(io.StringIO(buf.getvalue()), delimiter="\t")
    assert list(reloaded.columns) == list(df.columns)
    assert reloaded["UC_Wavemeter Wavelength (nm)"].iloc[0] == 800.1


def test_missing_bin_number_defaults_to_one() -> None:
    primary = _primary_df().drop(columns=["bin_number"])
    df = build_legacy_scalar_dataframe(_start_doc(), primary)
    assert list(df["Bin #"]) == [1, 1, 1, 1]
