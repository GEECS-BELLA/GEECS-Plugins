"""The scan-log readers: the ini's sections in one parse, and an s-file's first row."""

from __future__ import annotations

from pathlib import Path

from geecs_data_utils.data.sfile import (
    column_alias,
    header_for_variable,
    read_sfile_head,
    sfile_column,
    sfile_head_for_scan,
)
from geecs_data_utils.scan_info import (
    read_scan_info_file,
    read_scan_info_sections,
    scan_log_variables,
)

INI = """[Scan Info]
Scan No = 3
ScanStartInfo = "jet at 50% then 60%"
Scan Parameter = "U_S1H Current"

[Scan Log]
Variables = ["U_Hexapod:ypos", "U_ESP:Position.Axis 1", "U_X:a=b"]
"""


def test_a_percent_in_a_value_no_longer_blanks_the_file(tmp_path: Path) -> None:
    path = tmp_path / "ScanInfoScan003.ini"
    path.write_text(INI)
    info = read_scan_info_file(path)
    assert info["ScanStartInfo"] == "jet at 50% then 60%"
    assert info["Scan Parameter"] == "U_S1H Current"


def test_every_section_from_one_parse_and_the_scan_log_list(tmp_path: Path) -> None:
    path = tmp_path / "ScanInfoScan003.ini"
    path.write_text(INI)
    sections = read_scan_info_sections(path)
    assert set(sections) == {"Scan Info", "Scan Log"}
    assert scan_log_variables(sections) == [
        "U_Hexapod:ypos",
        "U_ESP:Position.Axis 1",
        "U_X:a=b",
    ]


def test_no_or_a_broken_scan_log_section_is_an_empty_list(tmp_path: Path) -> None:
    assert scan_log_variables({}) == []
    assert scan_log_variables({"Scan Log": {"Variables": "[not json"}}) == []
    assert scan_log_variables({"Scan Log": {"Variables": '{"a": 1}'}}) == []
    assert read_scan_info_sections(tmp_path / "missing.ini") == {}


def test_the_head_is_two_lines(tmp_path: Path) -> None:
    path = tmp_path / "s.txt"
    path.write_text("Bin #\tU_A x\n1\t2.5\n1\t9.9\n")
    assert read_sfile_head(path) == (["Bin #", "U_A x"], ["1", "2.5"])
    (tmp_path / "empty.txt").write_text("")
    assert read_sfile_head(tmp_path / "empty.txt") == ([], [])


def test_a_scan_head_prefers_the_analysis_sfile(tmp_path: Path) -> None:
    folder = tmp_path / "26_0911" / "scans" / "Scan004"
    folder.mkdir(parents=True)
    (folder / "ScanDataScan004.txt").write_text("a\n1\n")
    assert sfile_head_for_scan(folder) == (["a"], ["1"], True)
    (tmp_path / "26_0911" / "analysis").mkdir()
    (tmp_path / "26_0911" / "analysis" / "s4.txt").write_text("b\n2\n")
    assert sfile_head_for_scan(folder) == (["b"], ["2"], True)
    assert sfile_head_for_scan(tmp_path / "nowhere") == ([], [], False)


def test_columns_are_matched_bare_or_aliased() -> None:
    assert header_for_variable("U_ESP:Position.Axis 1") == "U_ESP Position.Axis 1"
    header = ["Shotnumber", "U_Hexapod ypos Alias:Hex Y", "U_S1H Current"]
    assert sfile_column("U_Hexapod ypos", header) == "U_Hexapod ypos Alias:Hex Y"
    assert sfile_column("U_S1H Current", header) == "U_S1H Current"
    assert sfile_column("U_S1H Curr", header) is None


def test_a_column_alias_is_the_text_after_the_separator() -> None:
    assert (
        column_alias("DDG-AA-ShotCntrl Delay.Ch C Alias:JetDelay (s)") == "JetDelay (s)"
    )
    assert column_alias("U_S1H Current") is None
    assert column_alias("U_S1H Current Alias:") is None  # an empty alias is none
