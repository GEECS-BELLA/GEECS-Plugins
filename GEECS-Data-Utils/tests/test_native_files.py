"""Contract tests for the native GEECS per-shot file naming convention.

``geecs_data_utils.native_files`` is the single source of truth for the
``{stem}_{acq_timestamp:.3f}{tail}`` convention consumed by GeecsBluesky
(producer), ScanAnalysis (reader) and GEECS-Scanner-GUI (waiter).  These
tests pin the exact rendering, the millisecond canonicalization (including
the ±1 ms rounding-boundary rationale), suffixed path construction, and the
legacy Master Control pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_data_utils import (
    filename_timestamp_regex,
    legacy_filename_regex,
    native_file_name,
    native_file_name_from_key,
    native_file_path,
    native_file_stem,
    render_timestamp,
    timestamp_key,
    timestamp_key_candidates,
)


class TestRenderTimestamp:
    """The one and only filename timestamp format: ``%.3f``."""

    @pytest.mark.parametrize(
        ("value", "rendered"),
        [
            (1234567890.1234, "1234567890.123"),  # truncating round
            (1000.5, "1000.500"),  # zero-padded to 3 decimals
            (55.5, "55.500"),
            (0.0, "0.000"),
            (12.3456, "12.346"),  # rounds up
            (3866137959.5239997, "3866137959.524"),  # boundary double rounds up
            (1749923456.7, "1749923456.700"),
        ],
    )
    def test_exact_rendering(self, value: float, rendered: str) -> None:
        assert render_timestamp(value) == rendered

    def test_matches_python_percent_format(self) -> None:
        # The contract is literally '%.3f'; pin equivalence explicitly.
        for value in (0.0005, 1.0, 3866137959.5239997, 987654321.9999):
            assert render_timestamp(value) == "%.3f" % value


class TestNativeFileName:
    def test_device_timestamp_tail_composition(self) -> None:
        assert (
            native_file_name("UC_TopView", 1234567890.1234, ".png")
            == "UC_TopView_1234567890.123.png"
        )

    def test_tail_is_verbatim(self) -> None:
        # Compound tails must round-trip untouched (no extension parsing).
        assert (
            native_file_name("U_PicoScope", 1000.5, ".tdms_index")
            == "U_PicoScope_1000.500.tdms_index"
        )

    def test_from_key_renders_key_over_1000(self) -> None:
        assert (
            native_file_name_from_key("UC_Cam", 1000500, ".png")
            == "UC_Cam_1000.500.png"
        )

    def test_from_key_agrees_with_render_of_exact_timestamp(self) -> None:
        ts = 3866137959.524
        key = timestamp_key(ts)
        assert native_file_name_from_key("D", key, ".png") == native_file_name(
            "D", ts, ".png"
        )


class TestTimestampKey:
    def test_canonicalizes_to_integer_milliseconds(self) -> None:
        assert timestamp_key(3866137959.524) == 3866137959524

    def test_text_roundtripped_row_value_still_joins_exactly(self) -> None:
        # An s-file text round-trip perturbs the double's last bits; the
        # canonicalized keys still match exactly in the common case.
        row_value = 3866137959.5239997  # text form of the device's …59.524
        assert timestamp_key(row_value) == timestamp_key(
            float(render_timestamp(row_value))
        )

    def test_row_double_and_filename_rendering_share_a_candidate(self) -> None:
        # The rationale for ±1 ms: at the %.3f rounding boundary the two
        # canonicalization routes disagree by exactly one integer.
        # round(ts * 1000) rounds the *scaled* double (banker's rounding on
        # …640.5 → …640) while the filename carries the correctly rounded
        # decimal rendering of ts itself ('…959.641' → key …641). The
        # candidate set must bridge exactly that gap — and no more.
        row_value = 3866137959.6405
        filename_ts = float(render_timestamp(row_value))  # what the file says
        row_key = timestamp_key(row_value)
        file_key = timestamp_key(filename_ts)
        assert row_key != file_key  # this is the boundary case
        assert abs(row_key - file_key) == 1
        assert file_key in timestamp_key_candidates(row_key)

    def test_candidates_exact_first_then_neighbours(self) -> None:
        # Probe order is part of the contract: exact match wins before the
        # boundary neighbours are consulted.
        assert timestamp_key_candidates(1000500) == (1000500, 1000499, 1000501)

    def test_zero(self) -> None:
        assert timestamp_key(0.0) == 0


class TestNativeFilePath:
    def test_plain_device(self) -> None:
        assert native_file_path("/s/Scan012", "UC_Cam", 1749923456.7, ".txt") == Path(
            "/s/Scan012/UC_Cam/UC_Cam_1749923456.700.txt"
        )

    def test_directory_suffix_names_folder_and_filename(self) -> None:
        # A suffixed diagnostic never writes into the bare device folder.
        assert native_file_path(
            Path("/scans/Scan012"),
            "U_FROG",
            12.3456,
            ".png",
            directory_suffix="-Temporal",
        ) == Path("/scans/Scan012/U_FROG-Temporal/U_FROG-Temporal_12.346.png")

    def test_presuffixed_stem_with_empty_suffix_is_equivalent(self) -> None:
        # Callers holding a pre-composed data_device_name pass it as the
        # device name with no suffix — same result either way.
        assert native_file_path(
            "/s", "U_FROG-Temporal", 7.25, ".png"
        ) == native_file_path(
            "/s", "U_FROG", 7.25, ".png", directory_suffix="-Temporal"
        )

    def test_stem_helper(self) -> None:
        assert native_file_stem("U_FROG", "-Temporal") == "U_FROG-Temporal"
        assert native_file_stem("UC_Cam") == "UC_Cam"


class TestFilenameTimestampRegex:
    def test_extracts_timestamp_group(self) -> None:
        m = filename_timestamp_regex(".png").search("UC_Cam_3866137959.524.png")
        assert m is not None
        assert m.group("ts") == "3866137959.524"

    def test_requires_exact_tail(self) -> None:
        pattern = filename_timestamp_regex(".png")
        assert pattern.search("UC_Cam_3866137959.524.tiff") is None
        assert pattern.search("UC_Cam_3866137959.524.png.bak") is None

    def test_tail_is_escaped_not_a_regex(self) -> None:
        # '.' in the tail must not act as a wildcard.
        assert filename_timestamp_regex(".png").search("UC_Cam_1.5Xpng") is None

    def test_rejects_names_without_decimal_timestamp(self) -> None:
        # Legacy shot-number names have no fractional part before the tail.
        assert filename_timestamp_regex(".png").search("Scan012_UC_Cam_005.png") is None

    def test_roundtrip_through_native_file_name(self) -> None:
        name = native_file_name("U_FROG-Temporal", 12.3456, ".png")
        m = filename_timestamp_regex(".png").search(name)
        assert m is not None
        assert timestamp_key(float(m.group("ts"))) == timestamp_key(
            float(render_timestamp(12.3456))
        )


class TestLegacyFilenameRegex:
    def test_matches_mc_convention(self) -> None:
        m = legacy_filename_regex(".png").match("Scan012_UC_ALineEBeam3_005.png")
        assert m is not None
        assert m.group("scan_number") == "012"
        assert m.group("device_subject") == "UC_ALineEBeam3"
        assert m.group("shot_number") == "005"

    def test_device_subject_with_underscores_is_nongreedy(self) -> None:
        m = legacy_filename_regex(".png").match("Scan999_UC_Amp2_IR_input_010.png")
        assert m is not None
        assert m.group("device_subject") == "UC_Amp2_IR_input"
        assert m.group("shot_number") == "010"

    @pytest.mark.parametrize(
        "name",
        [
            "Scan12_UC_Cam_005.png",  # scan number needs >= 3 digits
            "Scan012_UC_Cam_05.png",  # shot number needs >= 3 digits
            "Scan012_UC_Cam_005.tiff",  # wrong tail
            "UC_Cam_3866137959.524.png",  # native timestamp name
            "prefix_Scan012_UC_Cam_005.png",  # must match from the start
        ],
    )
    def test_rejections(self, name: str) -> None:
        assert legacy_filename_regex(".png").match(name) is None

    def test_tail_is_escaped(self) -> None:
        # A tail containing regex metacharacters is treated literally.
        pattern = legacy_filename_regex(".p+g")
        assert pattern.match("Scan012_UC_Cam_005.p+g") is not None
        assert pattern.match("Scan012_UC_Cam_005.ppg") is None
