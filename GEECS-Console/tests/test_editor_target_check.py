"""editors/base.py target check (#772): pure functions, no Qt."""

from __future__ import annotations

from geecs_console.editors.base import near_miss, resolve_device, target_problem

LISTING = {
    "U_Grating2Rotation": ["Position.Axis 1", "Position.Axis 2", "Velocity"],
    "U_ESP_JetXYZ": ["Position.Axis 3"],
}


class TestResolveDevice:
    def test_exact_key_wins(self):
        assert resolve_device(LISTING, "U_ESP_JetXYZ") == "U_ESP_JetXYZ"

    def test_unique_case_insensitive_match(self):
        assert resolve_device(LISTING, "u_esp_jetxyz") == "U_ESP_JetXYZ"

    def test_unknown_is_none(self):
        assert resolve_device(LISTING, "U_Nope") is None

    def test_ambiguous_case_match_is_none(self):
        listing = {"U_Dev": ["a"], "u_dev": ["b"]}
        assert resolve_device(listing, "U_DEV") is None


class TestNearMiss:
    def test_the_live_case_position_axis1(self):
        assert near_miss("Position.Axis1", LISTING["U_Grating2Rotation"]) == (
            "Position.Axis 1"
        )

    def test_case_only_difference_hints_the_listed_spelling(self):
        assert near_miss("velocity", LISTING["U_Grating2Rotation"]) == "Velocity"

    def test_nothing_close_is_none(self):
        assert near_miss("Temperature", LISTING["U_Grating2Rotation"]) is None


class TestTargetProblem:
    def test_known_pair_is_fine(self):
        assert target_problem(LISTING, "U_Grating2Rotation", "Position.Axis 1") is None

    def test_device_only_check(self):
        assert target_problem(LISTING, "U_Grating2Rotation") is None
        assert "unknown device" in target_problem(LISTING, "U_Grating2")

    def test_unknown_variable_names_it_with_a_hint(self):
        problem = target_problem(LISTING, "U_Grating2Rotation", "Position.Axis1")
        assert "'Position.Axis1'" in problem
        assert "U_Grating2Rotation" in problem
        assert "did you mean 'Position.Axis 1'" in problem

    def test_unknown_variable_without_a_close_name_has_no_hint(self):
        problem = target_problem(LISTING, "U_Grating2Rotation", "Temperature")
        assert "'Temperature' is not a variable of 'U_Grating2Rotation'" == problem

    def test_unknown_device_hints_the_close_spelling(self):
        problem = target_problem(LISTING, "U_Grating2Rotatoin", "Position.Axis 1")
        assert problem.startswith("unknown device 'U_Grating2Rotatoin'")
        assert "did you mean 'U_Grating2Rotation'" in problem

    def test_device_match_is_case_insensitive(self):
        assert target_problem(LISTING, "u_grating2rotation", "Velocity") is None

    def test_empty_listing_means_unchecked(self):
        assert target_problem({}, "U_Anything", "Whatever") is None
