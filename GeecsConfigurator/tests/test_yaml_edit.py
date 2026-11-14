"""A save changes only the text of what changed: the splice against golden files.

The fixture is shaped like the scan-variables file of the box trial
(2026-11-13), where one added variable re-dumped the whole file: folded
descriptions, quoted values, comments, blank lines, an indentless list.
"""

from __future__ import annotations

import copy
import difflib

import pytest
import yaml

from geecs_configurator.store import DocumentStore
from geecs_configurator.yaml_edit import Unpatchable, patch

GOLDEN = """\
schema_version: 1
# hand-written: keep the folded descriptions
variables:
  ALine_angle_x:
    kind: pseudo
    description: >-
      Horizontal angle bump at the undulator entrance with no position change
      there: S3H kicks by +θ, S4H by −2θ (the ×−2), which leaves zero
      displacement at the target.
    targets:
    - target: U_S3H:Current
      forward: composite_var * 1
    - target: U_S4H:Current  # the second kick
      forward: composite_var * -2
    mode: relative

  VS3H:   # a corrector
    target: U_VS3H:Current
    kind: motor
  'Beamline Out x':
    target: "U_ESP302_01:Position.Axis 3"
    kind: setpoint
"""

NEW_VARIABLE = {"target": "U_ESP_JetXYZ:Positions.Axis1", "kind": "motor", "description": "test of the new configurator"}  # fmt: skip


def _doc(text: str) -> dict:
    return yaml.safe_load(text)


def _changed(before: str, after: str) -> list[str]:
    """The changed lines, each with its +/- (line endings kept, so a mixed ending shows)."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    return [line for line in difflib.ndiff(a, b) if line[:1] in "+-"]


@pytest.fixture(params=["\n", "\r\n"], ids=["lf", "crlf"])
def golden(request: pytest.FixtureRequest) -> str:
    return GOLDEN.replace("\n", request.param)


def test_the_golden_file_is_a_valid_document() -> None:
    assert DocumentStore.validate("scan_variables", _doc(GOLDEN)).ok


def test_an_unchanged_draft_is_byte_identical(golden: str) -> None:
    assert patch(golden, _doc(golden)) == golden


def test_adding_one_variable_changes_only_the_added_lines(golden: str) -> None:
    nl = "\r\n" if "\r\n" in golden else "\n"
    draft = _doc(golden)
    draft["variables"]["temp_test"] = NEW_VARIABLE
    out = patch(golden, draft)
    assert out == golden + nl.join(["  temp_test:", "    target: U_ESP_JetXYZ:Positions.Axis1", "    kind: motor", "    description: test of the new configurator", ""])  # fmt: skip
    assert _doc(out) == draft


def test_editing_one_value_changes_only_that_line(golden: str) -> None:
    nl = "\r\n" if "\r\n" in golden else "\n"
    draft = _doc(golden)
    draft["variables"]["VS3H"]["target"] = "U_VS3H:Voltage"
    out = patch(golden, draft)
    assert _changed(golden, out) == [f"-     target: U_VS3H:Current{nl}", f"+     target: U_VS3H:Voltage{nl}"]  # fmt: skip
    assert _doc(out) == draft


def test_a_quoted_value_edited_inside_a_list_item_changes_one_line(golden: str) -> None:
    draft = _doc(golden)
    draft["variables"]["ALine_angle_x"]["targets"][1]["forward"] = "composite_var * -3"
    out = patch(golden, draft)
    assert [line.rstrip() for line in _changed(golden, out)] == ["-       forward: composite_var * -2", "+       forward: composite_var * -3"]  # fmt: skip


def test_removing_a_variable_removes_only_its_lines(golden: str) -> None:
    draft = _doc(golden)
    del draft["variables"]["VS3H"]
    out = patch(golden, draft)
    assert [line.rstrip() for line in _changed(golden, out)] == ["-   VS3H:   # a corrector", "-     target: U_VS3H:Current", "-     kind: motor"]  # fmt: skip
    assert _doc(out) == draft


def test_a_new_multi_line_string_is_a_literal_block_in_the_file_s_line_ending(golden: str) -> None:  # fmt: skip
    nl = "\r\n" if "\r\n" in golden else "\n"
    draft = _doc(golden)
    draft["variables"]["VS3H"]["description"] = "first line\nsecond line"
    out = patch(golden, draft)
    assert f"    description: |-{nl}      first line{nl}      second line{nl}" in out
    assert _doc(out) == draft
    if nl == "\r\n":
        assert "\n" not in out.replace("\r\n", "")  # never a mixed ending


def test_a_list_item_inserted_in_the_middle_moves_no_other(golden: str) -> None:
    draft = _doc(golden)
    targets = draft["variables"]["ALine_angle_x"]["targets"]
    targets.insert(1, {"target": "U_S3V:Current", "forward": "composite_var * 0"})
    out = patch(golden, draft)
    assert [line[:1] for line in _changed(golden, out)] == ["+", "+"]
    assert (
        "    - target: U_S4H:Current  # the second kick" in out
    )  # its comment stays with it
    assert _doc(out) == draft


def test_a_new_item_among_flow_items_is_written_in_flow() -> None:
    text = "states:\n  SCAN:\n    - {device: A, variable: V, value: '4.0'}  # on\n"
    draft = _doc(text)
    draft["states"]["SCAN"].append({"device": "B", "variable": "W", "value": "1"})
    out = patch(text, draft)
    assert out == text + "    - {device: B, variable: W, value: '1'}\n"


def test_a_value_that_would_read_as_another_type_is_quoted() -> None:
    text = "a: x  # keep\nb: 1\n"
    out = patch(text, {"a": "2026-11-10", "b": 1})
    assert out == "a: '2026-11-10'  # keep\nb: 1\n"


def test_untouched_dates_stay_unless_strict() -> None:
    text = "at: 2026-11-10\nn: 1\n"
    assert patch(text, {"at": "2026-11-10", "n": 2}) == "at: 2026-11-10\nn: 2\n"
    assert patch(text, {"at": "2026-11-10", "n": 2}, strict_dates=True) == "at: '2026-11-10'\nn: 2\n"  # fmt: skip


def test_a_bare_yaml_1_1_boolean_key_is_kept_as_written() -> None:
    text = "states:\n  OFF:\n    - {device: A, value: '0.5'}\n  SCAN: []\n"
    out = patch(text, {"states": {"OFF": [{"device": "A", "value": "0.5"}], "SCAN": [{"device": "B"}]}})  # fmt: skip
    assert out.startswith("states:\n  OFF:\n    - {device: A, value: '0.5'}\n")


def test_a_list_item_s_first_keys_removed_together_leave_the_next_on_the_dash_s_line() -> (
    None
):
    text = "l:\n- a: 1  # first\n  b: 2\n  c: 3\n- a: 4\n"
    assert patch(text, {"l": [{"c": 3}, {"a": 4}]}) == "l:\n- c: 3\n- a: 4\n"
    assert (
        patch(text, {"l": [{"b": 2, "c": 3}, {"a": 4}]})
        == "l:\n- b: 2\n  c: 3\n- a: 4\n"
    )


def test_an_unedited_lowercase_boolean_key_stays_where_and_as_it_is() -> None:
    text = "states:\n  off:\n  - {device: A, value: '0.5'}\n  SCAN:\n  - {device: B, value: '1'}\n"
    draft = {"states": {"OFF": [{"device": "A", "value": "0.5"}], "SCAN": [{"device": "B", "value": "2"}]}}  # fmt: skip
    assert patch(text, draft) == text.replace("value: '1'", "value: '2'")


def test_a_non_finite_float_is_written_as_itself() -> None:
    assert patch("v: 1.0\n", {"v": float("inf")}) == "v: .inf\n"


def test_a_change_inside_an_aliased_anchor_is_unpatchable() -> None:
    text = "a: &x [1, 2]\nb: *x\n"
    with pytest.raises(Unpatchable):
        patch(text, {"a": [1, 3], "b": [1, 2]})
    assert patch(text, copy.deepcopy(_doc(text))) == text
