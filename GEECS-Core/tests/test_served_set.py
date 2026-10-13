"""The served-set rule, one case per drift the three former copies had."""

from __future__ import annotations

from geecs_core.db.served_set import served_variables


def row(name, *, settable=False, choices="numeric", variabletype=None, tag=""):
    return {
        "name": name,
        "settable": settable,
        "choices": choices,
        "variabletype": variabletype,
        "units": tag,
    }


def names(result, device):
    return [sv.name for sv in result.get(device, ())]


def test_get_list_plus_settables_in_row_order() -> None:
    var_map = {"D": [row("a"), row("b", settable=True), row("c"), row("d")]}
    out = served_variables(var_map, {"D": ["d", "a"]})
    assert names(out, "D") == ["a", "b", "d"]
    flags = {sv.name: (sv.monitored, sv.settable) for sv in out["D"]}
    assert flags == {"a": (True, False), "b": (False, True), "d": (True, False)}


def test_endpoint_filter_drops_devices_and_orders_output() -> None:
    var_map = {"A": [row("x")], "B": [row("x")], "NOEP": [row("x")]}
    sub = {"A": ["x"], "B": ["x"], "NOEP": ["x"]}
    out = served_variables(var_map, sub, devices=["B", "A"])
    assert list(out) == ["B", "A"]
    assert "NOEP" in served_variables(var_map, sub)  # no filter: var_map's devices


def test_non_scalar_variables_are_never_served() -> None:
    var_map = {
        "D": [
            row("image", choices="image"),
            row("trace", choices="1darray", settable=True),
            row("mode", variabletype="choice", choices="a,b"),
            row("path", choices="path", settable=True),
        ]
    }
    out = served_variables(var_map, {"D": ["image", "trace", "mode"]})
    assert names(out, "D") == ["mode", "path"]


def test_duplicate_rows_first_qualifying_row_wins() -> None:
    # The first row neither monitored nor settable must not shadow the settable one.
    var_map = {"D": [row("v", tag="first"), row("v", settable=True, tag="second")]}
    out = served_variables(var_map, {})
    assert [sv.row["units"] for sv in out["D"]] == ["second"]
    # Among qualifying rows, the first one wins.
    var_map = {"D": [row("v", tag="first"), row("v", settable=True, tag="second")]}
    out = served_variables(var_map, {"D": ["v"]})
    assert [(sv.row["units"], sv.settable) for sv in out["D"]] == [("first", False)]


def test_first_qualifying_row_decides_even_when_it_is_not_scalar() -> None:
    var_map = {"D": [row("v", choices="image"), row("v")]}
    assert served_variables(var_map, {"D": ["v"]}) == {}


def test_get_list_matches_case_insensitively_and_keeps_db_spelling() -> None:
    var_map = {"D": [row("MeanCounts"), row("Other")]}
    out = served_variables(var_map, {"D": ["meancounts"]})
    assert names(out, "D") == ["MeanCounts"]


def test_case_only_twins_one_get_list_entry_serves_its_own_spelling() -> None:
    # Two served twins normalise to one PV, which stops the CA gateway at startup.
    var_map = {"D": [row("Trigger"), row("trigger")]}
    assert names(served_variables(var_map, {"D": ["trigger"]}), "D") == ["trigger"]
    assert names(served_variables(var_map, {"D": ["TRIGGER"]}), "D") == ["Trigger"]


def test_case_only_twins_both_listed_are_both_kept_for_the_collision_check() -> None:
    var_map = {"D": [row("Trigger"), row("trigger")]}
    out = served_variables(var_map, {"D": ["Trigger", "trigger"]})
    assert names(out, "D") == ["Trigger", "trigger"]


def test_settables_mode() -> None:
    var_map = {"D": [row("a"), row("b", settable=True)]}
    assert names(
        served_variables(var_map, {"D": ["a"]}, include_settable=False), "D"
    ) == ["a"]
    # every-variable mode: sub_map None serves all, each counted as monitored
    out = served_variables(var_map, None)
    assert [(sv.name, sv.monitored) for sv in out["D"]] == [("a", True), ("b", True)]


def test_device_with_nothing_served_is_absent() -> None:
    var_map = {"IDLE": [row("a")], "IMG": [row("i", choices="image")]}
    out = served_variables(var_map, {"IMG": ["i"]}, devices=["IDLE", "IMG", "GONE"])
    assert out == {}
