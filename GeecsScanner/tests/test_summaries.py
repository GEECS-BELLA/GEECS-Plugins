"""Reading stock plan items back into a line and a shot count."""

from __future__ import annotations

import pytest

from geecs_scanner.service.summaries import exit_word, summarize_item


def _item(name: str, args: list, **kwargs) -> dict:
    return {"name": name, "args": args, "kwargs": kwargs, "user": "u", "item_uid": "x"}


def test_scan_triplet_with_trailing_num() -> None:
    s = summarize_item(
        _item(
            "scan",
            [["UC_A", "U_ICT.scalars"], "U_S1H.current", 0, 2, 11],
            shots_per_step=10,
            acquisition="gated",
            trigger_profile="standard_1hz",
        )
    )
    assert s.text == "scan U_S1H.current 0 → 2 · 11 steps · 10 shots/step · gated"
    assert s.steps == 11 and s.shots_per_step == 10 and s.planned_shots == 110
    assert (
        s.detectors == ["UC_A", "U_ICT.scalars"] and s.trigger_profile == "standard_1hz"
    )


def test_grid_scan_multiplies_steps() -> None:
    s = summarize_item(
        _item(
            "grid_scan",
            [["UC_A"], "U_S1H.current", -1, 1, 5, "U_Hexapod.xpos", 0, 4, 3],
            shots_per_step=2,
            acquisition="strict",
        )
    )
    assert "U_S1H.current -1 → 1 × U_Hexapod.xpos 0 → 4" in s.text
    assert s.steps == 15 and s.planned_shots == 30


def test_list_scan_counts_points() -> None:
    s = summarize_item(
        _item(
            "list_scan", [["UC_A"], "U_S1H.current", [0.1, 0.2, 0.5]], shots_per_step=4
        )
    )
    assert "U_S1H.current [3 pts]" in s.text and s.planned_shots == 12


def test_count_reads_num_and_background() -> None:
    s = summarize_item(
        _item(
            "count",
            [["UC_A"]],
            num=50,
            acquisition="strict",
            md={"background": True, "description": "darks"},
        )
    )
    assert s.text == 'background · count · 50 shots · strict — "darks"'
    assert s.planned_shots == 50 and s.background is True


def test_preset_name_and_description_from_md() -> None:
    s = summarize_item(
        _item(
            "count",
            [["UC_A"]],
            num=3,
            md={"description": "align", "geecs": {"preset": "eb_align_1hz"}},
        )
    )
    assert s.preset == "eb_align_1hz" and s.description == "align"


def test_non_scan_items_and_unknown_shapes_never_raise() -> None:
    assert (
        summarize_item(_item("mv", ["U_S1H.current", 0.5])).text
        == "move U_S1H.current → 0.5"
    )
    assert (
        summarize_item(_item("run_action", ["close_shutters"])).text
        == "action close_shutters"
    )
    assert (
        summarize_item({"name": "spiral", "args": [["UC_A"], "a", "b"]}).text
        == "spiral"
    )
    assert summarize_item({}).text == "?"


@pytest.mark.parametrize(
    "status,expected",
    [
        ("completed", ("ok", "ok")),
        ("failed", ("failed", "failed")),
        ("stopped", ("stopped", "failed")),
        ("aborted", ("stopped", "failed")),
        ("halted", ("halted", "failed")),
        (None, ("unknown", "unknown")),
    ],
)
def test_exit_words(status, expected) -> None:
    assert exit_word(status) == expected


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"kind": "range", "start": 2, "stop": 5, "num": 7}, "sweep M 2 → 5 · 7 steps"),
        ({"kind": "range", "start": 9, "stop": 1, "num": 7}, "sweep M 9 → 1 · 7 steps"),
        ({"kind": "list", "positions": [2, 1, 2]}, "sweep M [3 pts] · 3 steps"),
        (
            {"kind": "log", "start_exp": -2, "stop_exp": 1, "num": 4, "relative": True},
            "sweep M (relative) 10^-2 → 10^1 · 4 steps",
        ),
    ],
)
def test_sweep_summary_retains_axis_range_or_list_count(spec, expected):
    summary = summarize_item(
        _item(
            "sweep",
            [[]],
            sweep={"trajectory": {"kind": "axes", "axes": [{"axis": "M", **spec}]}},
            shots_per_step=2,
        )
    )
    assert summary.text.startswith(expected)
    assert summary.planned_shots == summary.steps * 2
