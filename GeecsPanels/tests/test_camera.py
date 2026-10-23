"""The camera contract and the freshness rule."""

from __future__ import annotations

from geecs_panels.camera import (
    COUNTER_PLUGINS,
    READINGS,
    Reading,
    counter_suffix,
    judge,
    monitored_suffixes,
    string_suffixes,
)

CENTROID = next(r for r in READINGS if r.key == "cx")
EXPOSURE = next(r for r in READINGS if r.key == "exposure")


def _judge(reading: Reading, *, age=0.1, flag=1, connected=True, stale_after=3.0):
    conn = {
        reading.suffix: connected,
        counter_suffix("STAT"): True,
        "STAT:ComputeCentroid": True,
    }
    return judge(
        reading,
        connected=conn,
        values={"STAT:ComputeCentroid": flag},
        counter_age={"STAT": age},
        stale_after=stale_after,
    )


def test_a_value_is_fresh_while_its_plugin_counts_frames() -> None:
    """A steady centroid stays fresh: the counter, not the value's monitor, decides."""
    assert _judge(CENTROID, age=0.1) == "fresh"


def test_a_value_goes_stale_when_its_plugin_stops() -> None:
    assert _judge(CENTROID, age=3.5) == "stale"


def test_a_counter_never_seen_to_advance_is_stale() -> None:
    """Opening the panel on a stopped camera shows stale, not fresh."""
    assert _judge(CENTROID, age=None) == "stale"


def test_a_value_its_plugin_is_not_computing_is_off() -> None:
    """ComputeCentroid off: the counter ticks but the centroid is frozen."""
    assert _judge(CENTROID, age=0.1, flag=0) == "off"


def test_a_disconnected_value_says_so() -> None:
    assert _judge(CENTROID, connected=False) == "disconnected"


def test_a_setting_is_fresh_while_connected() -> None:
    """Exposure posts on change only; there is no counter to judge it by."""
    assert (
        judge(
            EXPOSURE,
            connected={EXPOSURE.suffix: True},
            values={},
            counter_age={},
            stale_after=3.0,
        )
        == "fresh"
    )


def test_the_monitored_set_holds_every_reading_flag_and_counter_once() -> None:
    suffixes = monitored_suffixes()
    assert len(suffixes) == len(set(suffixes))
    assert {r.suffix for r in READINGS} <= set(suffixes)
    assert {r.computed_by for r in READINGS if r.computed_by} <= set(suffixes)
    assert {counter_suffix(p) for p in COUNTER_PLUGINS} <= set(suffixes)


def test_every_plugin_a_reading_names_has_its_counter_watched() -> None:
    assert {r.plugin for r in READINGS if r.plugin} <= set(COUNTER_PLUGINS)


def test_enums_are_read_as_strings() -> None:
    assert "cam1:DetectorState_RBV" in string_suffixes()
