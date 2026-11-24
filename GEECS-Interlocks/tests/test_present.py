"""The words both readers use: the verdict, the values, the kit's state words."""

import pytest

from geecs_interlocks.monitor import STATES, InterlockStatus, Snapshot
from geecs_interlocks.present import KIT_STATE, is_age, number_text, verdict_text


def snap(*states):
    return Snapshot(
        tuple(
            InterlockStatus(f"i{n}", "", "", s, since=0) for n, s in enumerate(states)
        )
    )


@pytest.mark.parametrize(
    "states,headline",
    [
        ((), "No interlocks"),
        (("off",), "Nothing armed"),
        (("ok", "off"), "Safe"),
        (("ok", "tripped"), "Tripped"),
        (("waiting", "settling"), "Not ready"),
    ],
)
def test_verdict_headline(states, headline):
    assert verdict_text(snap(*states))[0] == headline


def test_every_state_has_a_kit_word_the_kit_knows():
    from geecs_web_theme import STATES as KIT_WORDS

    assert set(KIT_STATE) == set(STATES)
    assert set(KIT_STATE.values()) <= set(KIT_WORDS)


def test_number_text():
    assert number_text(None) == "—"
    assert number_text(1.23456e-5) == "1.235e-05"
    assert number_text(2.25, seconds=True) == "2.2 s"
    assert number_text(42.4, seconds=True) == "42 s"


def test_is_age_reads_the_kind():
    assert is_age("stale") and is_age("external")
    assert not is_age("threshold") and not is_age("config_change")


def test_an_interlock_value_is_an_age_only_for_one_age_check():
    from geecs_interlocks.monitor import CheckStatus
    from geecs_interlocks.present import status_is_age

    def status(*kinds):
        checks = tuple(CheckStatus("c", False, kind=k) for k in kinds)
        return InterlockStatus("i", "", "", "ok", checks=checks)

    assert status_is_age(status("stale"))
    assert not status_is_age(status("threshold"))
    assert not status_is_age(status("stale", "stale"))  # a group shows its count


def test_a_wall_clock_step_neither_hides_nor_fakes_a_stall():
    """Staleness is judged on the monotonic stamp, not the wall clock."""
    import time

    from geecs_interlocks.present import stall_age

    fresh = Snapshot(taken=0.0, mono=time.monotonic())  # wall clock far in the past
    assert stall_age(fresh) is None
    stalled = Snapshot(
        taken=time.time() + 3600, mono=time.monotonic() - 5
    )  # wall clock ahead
    assert stall_age(stalled) > 4
