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


def test_is_age():
    assert is_age("stage on UDP 5050") and is_age("G connected")
    assert not is_age("G P > 10")
