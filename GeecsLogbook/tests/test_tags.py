"""Tags come out of the body, narrowly."""

from __future__ import annotations

import pytest

from geecs_logbook.tags import parse_tags


@pytest.mark.parametrize(
    "body,expected",
    [
        ("#laser tuned", ["laser"]),
        ("see #Laser and #laser again", ["laser"]),
        ("#e-beam #jet_z #x1", ["e-beam", "jet_z", "x1"]),
        ("# A heading, not a tag", []),
        ("issue #12 and #1", []),
        ("C# and a&#b and url/#frag and file.#ext", []),
        ("[see](#results) and ?#top", []),
        ("#e-beam- trails", []),
        ("#日本 is not ascii; #eé neither", []),
        ("`#include` and ```\n#define X\n``` but #real", ["real"]),
        ("word#glued is not a tag", []),
        ("é#glued is not a tag either", []),
        ("(#laser) and #jet.", ["laser", "jet"]),
        ("", []),
    ],
)
def test_parse_tags(body: str, expected: list[str]) -> None:
    """What counts, what does not."""
    assert parse_tags(body) == expected


def test_length_cap() -> None:
    """A tag is at most 32 characters; longer is not a tag."""
    assert parse_tags("#" + "a" * 32) == ["a" * 32]
    assert parse_tags("#" + "a" * 33) == []
