"""The one-line stand-in an entry shows when it is collapsed.

Each case is a body shape people actually write here — including the two
that a regex-based first version got wrong.
"""

from __future__ import annotations

import pytest

from geecs_logbook.render import summarize


@pytest.mark.parametrize(
    "body,expected",
    [
        ("A plain sentence.", "A plain sentence."),
        # steerable: a heading is the writer choosing their own summary
        ("### Calibration\n\nran the wedge", "Calibration"),
        # a callout is a flavour marker; the chip already shows it
        (
            "> [!WARNING] Jet pressure drifting\n> checked it",
            "Jet pressure drifting checked it",
        ),
        # a wrapped paragraph is ONE block and reads as one line — the
        # line-based first version returned only "a line"
        ("a line\nand its continuation", "a line and its continuation"),
        ("- first bullet\n- second", "first bullet"),
        ("1. step one\n2. step two", "step one"),
        # emphasis that IS emphasis loses its markers; link text survives
        ("See the [run log](http://x) for **detail**.", "See the run log for detail."),
        ("Charge on `U_ICT` was low.", "Charge on U_ICT was low."),
        # LAB NOTATION, not markdown. A single `~` is not strikethrough and a
        # single `*` is not emphasis, but a regex that strips both turned
        # "~20 mJ, jitter ~3%" into "20 mJ, jitter 3%" — a different number
        # presented as the note's content, in the only line a reader sees
        # when the entry is shut.
        ("~20 mJ on target, jitter ~3%", "~20 mJ on target, jitter ~3%"),
        ("intensity 3*10^18 W/cm2", "intensity 3*10^18 W/cm2"),
        ("half the shots (~50) were clipped", "half the shots (~50) were clipped"),
        # a fenced block is skipped whole — "import os" is not a summary
        ("```\nimport os\n```\nafter the fence", "after the fence"),
        ("\n\n\nleading blank lines", "leading blank lines"),
        ("", ""),
        ("```\nimport os\n```", ""),
        # A table's first cell is an ordinary inline token, so it wins unless
        # cells are skipped — and "Parameter" or "Date" is content-free while
        # LOOKING like a summary. Both shapes come from the editor's own
        # headline features: the toolbar's table skeleton and a spreadsheet
        # paste. A test pinning this existed and was dropped when the
        # function moved packages; nothing noticed.
        (
            "| Parameter | Value |\n|---|---|\n| a | b |\n\nlooked fine",
            "looked fine",
        ),
        ("| Date | Shots |\n|---|---|\n| x | 1 |\n\nprose after", "prose after"),
        # A table followed IMMEDIATELY by prose — no blank line — is a shape
        # the editor produces on its own: insertBlock appends one newline, so
        # both the table button and the spreadsheet paste leave the cursor
        # there. markdown-it absorbs that sentence as another ROW, so the
        # prose is genuinely inside a cell and cannot be preferred. The first
        # cell is the fallback: thin, but true, and not empty.
        (
            "| Parameter | Value |\n|---|---|\n| p | 3 bar |\nreseated, looked fine",
            "Parameter",
        ),
        # a table with nothing else gets the same fallback
        ("| Parameter | Value |\n|---|---|\n| p | 3 bar |", "Parameter"),
        # a note that is one pasted screenshot — the commonest attachment
        # shape here — must not collapse to an author and a timestamp
        (
            "![Screenshot 2026-09-12 at 14.02](attachments/a/b.png)",
            "Screenshot 2026-09-12 at 14.02",
        ),
    ],
)
def test_summary_of_the_shapes_people_write(body: str, expected: str) -> None:
    """Each case is a body shape the logbook actually produces."""
    assert summarize(body) == expected


def test_a_long_first_line_is_truncated() -> None:
    """The summary shares a row with the author, stamp and chips."""
    got = summarize("x" * 400)
    assert len(got) <= 120
    assert got.endswith("…")


def test_it_lives_beside_the_parser_it_uses() -> None:
    """A body parse belongs where the body parser is.

    ``geecs_schemas.log_entry``'s docstring says nothing in the repository
    parses ``body_md`` for structure — only the renderer and the tag scan.
    A first version added a hand-rolled third parser there anyway, three
    hundred lines below that sentence, and got lab notation wrong because
    it was guessing at markdown with a regex instead of asking the parser
    that was already in the building.
    """
    from geecs_schemas.log_entry import LogEntry

    assert not hasattr(LogEntry, "summary")
