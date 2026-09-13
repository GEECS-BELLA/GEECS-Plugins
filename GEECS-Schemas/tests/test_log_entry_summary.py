"""The one-line stand-in an entry shows when it is collapsed.

Every entry in every logbook is collapsible, so the shut state has to say
something worth reading. These pin the shapes people actually write —
each case here is a real body from the logbook or its seed script.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from geecs_schemas.log_entry import LogEntry


def entry(body: str) -> LogEntry:
    """Build a minimal entry carrying ``body``."""
    now = datetime.now(timezone.utc)
    return LogEntry(
        entry_id="e1",
        day="2026-09-12",
        author="someone",
        kind="note",
        status="kept",
        body_md=body,
        created_at=now,
        updated_at=now,
        version=1,
    )


@pytest.mark.parametrize(
    "body,expected",
    [
        ("A plain sentence.", "A plain sentence."),
        # steerable: a heading is the writer choosing their own summary
        ("### Calibration\n\nran the wedge", "Calibration"),
        # a callout is a flavour marker; the chip already shows it
        (
            "> [!WARNING] Jet pressure drifting\n> checked the regulator",
            "Jet pressure drifting",
        ),
        ("- first bullet\n- second", "first bullet"),
        ("1. step one\n2. step two", "step one"),
        # inline markup: link text survives, decoration does not
        ("See the [run log](http://x) for **detail**.", "See the run log for detail."),
        ("Charge on `U_ICT` was low.", "Charge on U_ICT was low."),
        # underscores are emphasis in markdown AND in every device name here
        ("`UC_Amp3_IR_input` timed out", "UC_Amp3_IR_input timed out"),
        # a fenced block is skipped whole — "import os" is not a summary
        ("```\nimport os\n```\nafter the fence", "after the fence"),
        ("| a | b |\n|---|---|\nprose after", "prose after"),
        ("\n\n\nleading blank lines", "leading blank lines"),
        ("", ""),
        ("```\nimport os\n```", ""),
    ],
)
def test_summary_of_the_shapes_people_write(body: str, expected: str) -> None:
    """Each case is a body shape the logbook actually produces."""
    assert entry(body).summary == expected


def test_summary_is_derived_not_stored() -> None:
    """It is a property, so it costs no field and no migration.

    Every entry ever written already has one — which is the argument for
    deriving it rather than adding a title field that would be empty for
    all of them.
    """
    assert "summary" not in LogEntry.model_fields
    assert entry("something").model_dump().get("summary") is None
