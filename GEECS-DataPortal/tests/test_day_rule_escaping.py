"""The day page's rule editor escapes every value a ``where=`` link carries."""

from __future__ import annotations

import re
from pathlib import Path

DAY = Path(__file__).parents[1] / "geecs_portal" / "templates" / "day.html"


def test_rule_values_never_reach_innerhtml_raw():
    # a crafted link's low/high once went straight into the row's innerHTML
    text = DAY.read_text(encoding="utf-8")
    raw = re.findall(r"\$\{rule\.(?:low|high)\b", text)
    assert raw == [], "rule.low / rule.high must go through esc()"
    assert "${esc(rule.low)}" in text and "${esc(rule.high)}" in text
