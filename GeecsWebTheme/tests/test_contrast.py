"""Every palette clears WCAG AA for the text it is used on.

The first cut of these palettes failed this on every one of the six —
``--muted``, which carries the 11–12 px labels everywhere, sat between
3.2:1 and 4.4:1, and the default palette's accent missed too. Nobody
noticed by looking, which is the point: contrast is a measurement, not an
impression, so it is pinned here at generation and never again by eye.

Pairs checked are the ones the surfaces actually put together:

- ``--muted`` on ``--paper``, ``--surface`` and ``--surface-2`` (labels,
  captions, table headers, the picker)
- ``--accent`` on ``--paper`` (links, the brand, active states)
- ``--on-accent`` on ``--accent`` (primary buttons, the active day)
- ``--ink`` on ``--paper`` (body text; should be far above the bar)

``--trace-*`` are plot marks on the plot ground and are not text; exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CSS = Path(__file__).resolve().parents[1] / "geecs_web_theme/static/theme.css"

#: WCAG 2.x AA for normal text.
AA = 4.5


def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` colours."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _palettes() -> dict[str, dict[str, str]]:
    """Every themed block's tokens, keyed like ``laser-dark``."""
    css = _CSS.read_text()
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(
        r':root\[data-theme="(\w+)"\](\[data-mode="dark"\])?\s*\{([^}]*)\}', css
    ):
        key = f"{m.group(1)}-{'dark' if m.group(2) else 'light'}"
        out[key] = dict(re.findall(r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})", m.group(3)))
    assert len(out) == 6, f"expected six palettes, found {sorted(out)}"
    return out


_PAIRS = [
    ("--muted", "--paper"),
    ("--muted", "--surface"),
    ("--muted", "--surface-2"),
    ("--accent", "--paper"),
    ("--on-accent", "--accent"),
    ("--ink", "--paper"),
]


@pytest.mark.parametrize("palette", sorted(_palettes()))
@pytest.mark.parametrize("fg,bg", _PAIRS)
def test_text_pairs_clear_aa(palette: str, fg: str, bg: str) -> None:
    """A foreground/ground pair the surfaces use for text is >= 4.5:1."""
    tokens = _palettes()[palette]
    ratio = contrast(tokens[fg], tokens[bg])
    assert ratio >= AA, f"{palette}: {fg} on {bg} is {ratio:.2f}:1 (needs {AA})"


def test_body_text_is_comfortably_above_the_bar() -> None:
    """Ink on paper should not be merely passing; prose lives there."""
    for palette, tokens in _palettes().items():
        assert contrast(tokens["--ink"], tokens["--paper"]) >= 10, palette
