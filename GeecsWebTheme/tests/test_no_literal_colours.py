"""Every web surface styles through the tokens, never with a literal colour.

This is the test that keeps "changing a theme is easy" true. A token layer
only works if components use it; the moment one rule hardcodes a colour,
that component silently stops responding to themes and nobody finds out
until they switch and one thing stays the wrong colour.

Rather than trusting the discipline, this walks every stylesheet and
template in the repository's web surfaces and fails on any colour literal
outside the places where one is genuinely correct.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Repository root, from this file.
_REPO = Path(__file__).resolve().parents[2]

#: The web surfaces bound by the rule. Adding a surface means adding it
#: here — a new page that skips the tokens should fail loudly, not quietly.
_SURFACES = [
    "GeecsWebTheme/geecs_web_theme/static/theme.css",
    "GEECS-DataPortal/geecs_portal/templates/base.html",
    "GEECS-DataPortal/geecs_portal/templates/day.html",
    "GEECS-DataPortal/geecs_portal/templates/run.html",
    "GeecsScanLog/geecs_scan_log/static/scanlog.css",
    "GeecsScanLog/geecs_scan_log/templates/day.html",
    "ScanAnalysis/scan_analysis/config_editor/static/editor.css",
    "ScanAnalysis/scan_analysis/config_editor/templates/editor.html",
]

#: Hex colours and rgb()/rgba() calls.
_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)")

#: Lines carrying a literal for a reason. Each entry is a substring that
#: must appear on the offending line, and each needs a stated reason —
#: the allowlist is where this rule erodes if it erodes.
_ALLOWED = {
    # The token definitions themselves: literals here are the point.
    "--": "a token definition",
    # Each swatch must show ITS palette whichever one is active.
    ".sw-bella": "theme swatch",
    ".sw-laser": "theme swatch",
    ".sw-plasma": "theme swatch",
    # A deliberately neutral hairline that reads on any ground.
    "rgba(127,127,127": "neutral ring, ground-independent",
    # SVG plot furniture drawn against a white raster from matplotlib.
    "img.plot": "matplotlib renders on white; the frame matches it",
    ".ce-preview img": "the analysis preview is a white raster too",
    # Shadows and scrims are opacity over whatever is behind them, so they
    # read correctly on every ground and have no token to take.
    "box-shadow": "shadow: opacity over any ground",
    "rgba(0,0,0,0.55)": "modal scrim: opacity over any ground",
}


def _css_lines(path: Path) -> list[tuple[int, str]]:
    """Return the CSS lines of a file, with their 1-based line numbers.

    For an HTML file that means the contents of its ``<style>`` blocks and
    nothing else. Scanning the whole file produces false positives that
    look exactly like real ones: ``#765`` in a JavaScript comment
    referencing a pull request is a valid three-digit hex colour to a
    regex, and a guardrail that cries wolf gets switched off.

    Comments are stripped for the same reason — a hex in a note about a
    colour is not a use of it.
    """
    text = path.read_text()
    if path.suffix in {".html", ".htm"}:
        text = "\n".join(
            # keep the line count honest so reported numbers point somewhere
            block if in_style else re.sub(r"\S", " ", block)
            for block, in_style in _split_style_blocks(text)
        )
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return [(n, ln) for n, ln in enumerate(text.splitlines(), 1) if ln.strip()]


def _split_style_blocks(text: str) -> list[tuple[str, bool]]:
    """Split HTML into (chunk, is_inside_a_style_element) pieces."""
    pieces: list[tuple[str, bool]] = []
    cursor = 0
    for match in re.finditer(r"<style[^>]*>(.*?)</style>", text, flags=re.S | re.I):
        pieces.append((text[cursor : match.start(1)], False))
        pieces.append((match.group(1), True))
        cursor = match.end(1)
    pieces.append((text[cursor:], False))
    return pieces


def _offences(path: Path) -> list[str]:
    """Return the offending lines in one file."""
    found = []
    for number, line in _css_lines(path):
        if not _LITERAL.search(line):
            continue
        if any(token in line for token in _ALLOWED):
            continue
        found.append(f"{path.name}:{number}: {line.strip()[:90]}")
    return found


@pytest.mark.parametrize("relative", _SURFACES)
def test_surface_uses_only_tokens(relative: str) -> None:
    """A web surface carries no unexplained colour literal.

    If this fails, the fix is almost always to replace the literal with a
    token from ``theme.css``. If the literal is genuinely correct — it must
    look the same under every palette — add it to ``_ALLOWED`` *with a
    reason*, and expect that reason to be read.
    """
    path = _REPO / relative
    if not path.is_file():
        pytest.skip(f"{relative} not present on this branch")
    offences = _offences(path)
    assert not offences, (
        f"{len(offences)} literal colour(s) outside the token system:\n  "
        + "\n  ".join(offences)
    )


def test_every_theme_defines_every_token() -> None:
    """No palette is missing a token another one defines.

    A missing token does not fail loudly — it inherits whatever the bare
    ``:root`` block set, so one theme silently shows another's colour. This
    catches that at the source.
    """
    css = (_REPO / _SURFACES[0]).read_text()
    blocks = re.findall(
        r':root\[data-theme="(\w+)"\](\[data-mode="dark"\])?\s*\{([^}]*)\}', css
    )
    assert blocks, "no theme blocks found — has the stylesheet moved?"

    named: dict[str, set[str]] = {}
    for theme, dark, body in blocks:
        key = f"{theme}-{'dark' if dark else 'light'}"
        named[key] = set(re.findall(r"(--[\w-]+)\s*:", body))

    expected = max(named.values(), key=len)
    for key, tokens in sorted(named.items()):
        missing = expected - tokens
        assert not missing, f"{key} is missing {sorted(missing)}"
