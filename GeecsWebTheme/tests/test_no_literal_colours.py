"""Every web surface styles through the tokens, never with a literal colour.

This is the test that keeps "changing a theme is easy" true. A token layer
only works if components use it; the moment one rule hardcodes a colour,
that component silently stops responding to themes and nobody finds out
until they switch and one thing stays the wrong colour.

Three rules, each with a reason the first version of this file lacked:

- **A literal next to a token is still a literal.** The first version
  exempted any line containing ``--``, which is every line using
  ``var(--x)`` — so ``background:#ff00ff`` passed as long as the same rule
  also used a token. ``var(...)`` calls are stripped before matching, and
  only a token *definition* (``--name:`` at the start of a declaration)
  is exempt.
- **A missing surface fails, never skips.** Renaming ``run.html`` must not
  silently drop it from the guard.
- **Every token a surface references must be defined.** The first version
  compared theme blocks against each other and could not see that every
  consumer said ``--surface-2`` while the theme defined ``--surface2`` —
  which shipped every hover fill and button ground as transparent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_THEME_CSS = _REPO / "GeecsWebTheme/geecs_web_theme/static/theme.css"

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

#: Hex, rgb()/rgba(), hsl()/hsla(), and the named colours people actually
#: reach for as values. (The full CSS named-colour list would flag words
#: like "tan" in prose; these are the ones that show up in practice.)
_LITERAL = re.compile(
    r"#[0-9a-fA-F]{3,8}\b"
    r"|\brgba?\([^)]*\)"
    r"|\bhsla?\([^)]*\)"
    r"|(?<=:)\s*(?:white|black|red|green|blue|gray|grey)\b"
)

#: A token definition: the one place a literal is the whole point.
_DEFINITION = re.compile(r"^\s*--[\w-]+\s*:")

#: Shadows and scrims are opacity over whatever is behind them, so they
#: read on every ground and have no token to take. Restricted to black
#: and the ink-black the light shadow uses — a coloured glow is a colour
#: and must go through a token.
_GROUND_FREE = re.compile(
    r"rgba\(\s*(?:0\s*,\s*0\s*,\s*0|17\s*,\s*24\s*,\s*33)\s*,[^)]*\)"
)

#: Lines carrying a literal for a stated reason. The allowlist is where
#: this rule erodes if it erodes; every entry says why.
_ALLOWED = {
    # Each swatch must show ITS palette whichever one is active.
    ".sw-bella": "theme swatch",
    ".sw-laser": "theme swatch",
    ".sw-plasma": "theme swatch",
    # A deliberately neutral hairline that reads on any ground.
    "rgba(127,127,127": "neutral ring, ground-independent",
    # matplotlib renders onto white; the frame matches its own ground.
    "img.plot": "matplotlib raster frame",
    ".ce-preview img": "matplotlib raster frame",
}


def _split_style_blocks(text: str) -> list[tuple[str, bool]]:
    """Split HTML into (chunk, is_inside_a_style_element) pieces."""
    pieces: list[tuple[str, bool]] = []
    cursor = 0
    for m in re.finditer(r"<style[^>]*>(.*?)</style>", text, flags=re.S | re.I):
        pieces.append((text[cursor : m.start(1)], False))
        pieces.append((m.group(1), True))
        cursor = m.end(1)
    pieces.append((text[cursor:], False))
    return pieces


def _css_lines(path: Path) -> list[tuple[int, str]]:
    """Return the CSS lines of a file with their 1-based numbers.

    For an HTML file that means the contents of its ``<style>`` blocks and
    nothing else — ``#765`` in a JavaScript comment referencing a pull
    request is a valid three-digit hex to a regex, and a guardrail that
    cries wolf gets switched off. Comments are blanked for the same
    reason, character-for-character so line numbers still point somewhere.
    """
    text = path.read_text()
    if path.suffix in {".html", ".htm"}:
        text = "\n".join(
            chunk if inside else re.sub(r"\S", " ", chunk)
            for chunk, inside in _split_style_blocks(text)
        )
    text = re.sub(
        r"/\*.*?\*/", lambda m: re.sub(r"\S", " ", m.group()), text, flags=re.S
    )
    return [(n, ln) for n, ln in enumerate(text.splitlines(), 1) if ln.strip()]


def _offences(path: Path) -> list[str]:
    """Return the offending lines in one file."""
    found = []
    for number, line in _css_lines(path):
        if _DEFINITION.match(line):
            continue
        # A token USE is not a literal: strip it, then judge what is left.
        judged = re.sub(r"var\(\s*--[\w-]+\s*(?:,[^)]*)?\)", "var()", line)
        judged = _GROUND_FREE.sub("shadow()", judged)
        if not _LITERAL.search(judged):
            continue
        if any(marker in line for marker in _ALLOWED):
            continue
        found.append(f"{path.name}:{number}: {line.strip()[:100]}")
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
    assert path.is_file(), (
        f"{relative} is listed as a guarded surface but does not exist — "
        "if it moved, update _SURFACES rather than letting the guard lapse"
    )
    offences = _offences(path)
    assert not offences, (
        f"{len(offences)} literal colour(s) outside the token system:\n  "
        + "\n  ".join(offences)
    )


def test_a_literal_beside_a_token_is_still_caught(tmp_path: Path) -> None:
    """The hole the first version had: a token on the line excused a literal."""
    probe = tmp_path / "probe.css"
    probe.write_text("a{color:var(--accent);background:#ff00ff}\n")
    assert _offences(probe), "a literal next to var(--accent) slipped through"
    probe.write_text("a{color:var(--accent);box-shadow:0 0 0 2px #00ff00}\n")
    assert _offences(probe), "a coloured glow slipped through as a shadow"
    probe.write_text("a{box-shadow:0 1px 2px rgba(0,0,0,.4)}\n  --x: #123456;\n")
    assert not _offences(probe), "a black shadow or a definition was wrongly flagged"


def _blocks(css: str) -> dict[str, set[str]]:
    """Every ``:root…{}`` block in the theme, keyed by its selector."""
    out: dict[str, set[str]] = {}
    for m in re.finditer(r"(:root[^{]*)\{([^}]*)\}", css):
        out[m.group(1).strip()] = set(re.findall(r"(--[\w-]+)\s*:", m.group(2)))
    return out


def test_every_palette_defines_every_token() -> None:
    """No palette block is missing a token another defines.

    A missing token does not fail loudly — it inherits the bare ``:root``
    value, so one theme silently shows another's colour. Checks every
    block, the bare root included.
    """
    blocks = _blocks(_THEME_CSS.read_text())
    assert len(blocks) >= 7, f"expected root + 3×(light,dark); found {list(blocks)}"
    fonts = {"--ff-ui", "--ff-mono", "--ff-prose", "--r"}  # root-only by design
    expected = max(blocks.values(), key=len) - fonts
    for selector, tokens in sorted(blocks.items()):
        missing = expected - tokens
        assert not missing, f"{selector} is missing {sorted(missing)}"


def test_every_referenced_token_is_defined() -> None:
    """Every ``var(--x)`` in every surface names a token the theme defines.

    This is the test that would have caught ``--surface2`` vs
    ``--surface-2``: 34 references and zero definitions, invisible to a
    check that only compared theme blocks against each other.
    """
    defined: set[str] = set()
    for tokens in _blocks(_THEME_CSS.read_text()).values():
        defined |= tokens
    local_ok = {"--err"}  # a surface may alias a token for its own use
    problems = []
    for relative in _SURFACES:
        text = (_REPO / relative).read_text()
        for name in sorted(set(re.findall(r"var\(\s*(--[\w-]+)", text))):
            if name not in defined and name not in local_ok:
                problems.append(f"{Path(relative).name}: {name}")
    assert not problems, "referenced but never defined:\n  " + "\n  ".join(problems)


def test_python_and_boot_script_agree_on_the_theme_list() -> None:
    """``geecs_web_theme.THEMES`` and ``theme-boot.js`` name the same themes.

    The JS is the runtime authority (it stamps the page); the Python is
    what hosts read. Two copies by necessity — one runs in a browser — so
    this pins them together, and pins both to the CSS.
    """
    import sys

    sys.path.insert(0, str(_REPO / "GeecsWebTheme"))
    from geecs_web_theme import DEFAULT_THEME, THEMES  # noqa: E402

    boot = (_REPO / "GeecsWebTheme/geecs_web_theme/static/theme-boot.js").read_text()
    js_list = re.findall(
        r'"(\w+)"', re.search(r"themes:\s*\[([^\]]*)\]", boot).group(1)
    )
    js_default = re.search(r'defaultTheme:\s*"(\w+)"', boot).group(1)
    assert js_list == list(THEMES), (js_list, list(THEMES))
    assert js_default == DEFAULT_THEME, (js_default, DEFAULT_THEME)
    css = _THEME_CSS.read_text()
    for name in THEMES:
        assert f':root[data-theme="{name}"]' in css, f"{name} has no CSS block"
