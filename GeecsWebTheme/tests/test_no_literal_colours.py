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
    "GeecsLogbook/geecs_logbook/static/scanlog.css",
    "GeecsLogbook/geecs_logbook/templates/day.html",
    "GeecsLogbook/geecs_logbook/templates/month.html",
    "GeecsLogbook/geecs_logbook/templates/_entries.html",
    "GeecsLogbook/geecs_logbook/static/editor.js",
    "GeecsLogbook/geecs_logbook/static/nav.js",
    "ScanAnalysis/scan_analysis/config_editor/static/editor.css",
    "ScanAnalysis/scan_analysis/config_editor/templates/editor.html",
]

#: Hex, rgb()/rgba(), hsl()/hsla(), and the named colours people actually
#: reach for as values. (The full CSS named-colour list would flag words
#: like "tan" in prose; these are the ones that show up in practice.)
_NAMED = (
    "white|black|red|green|blue|gray|grey|orange|yellow|purple|pink|cyan|"
    "magenta|navy|teal|olive|maroon|silver|lime|aqua|fuchsia|tomato|"
    "lightgray|lightgrey|darkgray|darkgrey|whitesmoke|gold|crimson"
)
_LITERAL = re.compile(
    r"#[0-9a-fA-F]{3,8}\b"
    r"|%23[0-9a-fA-F]{3,8}\b"
    r"|\brgba?\([^)]*\)"
    r"|\bhsla?\([^)]*\)"
    r"|(?<=[:\s])(?:" + _NAMED + r")\b(?=\s*(?:[;}\"'!]|$))",
    re.IGNORECASE,
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


#: Outside ``<style>``, CSS still hides in ``style="…"`` attributes and in
#: JavaScript that assigns ``.style.cssText`` / ``.style.x``. Everything
#: else outside a style element (script logic, a ``#765`` PR reference in
#: a comment) is blanked.
_INLINE_STYLE = re.compile(
    r"""style\s*=\s*"([^"]*)"|style\s*=\s*'([^']*)'|style\.\w+\s*=\s*"([^"]*)"|style\.\w+\s*=\s*'([^']*)'|setAttribute\(\s*[\"']style[\"']\s*,\s*"([^"]*)"|setAttribute\(\s*[\"']style[\"']\s*,\s*'([^']*)'|\.setProperty\(\s*[\"'][\w-]+[\"']\s*,\s*"([^"]*)"|\.setProperty\(\s*[\"'][\w-]+[\"']\s*,\s*'([^']*)'|\b(?:fill|stroke)\s*=\s*"([^"]*)"|\b(?:fill|stroke)\s*=\s*'([^']*)'"""
)


def _css_lines(path: Path) -> list[tuple[int, str, bool]]:
    """Return the CSS lines of a file as ``(number, text, in_root_block)``.

    For an HTML file that means the contents of its ``<style>`` blocks plus
    the values of ``style="…"`` attributes and ``.style…="…"`` assignments
    — nothing else, because ``#765`` in a JavaScript comment referencing a
    pull request is a valid three-digit hex to a regex, and a guardrail
    that cries wolf gets switched off. Comments are blanked
    character-for-character so line numbers still point somewhere.

    ``in_root_block`` is whether the line sits inside a ``:root … {}``
    block — the only place a token *definition* is legitimately a literal.
    A ``--local: #ff00ff`` inside a component rule is a hidden literal.
    """
    text = path.read_text()
    if path.suffix in {".html", ".htm"}:
        pieces = []
        for chunk, inside in _split_style_blocks(text):
            if inside:
                pieces.append(chunk)
                continue
            kept = re.sub(r"\S", " ", chunk)
            for m in _INLINE_STYLE.finditer(chunk):
                g = next(
                    i for i in range(1, len(m.groups()) + 1) if m.group(i) is not None
                )
                kept = kept[: m.start(g)] + m.group(g) + kept[m.end(g) :]
            pieces.append(kept)
        text = "".join(pieces)
    text = re.sub(
        r"/\*.*?\*/", lambda m: re.sub(r"\S", " ", m.group()), text, flags=re.S
    )
    out = []
    depth = 0  # brace depth inside a :root block; 0 = outside
    for n, ln in enumerate(text.splitlines(), 1):
        # only :root itself (attribute selectors allowed), never a descendant
        opens_root = bool(
            re.match(
                r"\s*:root\s*(?:\[[^\]]*\])*\s*(?:,\s*:root\s*(?:\[[^\]]*\])*\s*)*\{",
                ln,
            )
        )
        in_root = opens_root or depth > 0
        if opens_root or depth > 0:
            depth = max(depth + ln.count("{") - ln.count("}"), 0)
        if ln.strip():
            out.append((n, ln, in_root))
    return out


def _offences(path: Path) -> list[str]:
    """Return the offending lines in one file."""
    found = []
    for number, line, in_root in _css_lines(path):
        judged = line
        if in_root and _DEFINITION.match(line):
            # A definition inside :root is the point; anything after it on
            # the same line is still judged.
            judged = re.sub(r"^\s*--[\w-]+\s*:[^;]*;?", "", line)
        # A token USE is not a literal. A var() fallback argument IS.
        judged = re.sub(r"var\(\s*--[\w-]+\s*\)", "var()", judged)
        judged = re.sub(r"var\(\s*--[\w-]+\s*,", "(", judged)
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


@pytest.mark.parametrize(
    "body,expected",
    [
        ("a{color:var(--accent);background:#ff00ff}\n", True),
        ("a{color:var(--accent);box-shadow:0 0 0 2px #00ff00}\n", True),
        (".foo {\n  --local: #ff00ff;\n  color: var(--local); }\n", True),
        (":root{\n  --x: #123456; color: #ff00ff;\n}\n", True),
        ("a{color:var(--accent,#ff00ff)}\n", True),
        ("a{color:White}\n", True),
        ("a{color:orange}\n", True),
        (
            "a{background:url(\"data:image/svg+xml,%3Csvg fill='%23ff00ff'/%3E\")}\n",
            True,
        ),
        (":root .foo {\n  --x: #ff00ff;\n}\n", True),
        ("a{box-shadow:0 1px 2px rgba(0,0,0,.4)}\n", False),
        (":root{\n  --x: #123456;\n}\n", False),
        (':root[data-theme="laser"][data-mode="dark"]{\n  --x: #123456;\n}\n', False),
        ("a{color:var(--accent)}\n", False),
    ],
)
def test_probe_literals(tmp_path: Path, body: str, expected: bool) -> None:
    """Each hole a review found, pinned: a literal beside a token, a local
    definition inside a component rule, a literal after a root definition,
    a var() fallback, cased and extended named colours, a hex in a data:
    URL — and the legitimate cases stay clean."""
    probe = tmp_path / "probe.css"
    probe.write_text(body)
    assert bool(_offences(probe)) is expected, body


def test_inline_style_outside_style_element_is_judged(tmp_path: Path) -> None:
    """``style="…"`` and ``.style.cssText = "…"`` carry CSS too."""
    probe = tmp_path / "probe.html"
    probe.write_text(
        '<div style="color:#ff00ff"></div>\n<style>a{color:var(--ink)}</style>\n'
    )
    assert _offences(probe), "an inline style attribute slipped through"
    probe.write_text('<script>el.style.cssText = "color:#ff00ff";</script>\n')
    assert _offences(probe), "a cssText assignment slipped through"
    probe.write_text("<script>// see #765 for the dead-button finding</script>\n")
    assert not _offences(probe), "a PR reference in a comment was flagged"
    for body in (
        "<div style='color:#ff00ff'></div>\n",
        "<script>el.style.cssText = 'color:#ff00ff';</script>\n",
        "<script>el.style.color = 'orange';</script>\n",
        "<script>el.setAttribute('style', 'color:#ff00ff');</script>\n",
        "<script>el.style.setProperty('color', '#ff00ff');</script>\n",
        '<svg><path fill="#ff00ff"/></svg>\n',
        "<svg><circle stroke='#ff00ff'/></svg>\n",
    ):
        probe.write_text(body)
        assert _offences(probe), body
    probe.write_text('<svg><path fill="currentColor" stroke="var(--rule)"/></svg>\n')
    assert not _offences(probe), "token/currentColor SVG was flagged"


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
    problems = []
    # var(--x) in CSS, getPropertyValue("--x") in JS, and the "$tok:--x"
    # sentinels the server emits for the page to resolve.
    ref = re.compile(
        r"var\(\s*(--[\w-]+)|getPropertyValue\(\s*[\"'`](--[\w-]+)|\$tok:(--[\w-]+)"
    )
    # A surface may define an indirection of its own — ``.tone-ok{--tone:
    # var(--ok)}`` — so one rule can read ``var(--tone)`` for any of ten
    # tones. It counts as defined only when its value is itself a token;
    # a local property holding a literal is caught by the literal guard.
    local = re.compile(r"(--[\w-]+)\s*:\s*var\(\s*--[\w-]+\s*\)")
    for relative in _SURFACES + ["GEECS-DataPortal/geecs_portal/figures.py"]:
        text = (_REPO / relative).read_text()
        names = {g for m in ref.finditer(text) for g in m.groups() if g}
        names -= set(local.findall(text))
        # --trace-${i} is built from a prefix at runtime; "--name" is the
        # documentation placeholder in comments explaining the sentinel form.
        names = {n for n in names if not n.endswith("-") and n != "--name"}
        if "--trace-" in text:
            names |= {"--trace-1", "--trace-4"}
        for name in sorted(names):
            if name not in defined:
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
