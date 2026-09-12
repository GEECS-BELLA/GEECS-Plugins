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
_KIT_CSS = _REPO / "GeecsWebTheme/geecs_web_theme/static/kit.css"

#: The spacing scale a density block owns. theme.css declares these in the
#: bare :root (the comfortable values) and every non-default density block
#: in kit.css overrides exactly this set. Adding a fourth spacing token
#: means adding it here too — at which point the density blocks that forgot
#: it fail, which is the point.
_DENSITY_TOKENS = {"--pad", "--row-h", "--gap"}
_KIT_HTML = _REPO / "GeecsWebTheme/geecs_web_theme/static/kit.html"

#: The web surfaces bound by the rule. Adding a surface means adding it
#: here — a new page that skips the tokens should fail loudly, not quietly.
_SURFACES = [
    "GeecsWebTheme/geecs_web_theme/static/theme.css",
    "GeecsWebTheme/geecs_web_theme/static/kit.css",
    "GeecsWebTheme/geecs_web_theme/static/kit.html",
    "GeecsWebTheme/geecs_web_theme/static/kit.js",
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
    # Root-only by design: a palette block carries colour, and these are
    # the typeface and structure defaults every theme shares until one
    # wants its own. Adding a palette override for one is a deliberate act
    # — it means that theme reads differently, which is the point — and it
    # then has to appear in every block, which this test will say.
    root_only = {
        "--ff-ui",
        "--ff-mono",
        "--ff-prose",
        "--r",
        "--r-lg",
        "--bw",
        "--tk",
        "--pad",
        "--row-h",
        "--gap",
        "--shell-max",
        "--scrim",
        "--lift",
    }
    expected = max(blocks.values(), key=len) - root_only
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


def test_python_and_boot_script_agree_on_the_density_list() -> None:
    """``geecs_web_theme.DENSITIES`` and ``theme-boot.js`` agree, and the
    kit implements every density that is not the default.

    Same arrangement as the theme list, for the same reason: the JS stamps
    the page, the Python is what a host reads, and a third copy of the
    names lives in the CSS. The default needs no block — it is what
    ``theme.css`` already defines.
    """
    import sys

    sys.path.insert(0, str(_REPO / "GeecsWebTheme"))
    from geecs_web_theme import DEFAULT_DENSITY, DENSITIES  # noqa: E402

    boot = (_REPO / "GeecsWebTheme/geecs_web_theme/static/theme-boot.js").read_text()
    js_list = re.findall(
        r'"(\w+)"', re.search(r"densities:\s*\[([^\]]*)\]", boot).group(1)
    )
    js_default = re.search(r'defaultDensity:\s*"(\w+)"', boot).group(1)
    assert js_list == list(DENSITIES), (js_list, list(DENSITIES))
    assert js_default == DEFAULT_DENSITY, (js_default, DEFAULT_DENSITY)
    assert DEFAULT_DENSITY in DENSITIES

    kit = _KIT_CSS.read_text()
    for name in DENSITIES:
        if name == DEFAULT_DENSITY:
            continue
        block = re.search(r':root\[data-density="%s"\]\s*\{([^}]*)\}' % name, kit)
        assert block, f"{name} has no kit block"
        # Not just "the selector is present": an EMPTY block passed the first
        # version of this test, which is exactly the drift it has to catch —
        # the comfortable values live in theme.css and the overrides here, so
        # forgetting one leaves compact silently showing a comfortable value.
        defined = set(re.findall(r"(--[\w-]+)\s*:", block.group(1)))
        assert defined == _DENSITY_TOKENS, (
            f'[data-density="{name}"] defines {sorted(defined)}, '
            f"expected {sorted(_DENSITY_TOKENS)}"
        )

    # _blocks() reads raw text, and theme.css's header comment mentions
    # ":root" — which the selector regex then runs together with the real
    # block. Existing callers only take max(...) by length so they never
    # noticed; keying by name needs the comments gone first.
    bare = re.sub(r"/\*.*?\*/", " ", _THEME_CSS.read_text(), flags=re.S)
    missing = _DENSITY_TOKENS - _blocks(bare)[":root"]
    assert not missing, f"theme.css :root does not declare {sorted(missing)}"


def test_kit_defines_no_token_the_theme_does_not() -> None:
    """``kit.css`` overrides tokens; it never introduces one.

    This is what keeps ``theme.css`` the single place to look for the
    vocabulary. The kit legitimately redefines the spacing scale under
    ``[data-density]`` and ``--shell-max`` on a wide shell — both are
    overrides of tokens the theme already declares. A *new* name here
    would be a second authority, and the next surface would have two
    files to read instead of one.
    """
    theme_tokens: set[str] = set()
    for tokens in _blocks(_THEME_CSS.read_text()).values():
        theme_tokens |= tokens
    kit_defined = set(re.findall(r"(--[\w-]+)\s*:", _KIT_CSS.read_text()))
    introduced = kit_defined - theme_tokens
    assert not introduced, (
        f"kit.css introduces {sorted(introduced)} — declare it in theme.css "
        "so there is one token vocabulary, not two"
    )


def test_kit_reference_page_assets_all_exist() -> None:
    """Every file ``kit.html`` pulls in sits beside it.

    The page is static and relative on purpose, so it works under any mount
    prefix. That also means a renamed asset fails silently in a browser —
    a blank page nobody sees until they open it. Here it fails loudly.
    """
    page = _KIT_HTML.read_text()
    refs = re.findall(r'(?:src|href)="([^"#:]+)"', page)
    assert refs, "kit.html references nothing — did the page lose its head?"
    for ref in refs:
        assert (_KIT_HTML.parent / ref).is_file(), f"kit.html references missing {ref}"


def test_status_vocabulary_is_pinned_to_the_kit() -> None:
    """``geecs_web_theme.STATES`` and ``kit.css`` name the same statuses.

    A mistyped state is the dangerous case and it is silent: a
    ``data-state="no_data"`` matches no rule, and ``.chip`` still renders a
    pill with ``border-color:transparent``, inherited colour and a
    ``currentColor`` dot — a plausible neutral chip that survives both
    review and the browser. Pinning both directions means the CSS cannot
    style a status the vocabulary does not have, and the vocabulary cannot
    name one the CSS does not colour.
    """
    import sys

    sys.path.insert(0, str(_REPO / "GeecsWebTheme"))
    from geecs_web_theme import STATES  # noqa: E402

    kit = _KIT_CSS.read_text()
    styled = set(re.findall(r'\.chip\[data-state=["\']([\w-]+)["\']\]', kit))
    assert styled == set(STATES), (
        f"kit.css styles {sorted(styled)}; STATES names {sorted(STATES)}"
    )
    # Every status also needs the dot form, used where the row label carries
    # the word instead.
    dots = set(re.findall(r'\.dot\[data-state=["\']([\w-]+)["\']\]', kit))
    assert dots == set(STATES), (
        f"kit.css dots {sorted(dots)}; STATES names {sorted(STATES)}"
    )


def test_pane_states_are_pinned_to_the_kit() -> None:
    """No surface names a pane state outside ``PANE_STATES``.

    Unlike the statuses, not every pane state needs its own rule — loading,
    empty and denied share the neutral ground on purpose, and only error and
    stale take a colour. So the pin runs one way for the CSS (it may style a
    subset, never something outside the vocabulary) and strictly for the
    reference page, which is the copy people will imitate.
    """
    import sys

    sys.path.insert(0, str(_REPO / "GeecsWebTheme"))
    from geecs_web_theme import PANE_STATES, STATES  # noqa: E402

    kit = _KIT_CSS.read_text()
    styled = set(
        re.findall(r'\.(?:state|banner)\[data-state=["\']([\w-]+)["\']\]', kit)
    )
    unknown = styled - set(PANE_STATES)
    assert not unknown, (
        f"kit.css styles pane states {sorted(unknown)} not in PANE_STATES"
    )

    page = _KIT_HTML.read_text()
    used = set(re.findall(r'data-state=["\']([\w-]+)["\']', page))
    stray = used - set(PANE_STATES) - set(STATES)
    assert not stray, (
        f"kit.html uses {sorted(stray)}, which is neither a status nor a pane "
        "state — a typo here is invisible in a browser"
    )


def _rule_selectors(css: str) -> list[str]:
    """Every rule selector in a stylesheet, at any nesting depth.

    A regex cannot do this. The first version of the caller used one, and
    it consumed the ``{`` of each ``@media`` prelude — so the FIRST rule
    inside every media block lost its anchor and was never examined. Four
    blocks, four invisible rules, one of them the mobile shell collapse.
    A brace walk has no such blind spot: at-rule preludes are recognised
    and skipped, ``@keyframes`` bodies are skipped whole (their ``0%`` and
    ``from`` stops are not selectors), and everything else yields.
    """
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    out: list[str] = []

    def walk(text: str) -> None:
        i, n = 0, len(text)
        while i < n:
            brace = text.find("{", i)
            if brace == -1:
                return
            head = text[i:brace].strip()
            depth, k = 0, brace
            while k < n:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            body = text[brace + 1 : k]
            if head.startswith("@keyframes"):
                pass  # stops are not selectors
            elif head.startswith("@"):
                walk(body)  # @media and friends: the rules inside still count
            else:
                out.extend(s.strip() for s in head.split(",") if s.strip())
            i = k + 1

    walk(css)
    return out


#: The only selectors allowed to escape the ``.kit`` scope: the density
#: blocks, which must match the root element, and the body rule that
#: carries the class itself.
_UNSCOPED_OK = re.compile(r'^(?::root\[data-density="[\w-]+"\]|body\.kit)$')
#: ``.kit`` as a whole class token — ``.kitchen`` is a different class and
#: must not be waved through by a bare string prefix.
_KIT_SCOPED = re.compile(r"^\.kit(?![\w-])")


def test_kit_rules_are_scoped_to_the_kit_class() -> None:
    """Every kit rule is gated on ``.kit``, so a surface can adopt per page.

    Both surfaces that will adopt this already use several of these class
    names, one of them load-bearingly: the portal's run page is
    ``.pane{display:none}`` / ``.pane.on{display:block}`` — its tab
    mechanism — which ties on specificity with an ungated ``.pane`` here and
    would be decided by stylesheet order alone.
    """
    ungated = [
        s
        for s in _rule_selectors(_KIT_CSS.read_text())
        if not _KIT_SCOPED.match(s) and not _UNSCOPED_OK.match(s)
    ]
    assert not ungated, f"kit.css rules not scoped to .kit: {sorted(set(ungated))}"


@pytest.mark.parametrize(
    "css,expected",
    [
        # The three shapes the first version of this test waved through.
        ("@media (max-width:900px){\n  .shell{gap:1px}\n}\n", True),
        (".kitchen{display:flex}\n", True),
        (":root .pane{display:flex}\n", True),
        # …and the legitimate ones stay legitimate.
        ("@media (max-width:900px){\n  .kit .shell{gap:1px}\n}\n", False),
        (':root[data-density="compact"]{--pad:9px}\n', False),
        ("body.kit{margin:0}\n", False),
        ("@keyframes k{0%,100%{opacity:1}}\n", False),
        (".kit .panel > header{gap:1px}\n", False),
    ],
)
def test_probe_scoping(css: str, expected: bool) -> None:
    """Each hole the re-review found, pinned, plus the cases that must pass."""
    ungated = [
        s
        for s in _rule_selectors(css)
        if not _KIT_SCOPED.match(s) and not _UNSCOPED_OK.match(s)
    ]
    assert bool(ungated) is expected, (css, ungated)
