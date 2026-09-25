"""Every web surface styles through the tokens, never with a literal colour.

This is the test that keeps "changing a theme is easy" true. A token layer
only works if components use it; the moment one rule hardcodes a colour,
that component silently stops responding to themes and nobody finds out
until they switch and one thing stays the wrong colour.

The checks read CSS through **tinycss2** (``geecs_web_theme.testing``), a
real parser, not regular expressions. The regex versions that preceded this
file had six silent holes found over three review rounds — a consumed
``@media`` anchor, a filter testing the empty string, a match starting
inside a Jinja comment, a first-declaration lookup where the cascade reads
the last — and every hole passed green. A parser-based check that is wrong
fails loudly instead. Ask a real parser.

What is pinned, and why each earns its place:

- **no literal colours** in any registered surface (the rule itself);
- **every token a surface references is defined**, and **every palette
  defines every token** — the ``--surface2`` / ``--surface-2`` mismatch
  shipped every hover fill transparent and only a cross-file check sees it;
- **the kit introduces no token** the theme does not declare (one
  vocabulary, one file to read);
- **the vocabularies agree** across Python, ``theme-boot.js`` and the CSS
  (themes, densities, status words, pane states) — a mistyped
  ``data-state`` renders a plausible neutral chip and survives review;
- **every kit rule is scoped to** ``.kit`` — the portal's ``.pane`` tab
  mechanism ties on specificity with an ungated ``.pane``;
- **the reference page shows only what the kit styles** — it is the page
  adopters copy from.

Deleted on purpose (2026-09-13): the ``[hidden]`` ordering check (a one-time
bug, now a comment beside the rule), the per-component specimen list for the
reference page (a maintenance list, not an invariant), and the probe cases
that pinned holes in the old regexes (moot with a parser).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from geecs_web_theme import (
    DEFAULT_DENSITY,
    DEFAULT_THEME,
    DENSITIES,
    PANE_STATES,
    STATES,
    THEMES,
)
from geecs_web_theme.testing import (
    attribute_selector_values,
    classes_used,
    colour_literals,
    defined_tokens,
    html_style_sources,
    js_colour_literals,
    referenced_tokens,
    rule_selectors,
    styled_classes,
    token_indirections,
    unknown_data_states,
)

_REPO = Path(__file__).resolve().parents[2]
_THEME_CSS = _REPO / "GeecsWebTheme/geecs_web_theme/static/theme.css"
_KIT_CSS = _REPO / "GeecsWebTheme/geecs_web_theme/static/kit.css"
_KIT_HTML = _REPO / "GeecsWebTheme/geecs_web_theme/static/kit.html"
_BOOT_JS = _REPO / "GeecsWebTheme/geecs_web_theme/static/theme-boot.js"

#: The spacing scale a density block owns. Adding a fourth spacing token
#: means adding it here — at which point the density block that forgot it
#: fails, which is the point.
_DENSITY_TOKENS = {"--pad", "--row-h", "--gap"}

#: The web surfaces bound by the rule. Adding a surface means adding it
#: here — a new page that skips the tokens should fail loudly, not quietly.
#: Vendored third-party assets (the portal's Plotly bundle, the logbook's
#: `static/vendor/katex-*`) are exempt: they are upstream bytes, not a
#: surface, and the page around them still styles through the tokens.
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
    "GeecsLogbook/geecs_logbook/static/math.js",
    "ScanAnalysis/scan_analysis/config_editor/static/editor.css",
    "ScanAnalysis/scan_analysis/config_editor/templates/editor.html",
    "GeecsScanner/geecs_scanner/templates/console.html",
    "GeecsScanner/geecs_scanner/static/scanner.css",
    "GeecsScanner/geecs_scanner/static/scanner.js",
]

#: Selectors whose rules may carry a literal, each with its reason. This is
#: where the rule erodes if it erodes; expect the reason to be read.
_ALLOWED = {
    ".themepick .sw": "the swatch ring is a neutral grey that reads on any ground",
    ".themepick .sw-bella": "each swatch shows ITS palette whichever one is active",
    ".themepick .sw-laser": "each swatch shows ITS palette whichever one is active",
    ".themepick .sw-plasma": "each swatch shows ITS palette whichever one is active",
    "img.plot": "matplotlib renders onto white; the frame matches its own ground",
    ".ce-preview img": "matplotlib renders onto white; the frame matches its own ground",
}


def _offences(path: Path) -> list[str]:
    """Every colour literal one surface carries, however it carries it."""
    text = path.read_text()
    if path.suffix in {".html", ".htm"}:
        css_chunks, scripts = html_style_sources(text)
        found = [
            lit
            for chunk in css_chunks
            for lit in colour_literals(chunk, allowed=_ALLOWED)
        ]
        found += [
            f"script string: {s}" for body in scripts for s in js_colour_literals(body)
        ]
        return found
    if path.suffix == ".js":
        return [f"string: {s}" for s in js_colour_literals(text)]
    return colour_literals(text, allowed=_ALLOWED)


@pytest.mark.parametrize("relative", _SURFACES)
def test_surface_uses_only_tokens(relative: str) -> None:
    """A web surface carries no unexplained colour literal.

    If this fails, the fix is almost always to replace the literal with a
    token from ``theme.css``. If the literal is genuinely correct — it must
    look the same under every palette — add its selector to ``_ALLOWED``
    *with a reason*.
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
    "css,expected",
    [
        ("a{color:var(--accent);background:#ff00ff}", True),
        ("a{color:var(--accent);box-shadow:0 0 0 2px #00ff00}", True),
        (".foo{--local:#ff00ff;color:var(--local)}", True),
        (":root{--x:#123456;color:#ff00ff}", True),
        ("a{color:var(--accent,#ff00ff)}", True),
        ("a{color:White}", True),
        ("a{color:orange}", True),
        ("a{background:url(\"data:image/svg+xml,%3Csvg fill='%23ff00ff'/%3E\")}", True),
        (":root .foo{--x:#ff00ff}", True),
        ("@media (max-width:900px){.a{color:#fff}}", True),
        ("@media (a){@supports (b){.a{color:#fff}}}", True),
        (".a{color:var(--ink); &:hover{color:#ff00ff}}", True),
        ("a{background:url(data:image/svg+xml,%3Csvg fill='%23ff00ff'/%3E)}", True),
        (
            "a{background:url(data:image/svg+xml,%3Csvg%20fill=%27%23ff00ff%27/%3E)}",
            True,
        ),
        ("a{background:url(icons.svg#frag)}", False),
        ("a{box-shadow:0 1px rgba(17,24,33,.4)}", True),
        ("/* #fff in a comment */ a{color:var(--ink)}", False),
        ("a{box-shadow:0 1px 2px rgba(0,0,0,.4)}", False),
        (":root{--x:#123456}", False),
        (':root[data-theme="laser"][data-mode="dark"]{--x:#123456}', False),
        ("a{color:var(--accent)}", False),
        ("@keyframes k{0%{opacity:1}}", False),
    ],
)
def test_probe_css_literals(css: str, expected: bool) -> None:
    """The cases that define the rule — a literal beside a token, a local
    definition inside a component, a var() fallback, a data: URL, a rule
    inside @media — and the legitimate cases that must stay clean."""
    assert bool(colour_literals(css)) is expected, css


@pytest.mark.parametrize(
    "html,expected",
    [
        ('<div style="color:#ff00ff"></div>', True),
        ("<div style='color:#ff00ff'></div>", True),
        ('<svg><path fill="#ff00ff"/></svg>', True),
        ("<svg><circle stroke='#ff00ff'/></svg>", True),
        ('<script>el.style.cssText = "color:#ff00ff";</script>', True),
        ("<script>el.style.color = 'orange';</script>", True),
        ("<script>el.setAttribute('style', 'color:#ff00ff');</script>", True),
        ("<script>el.style.setProperty('color', '#ff00ff');</script>", True),
        (
            '<style>a{color:var(--ink)}</style><div style="color:var(--ink)"></div>',
            False,
        ),
        ('<svg><path fill="currentColor" stroke="var(--rule)"/></svg>', False),
        ("<script>// see #765 for the dead-button finding</script>", False),
        ('<script>document.querySelector("#now")</script>', False),
        ('<script>el.style.cssText = "background:var(--surface-2)";</script>', False),
        ('{# <div style="color:#ff00ff"> in a Jinja comment #}', False),
        ('<div style="{{ inline }}"></div>', False),
        ('<div style="color:#ff00ff; width:{{ w }}px"></div>', True),
        ('<svg><path fill="{{ tone }}" stroke="#ff00ff"/></svg>', True),
        ("<script>const css = `color:#ff00ff`;</script>", True),
        (
            '<script>el.innerHTML = `<b class="chip" data-state="${s}">${w}</b>`;</script>',
            False,
        ),
    ],
)
def test_probe_html_and_script_literals(html: str, expected: bool) -> None:
    """CSS hides in ``style=`` attributes, SVG paint attributes and script
    strings; a PR reference in a comment and an element id are not colours."""
    css_chunks, scripts = html_style_sources(html)
    found = [lit for c in css_chunks for lit in colour_literals(c)]
    found += [s for body in scripts for s in js_colour_literals(body)]
    assert bool(found) is expected, (html, found)


# ---------------------------------------------------------------- tokens

_ROOT_ONLY = {
    "--ff-ui", "--ff-mono", "--ff-prose", "--r", "--r-lg", "--bw", "--tk",
    "--pad", "--row-h", "--gap", "--shell-max", "--scrim", "--lift",
}  # fmt: skip


def test_every_palette_defines_every_token() -> None:
    """No palette block is missing a token another defines.

    A missing token does not fail loudly — it inherits the bare ``:root``
    value, so one theme silently shows another's colour. The typeface and
    structure defaults are root-only by design; a palette that overrides
    one must then do so in every block, which this test will say.
    """
    blocks = defined_tokens(_THEME_CSS.read_text())
    assert len(blocks) >= 7, f"expected root + 3×(light,dark); found {sorted(blocks)}"
    expected = max(blocks.values(), key=len) - _ROOT_ONLY
    for selector, tokens in sorted(blocks.items()):
        missing = expected - tokens
        assert not missing, f"{selector} is missing {sorted(missing)}"
    assert _DENSITY_TOKENS <= blocks[":root"], (
        "theme.css :root must declare the spacing scale"
    )


def test_every_referenced_token_is_defined() -> None:
    """Every ``var(--x)`` in every surface names a token the theme defines.

    This is the test that would have caught ``--surface2`` vs
    ``--surface-2``: 34 references and zero definitions, invisible to a
    check that only compared theme blocks against each other. Scripts
    count too — the portal's run page writes one token through
    ``cssText`` — and a surface's own ``.tone-ok{--tone:var(--ok)}``
    indirection is a definition, not an undefined reference.
    """
    defined: set[str] = set()
    for tokens in defined_tokens(_THEME_CSS.read_text()).values():
        defined |= tokens
    problems = []
    for relative in _SURFACES + ["GEECS-DataPortal/geecs_portal/figures.py"]:
        path = _REPO / relative
        text = path.read_text()
        names: set[str] = set()
        local: set[str] = set()
        if path.suffix in {".html", ".htm"}:
            css_chunks, scripts = html_style_sources(text)
            for chunk in css_chunks:
                names |= referenced_tokens(chunk)
                local |= token_indirections(chunk)
            for body in scripts:
                names |= referenced_tokens(body, css=False)
        elif path.suffix == ".css":
            names = referenced_tokens(text)
            local = token_indirections(text)
        else:
            names = referenced_tokens(text, css=False)
        names -= local
        # --trace-${i} is built from a prefix at runtime.
        names = {n for n in names if not n.endswith("-") and n != "--name"}
        if "--trace-" in text:
            names |= {"--trace-1", "--trace-4"}
        problems += [f"{path.name}: {n}" for n in sorted(names - defined)]
    assert not problems, "referenced but never defined:\n  " + "\n  ".join(problems)


def test_kit_defines_no_token_the_theme_does_not() -> None:
    """``kit.css`` overrides tokens; it never introduces one.

    The kit legitimately redefines the spacing scale under ``[data-density]``
    and ``--shell-max`` on a wide shell — both overrides of tokens the theme
    already declares. A *new* name here would be a second authority.
    """
    theme_tokens = set().union(*defined_tokens(_THEME_CSS.read_text()).values())
    kit_tokens = set().union(*defined_tokens(_KIT_CSS.read_text()).values())
    introduced = kit_tokens - theme_tokens
    assert not introduced, (
        f"kit.css introduces {sorted(introduced)} — declare it in theme.css "
        "so there is one token vocabulary, not two"
    )


# ---------------------------------------------------------- vocabularies


def _boot_list(key: str) -> list[str]:
    boot = _BOOT_JS.read_text()
    return re.findall(r'"(\w+)"', re.search(rf"{key}:\s*\[([^\]]*)\]", boot).group(1))


def _boot_default(key: str) -> str:
    return re.search(rf'{key}:\s*"(\w+)"', _BOOT_JS.read_text()).group(1)


def test_python_boot_script_and_css_agree_on_themes() -> None:
    """``THEMES`` / ``DEFAULT_THEME``, ``theme-boot.js`` and the palette blocks agree.

    The JS is the runtime authority (it stamps the page); the Python is
    what hosts read; the CSS is what paints. Three copies by necessity.
    """
    assert _boot_list("themes") == list(THEMES)
    assert _boot_default("defaultTheme") == DEFAULT_THEME
    blocks = defined_tokens(_THEME_CSS.read_text())
    for name in THEMES:
        assert f':root[data-theme="{name}"]' in blocks, f"{name} has no CSS block"
        assert f':root[data-theme="{name}"][data-mode="dark"]' in blocks, (
            f"{name} has no dark block"
        )


def test_python_boot_script_and_kit_agree_on_densities() -> None:
    """``DENSITIES`` / ``DEFAULT_DENSITY``, ``theme-boot.js`` and ``kit.css`` agree.

    The default needs no block — it is what ``theme.css`` declares. Every
    other density overrides exactly the spacing scale: an EMPTY block once
    passed a weaker version of this test, which is exactly the drift it
    has to catch.
    """
    assert _boot_list("densities") == list(DENSITIES)
    assert _boot_default("defaultDensity") == DEFAULT_DENSITY
    assert DEFAULT_DENSITY in DENSITIES
    blocks = defined_tokens(_KIT_CSS.read_text())
    for name in DENSITIES:
        if name == DEFAULT_DENSITY:
            continue
        selector = f':root[data-density="{name}"]'
        assert selector in blocks, f"{name} has no kit block"
        assert blocks[selector] == _DENSITY_TOKENS, (
            f"{selector} defines {sorted(blocks[selector])}, expected {sorted(_DENSITY_TOKENS)}"
        )


def test_status_vocabulary_is_pinned_to_the_kit() -> None:
    """``STATES`` and ``kit.css`` name the same statuses, chip and dot alike.

    A mistyped state is silent: ``data-state="no_data"`` matches no rule and
    ``.chip`` still renders a plausible neutral pill. Pinning both directions
    means the CSS cannot style a status the vocabulary lacks, and the
    vocabulary cannot name one the CSS does not colour.
    """
    kit = _KIT_CSS.read_text()
    for component in ("chip", "dot"):
        styled = attribute_selector_values(kit, "data-state", on_class=component)
        assert styled == set(STATES), (
            f".{component} styles {sorted(styled)}; STATES names {sorted(STATES)}"
        )


def test_pane_states_are_pinned_to_the_kit() -> None:
    """The CSS styles a subset of ``PANE_STATES`` and the reference page names
    nothing outside the two vocabularies.

    Not every pane state needs its own rule — loading and empty share the
    neutral ground on purpose — so the CSS pin runs one way. ``data-age``
    takes exactly one value, ``stale``; a fresh reading carries none.
    """
    kit = _KIT_CSS.read_text()
    styled = attribute_selector_values(
        kit, "data-state", on_class="state"
    ) | attribute_selector_values(kit, "data-state", on_class="banner")
    unknown = styled - set(PANE_STATES)
    assert not unknown, f"kit.css styles pane states {sorted(unknown)}"
    page = _KIT_HTML.read_text()
    stray = unknown_data_states(page, {**STATES, **PANE_STATES})
    assert not stray, (
        f"kit.html uses {sorted(set(stray))}, neither a status nor a pane state"
    )
    ages = set(re.findall(r'data-age=["\']([^"\']*)["\']', page))
    assert ages <= {"stale"}, (
        f"kit.html uses data-age={sorted(ages)}; the only value is 'stale'"
    )


# --------------------------------------------------------------- the kit

#: The only selectors allowed to escape the ``.kit`` scope: the density
#: blocks, which must match the root element, and the body rule that
#: carries the class itself.
_UNSCOPED_OK = re.compile(r'^(?::root\[data-density="[\w-]+"\]|body\.kit)$')


def test_kit_rules_are_scoped_to_the_kit_class() -> None:
    """Every kit rule is gated on ``.kit``, so a surface can adopt per page.

    Both surfaces that adopted this already used several of these class
    names, one load-bearingly: the portal's run page is ``.pane{display:
    none}`` / ``.pane.on{display:block}`` — its tab mechanism — which ties
    on specificity with an ungated ``.pane`` and would be decided by
    stylesheet order alone. ``.kitchen`` is not ``.kit``.
    """
    ungated = [
        s
        for s in rule_selectors(_KIT_CSS.read_text())
        if not re.match(r"^\.kit(?![\w-])", s) and not _UNSCOPED_OK.match(s)
    ]
    assert not ungated, f"kit.css rules not scoped to .kit: {sorted(set(ungated))}"


def test_kit_reference_page_assets_all_exist() -> None:
    """Every file ``kit.html`` pulls in sits beside it — a renamed asset is a
    blank page nobody sees until they open it."""
    refs = re.findall(r'(?:src|href)="([^"#:]+)"', _KIT_HTML.read_text())
    assert refs, "kit.html references nothing — did the page lose its head?"
    for ref in refs:
        assert (_KIT_HTML.parent / ref).is_file(), f"kit.html references missing {ref}"


def test_reference_page_demonstrates_only_what_the_kit_provides() -> None:
    """Every class ``kit.html`` uses is styled by the kit or the theme.

    It is the page adopters copy from, so a component shown there that the
    kit does not style is worse than one missing: it gets copied, renders
    as browser defaults, and the adopter writes their own CSS for it — the
    per-surface divergence this package exists to stop.
    """
    missing = sorted(
        classes_used(_KIT_HTML.read_text())
        - styled_classes(_KIT_CSS.read_text(), _THEME_CSS.read_text())
    )
    assert not missing, (
        f"kit.html shows {missing} but nothing styles them — either the kit "
        "owes the component or the page should not be demonstrating it"
    )


def test_referenced_tokens_sees_css_a_script_builds() -> None:
    """``cssText = "…var(--x)…"`` in a script counts as a reference.

    The portal's run page writes one token this way; a check over
    stylesheet files alone would have let ``var(--surfce-2)`` there ship.
    """
    js = 'note.style.cssText = "padding:2px; background:var(--surfce-2)";'
    assert referenced_tokens(js, css=False) == {"--surfce-2"}


def test_allowlist_matches_whole_selector_parts() -> None:
    """``img.plot`` covers ``img.plot`` and ``img.plot:hover`` but not
    ``img.plotwrap``, ``img.plot-x``, nor the other half of ``img.plot, .other``."""
    marks = {"img.plot": ""}
    assert not colour_literals("img.plot{background:#fff}", allowed=marks)
    assert not colour_literals("img.plot:hover{background:#fff}", allowed=marks)
    assert not colour_literals("img.plot .x{background:#fff}", allowed=marks)
    assert colour_literals("img.plotwrap{background:#fff}", allowed=marks)
    assert colour_literals("img.plot-x{background:#fff}", allowed=marks)
    assert colour_literals("img.plot, .other{background:#fff}", allowed=marks)


def test_nested_rules_keep_their_parent_scope() -> None:
    """A rule nested inside ``.kit .a`` is reported as ``.kit .a &:hover`` —
    parent first — and an unscoped one inside ``@media{@supports{}}`` is seen."""
    assert rule_selectors(".kit .a{ &:hover{x:1} }") == [".kit .a", ".kit .a &:hover"]
    assert rule_selectors("@media (a){@supports (b){.pane{x:1}}}") == [".pane"]
    # comma lists on either side cross-multiply
    assert rule_selectors(".a, .b{ &:hover, &:focus{x:1} }") == [
        ".a", ".b", ".a &:hover", ".a &:focus", ".b &:hover", ".b &:focus",
    ]  # fmt: skip
    # declarations directly inside a nested at-rule belong to the parent
    assert colour_literals(".kit .a{ @media (min-width:1px){ color:#ff00ff } }") == [
        ".kit .a: color: #ff00ff"
    ]
    assert colour_literals(".kit .a{ @media (x){ &:hover{color:#ff00ff} } }")
    # an at-rule whose body is declarations only yields nothing
    assert (
        rule_selectors("@font-face{font-family:x;src:url(x.woff2)} @page{margin:1cm}")
        == []
    )


def test_a_token_named_in_a_css_comment_is_not_a_reference() -> None:
    """Comments are structure to the parser; only a script's built CSS text is
    scanned as text, and there ``var(`` counts."""
    assert referenced_tokens("/* was var(--old) */ a{color:var(--ink)}") == {"--ink"}
    assert referenced_tokens(
        "// was var(--old)\nel.style.cssText = 'x'", css=False
    ) == {"--old"}


def test_selector_lists_split_on_top_level_commas_only() -> None:
    """``:is(.panel, .well)`` is one selector, and both classes count as styled."""
    from geecs_web_theme.testing import selector_parts

    assert selector_parts(".kit :is(.panel, .well), .kit .x") == [
        ".kit :is(.panel, .well)",
        ".kit .x",
    ]
    assert rule_selectors(".kit :is(.panel, .well){x:1}") == [".kit :is(.panel, .well)"]
    # :not()/:has() name a class without styling it
    assert styled_classes(".kit :is(.panel, .well):not(.dim){x:1}") == {
        "kit", "panel", "well",
    }  # fmt: skip
    assert styled_classes(".kit .a:has(.thumb){x:1}") == {"kit", "a"}
    assert rule_selectors(".kit .a{ &:is(.b, .c){x:1} }") == [
        ".kit .a",
        ".kit .a &:is(.b, .c)",
    ]
