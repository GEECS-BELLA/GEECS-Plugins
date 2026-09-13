"""Rules the page templates have to keep, pinned.

These are not about what a page says — the router tests cover that — but
about how it addresses its own assets and links, which only bites in a
deployment nobody runs locally.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "geecs_logbook"
_TEMPLATES = sorted((_PKG / "templates").glob("*.html"))
_CSS = _PKG / "static/scanlog.css"
_KIT_CSS = _PKG.parents[1] / "GeecsWebTheme/geecs_web_theme/static/kit.css"


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_every_url_for_takes_the_path(template: Path) -> None:
    """``url_for(...)`` is always followed by ``.path``.

    Starlette's ``url_for`` returns an ABSOLUTE url built from the request
    the app saw. Behind TLS termination that is ``http://``, so an absolute
    URL in a ``<script src>`` is a mixed-content block and the script
    silently never loads — which is how the logbook's editor would have
    stopped working on an HTTPS-fronted portal while every test stayed
    green. In an ``href`` it is milder but still wrong: a host-rewriting
    proxy sends the reader to the wrong host.

    Both templates have carried a comment saying to use ``.path`` since
    they were written, and five call sites did not.
    """
    text = template.read_text()
    bare = [
        m.group(0)
        for m in re.finditer(r"url_for\((?:[^()]|\([^()]*\))*\)(?!\.path)", text)
    ]
    assert not bare, (
        f"{template.name}: {len(bare)} url_for call(s) without .path — {bare[:3]}"
    )


def test_every_literal_data_state_is_a_kit_state() -> None:
    """A hand-written ``data-state`` names a state the kit actually colours.

    ``KIT_STATE`` is pinned by ``test_models.py``, but three call sites
    write the attribute literally rather than through it — a failed-scan
    "failed" count, the month page's "today", and an agent entry. A typo in
    any of those renders an uncoloured chip: ``.chip`` still gives a pill
    with inherited colour and a ``currentColor`` dot, so it looks plausible
    in the browser and passes any test that asserts markup. That silent
    shape is precisely what ``geecs_web_theme.STATES`` exists to prevent.
    """
    from geecs_web_theme import STATES

    problems = []
    for template in _TEMPLATES:
        for m in re.finditer(r'data-state="([a-z_][\w-]*)"', template.read_text()):
            if m.group(1) not in STATES:
                problems.append(f"{template.name}: {m.group(1)}")
    assert not problems, (
        f"literal data-state values the kit does not colour: {problems}; "
        f"kit knows {sorted(STATES)}"
    )


def test_a_single_class_variant_is_declared_after_its_base() -> None:
    """``.tag-retired`` must come after ``.tag``, not before it.

    Both are one class, so they tie on specificity and source order alone
    decides — a variant declared first is silently overridden by the base
    it exists to vary. That shipped once: ``.tag-retired`` landed above
    ``.tag`` and the retired-template name rendered accent-coloured,
    pixel-identical to the real tag beside it. Markup assertions cannot
    see it, because the markup is correct.

    The rule only applies to classes that actually land on the SAME
    element. A first version keyed on the name alone and flagged
    ``.entry-body`` as a variant of ``.entry`` — they share a prefix but
    never a element, so ordering them is meaningless, and a guard that
    cries wolf gets switched off. So: read the pairs out of the markup.
    """
    together: set[tuple[str, str]] = set()
    for template in _TEMPLATES:
        text = re.sub(r"\{#.*?#\}", " ", template.read_text(), flags=re.S)
        for attr in re.finditer(r'class="([^"]*)"', text):
            # Blank the Jinja inside the attribute, as the script guard does.
            # Without it `class="entry{% if %} entry-agent{% endif %}"`
            # tokenises to `entry{%`, `if`, `entry-agent{%` — so the pair this
            # guard was rewritten to catch was never even considered, and
            # .avatar-agent/.avatar (which collide on all three properties)
            # went unseen too.
            literal = re.sub(r"\{[%{].*?[%}]\}", " ", attr.group(1), flags=re.S)
            names = [w for w in literal.split() if w]
            for name in names:
                for other in names:
                    if other != name and name.startswith(other + "-"):
                        together.add((f".{name}", f".{other}"))

    css = "\n".join(
        re.sub(r"/\*.*?\*/", " ", path.read_text(), flags=re.S)
        for path in (_CSS, _KIT_CSS)
    )
    last: dict[str, int] = {}
    for m in re.finditer(r"(?:^|(?<=[};{]))\s*([^{};\s][^{};]*?)\s*\{", css):
        for part in (s.strip() for s in m.group(1).split(",")):
            if re.fullmatch(r"\.[\w-]+", part):
                # Assign, do not setdefault. Eight single-class selectors in
                # this sheet are declared TWICE (.entries .avatar .author
                # .stamp .entry-body .insert .composer .errmsg), and the
                # cascade is decided by the LAST one. Recording the first
                # made the guard pass for the wrong reason: a variant placed
                # BETWEEN two copies of its base is still overridden, and
                # that was invisible.
                last[part] = m.start(1)

    inverted = [
        (variant, base)
        for variant, base in sorted(together)
        if variant in last and base in last and last[base] > last[variant]
    ]
    assert not inverted, (
        "declared before the rule they vary, and they share an element so "
        f"source order decides: {inverted}"
    )


def test_no_collapsible_element_is_given_a_display() -> None:
    """A ``<details>`` must keep its default ``display``, or it never folds.

    Setting ``display`` on the element itself — grid, flex, anything but
    block — makes the browser lay out every child regardless of ``open``,
    so the thing renders permanently expanded and clicking does nothing.
    There is no error and no warning; it simply stops being a disclosure.

    This shipped. ``scanlog.css`` carried ``.entry{display:grid}`` from
    when an entry was an ``<article>`` laid out as avatar-plus-body. When
    entries became ``<details>`` the rule stayed, and every note on the
    page was uncollapsible — reported by the owner, invisible to 216
    passing tests, because the markup was correct and only the rendering
    was wrong.

    The classes here are the ones the templates put on a ``<details>``.
    """
    collapsible = set()
    for template in _TEMPLATES:
        for m in re.finditer(r"<details[^>]*?class=\"([^\"]*)\"", template.read_text()):
            # the attribute carries Jinja — `class="entry{% if %} entry-agent{% endif %}"`
            # — so blank the expressions and keep the literal words. A first
            # version required a Jinja-free attribute and therefore skipped
            # the one element this test exists to protect.
            literal = re.sub(r"\{[%{].*?[%}]\}", " ", m.group(1), flags=re.S)
            collapsible |= {w for w in literal.split() if w}
    assert "entry" in collapsible, f"expected the entry among {collapsible}"

    # Lookbehind, not a consuming class: re.finditer resumes after the
    # previous match, so a `}` matched as an anchor is eaten and the NEXT
    # rule has none — every second rule goes unexamined. That flaw shipped
    # once already in the kit's scoping test, found by review; this is the
    # same mistake in a different file, found by the owner reporting that
    # a fix did not work.
    # BOTH sheets: `panel` is on the scan block and `.kit .panel` lives in
    # kit.css, which a scanlog-only guard never opens — adding display:flex
    # there would expand every scan block on the page, guard green.
    css = "\n".join(
        re.sub(r"/\*.*?\*/", " ", path.read_text(), flags=re.S)
        for path in (_CSS, _KIT_CSS)
    )
    # An at-rule's prelude has to go, not just be skipped: the first rule
    # INSIDE `@media (…){ … }` is anchored by the media block's own `{`,
    # which is neither `}` nor `;`, so it never matched at all. Replacing
    # the prelude with a bare `{` keeps the braces balanced and exposes the
    # rules within, which is why the lookbehind admits `{` too.
    # (Same shape as the anchor bug in the kit's scoping
    # test, which is now three times this session.)
    # Statement at-rules first (`@import …;`, `@layer base;`), then blocks,
    # and BOTH bounded at `;`. A greedy `[^{]*` crossed semicolons and
    # newlines, so one `@import` above a rule erased every rule between it
    # and the next `{` in the file — including the one the guard exists to
    # find. Neither sheet has one today, which is exactly why it was silent.
    css = re.sub(r"@[\w-]+[^{;]*;", " ", css)
    css = re.sub(r"@[\w-]+[^{;]*\{", "{", css)
    offenders = []
    for m in re.finditer(r"(?:^|(?<=[};{]))\s*([^{};\s][^{};]*?)\s*\{([^}]*)\}", css):
        body = m.group(2).replace(" ", "")
        if "display:" not in body or body.startswith("display:none"):
            continue
        for part in (s.strip() for s in m.group(1).split(",")):
            if part.startswith("@"):
                continue
            # The LAST compound in the selector is the element being styled.
            # A first version used fullmatch on the whole selector and so
            # only ever caught the bare `.entry{…}` form — `.panel.scan`,
            # `.entries > .entry`, `.entry:not([open])`, `#logbook .entry`
            # and anything inside an @media block all sailed past.
            last = re.split(r"[ >+~]+", part)[-1]
            classes = set(re.findall(r"\.([\w-]+)", last))
            if classes & collapsible:
                offenders.append((part, body[:40]))
    assert not offenders, (
        f"these set display on a <details>, which stops it collapsing: {offenders}"
    )


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_inline_scripts_parse(template: Path) -> None:
    """A page's inline ``<script>`` is syntactically valid JavaScript.

    Nothing else checks this. A template renders fine, every router test
    passes, and the browser silently refuses to execute a block with a
    syntax error — so every behaviour in it dies at once and the suite
    says nothing.

    That shipped: deleting the scan filter removed the handler's body and
    left its closing ``});`` behind, which broke Collapse All and the rail
    jump on the same page. 217 tests were green. The owner found it by
    clicking a button.

    Jinja is blanked to a string literal before parsing — this checks the
    JavaScript's shape, not what any particular render produces.
    """
    # Jinja comments first: `{# … a <script> body is raw text … #}` mentions
    # the tag, and an extractor that does not blank comments starts a match
    # inside one and swallows the real script after it.
    text = re.sub(r"\{#.*?#\}", " ", template.read_text(), flags=re.S)
    # (attributes, body) — a <script type="application/json"> carries a data
    # payload, not code, and node would choke on it. Judge by the type
    # attribute, never by inspecting the body: the first version of this
    # test wrote `"application/json" not in b[:0]`, which tests the empty
    # string and is therefore always true.
    blocks = re.findall(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", text, re.S)
    scripts = [
        body
        for attrs, body in blocks
        if body.strip()
        and not re.search(r'type\s*=\s*"(?!text/javascript|module)', attrs)
    ]
    if not scripts:
        pytest.skip("no inline script in this template")

    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available to parse JavaScript")

    for i, block in enumerate(scripts):
        code = re.sub(r"\{\{.*?\}\}|\{%.*?%\}", '"jinja"', block, flags=re.S)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(code)
            path = fh.name
        try:
            done = subprocess.run(
                [node, "--check", path], capture_output=True, text=True
            )
        finally:
            os.unlink(path)
        assert done.returncode == 0, (
            f"{template.name} inline script #{i + 1} does not parse:\n"
            + done.stderr.strip()
        )


def test_both_books_share_one_collapse_preference() -> None:
    """The two Collapse-All blocks agree on their storage key.

    They are deliberately two: the scan log folds scans *and* entries, the
    ops book only entries, and a third file for ~20 lines is not worth it.
    But what genuinely couples them is the key — how dense a logbook reads
    is one preference, not two — and nothing pinned it. This does, without
    a new module: both name the same key, and neither names another.
    """
    keys: set[str] = set()
    for template in _TEMPLATES:
        text = re.sub(r"\{#.*?#\}", " ", template.read_text(), flags=re.S)
        for block in re.findall(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", text, re.S
        ):
            keys |= set(re.findall(r"localStorage\.\w+\(\s*[\"\']([^\"\']+)", block))
            keys |= {
                m for m in re.findall(r'(?:const|let|var)\s+KEY\s*=\s*"([^"]+)"', block)
            }
    assert keys == {"scanlog.expandAll"}, keys
