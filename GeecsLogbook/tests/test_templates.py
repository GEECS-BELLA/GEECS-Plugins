"""Rules the page templates have to keep, pinned.

These are not about what a page says — the router tests cover that — but
about how it addresses its own assets and links, which only bites in a
deployment nobody runs locally.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1] / "geecs_logbook"
_TEMPLATES = sorted((_PKG / "templates").glob("*.html"))
_CSS = _PKG / "static/scanlog.css"


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
    pixel-identical to the real tag beside it, which was the exact
    confusion the class was added to remove. Markup assertions cannot see
    it because the markup is correct.
    """
    css = re.sub(r"/\*.*?\*/", " ", _CSS.read_text(), flags=re.S)
    first: dict[str, int] = {}
    for m in re.finditer(r"(?:^|[};])\s*([^{};]+?)\s*\{", css):
        for part in (s.strip() for s in m.group(1).split(",")):
            if re.fullmatch(r"\.[\w-]+", part):
                first.setdefault(part, m.start(1))
    inverted = [
        (sel, base)
        for sel, pos in first.items()
        if (base := sel.rsplit("-", 1)[0]) != sel
        and base in first
        and first[base] > pos
    ]
    assert not inverted, (
        "declared before the rule they vary, so the base wins on source "
        f"order: {inverted}"
    )
