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


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_inline_scripts_parse(template: Path) -> None:
    """A page's inline ``<script>`` is syntactically valid JavaScript.

    Nothing else checks this. A template renders fine, every router test
    passes, and the browser silently refuses to execute a block with a
    syntax error — so every behaviour in it dies at once and the suite says
    nothing. That shipped: deleting the scan filter left its closing ``});``
    behind, which broke Collapse All and the rail jump together while 217
    tests were green.

    It shells out to ``node --check`` rather than parsing anything itself.
    That is deliberate and is why this guard has never produced a false
    result: every hand-rolled scanner written alongside it had a silent
    coverage gap — a consumed regex anchor, a filter testing the empty
    string, a match starting inside a Jinja comment, a first-declaration
    lookup where the cascade reads the last. Ask a real parser.

    Jinja is blanked to a string literal first: this checks the
    JavaScript's shape, not what any particular render produces.
    """
    # Jinja comments first: `{# … a <script> body is raw text … #}` mentions
    # the tag, and an extractor that does not blank comments starts a match
    # inside one and swallows the real script after it.
    text = re.sub(r"\{#.*?#\}", " ", template.read_text(), flags=re.S)
    # (attributes, body) — a <script type="application/json"> carries a data
    # payload, not code. Judge by the type attribute, never by inspecting
    # the body.
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
    What genuinely couples them is the key — how dense a logbook reads is
    one preference, not two — and nothing pinned it.
    """
    keys: set[str] = set()
    for template in _TEMPLATES:
        text = re.sub(r"\{#.*?#\}", " ", template.read_text(), flags=re.S)
        for block in re.findall(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", text, re.S
        ):
            keys |= set(re.findall(r"localStorage\.\w+\(\s*[\"\']([^\"\']+)", block))
            keys |= set(re.findall(r'(?:const|let|var)\s+KEY\s*=\s*"([^"]+)"', block))
    assert keys == {"scanlog.expandAll"}, keys
