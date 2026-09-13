"""Rules the page templates have to keep, pinned.

These are not about what a page says — the router tests cover that — but
about how it addresses its own assets and links, which only bites in a
deployment nobody runs locally.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TEMPLATES = sorted(
    (Path(__file__).resolve().parents[1] / "geecs_logbook/templates").glob("*.html")
)


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
