"""Test helpers for the rules every GEECS web surface's templates keep.

Three rules the logbook learned the hard way and the scanner copied by
hand, as importable helpers so the next surface writes three one-line tests
instead of a third copy of the logic. Standard library only — a package's
test suite imports this without the ``web`` extra.

- :func:`bare_url_for_calls` — a ``url_for(...)`` not followed by ``.path``.
  Starlette's ``url_for`` returns an ABSOLUTE url built from the request the
  app saw; behind TLS termination that is ``http://``, so an absolute URL in
  a ``<script src>`` is a mixed-content block and the script silently never
  loads while every test stays green.
- :func:`unknown_data_states` — a literal ``data-state="…"`` naming a word
  the kit does not colour. ``.chip`` still renders a plausible neutral pill
  for an unknown state, so the typo survives review and the browser.
- :func:`inline_scripts` and :func:`javascript_syntax_error` — a page's
  inline ``<script>`` blocks, and whether one fails ``node --check``. A
  browser silently refuses to run a block with a syntax error, so every
  behaviour in it dies at once and the suite says nothing. This shells out
  to a real parser on purpose: every hand-rolled scanner written alongside
  it had a silent coverage gap.

Each helper returns findings; the calling test asserts. None of them
imports pytest, so a suite decides for itself whether a missing ``node``
skips or fails.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "bare_url_for_calls",
    "inline_scripts",
    "javascript_syntax_error",
    "node_available",
    "unknown_data_states",
]

# Two levels of nested parentheses inside the call — ``url_for('a', d=fmt(x))``
# and ``url_for('a', d=f(g(x)))`` — which is as deep as a template argument
# gets; a third level is not matched and is documented as the limit.
_URL_FOR_BARE = re.compile(r"url_for\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)(?!\.path)")
# Any quoted value, however malformed — `"FAILED"`, `"ok "`, `""` are all
# words the kit does not colour and must be reported; the first cut only
# matched values that already looked like a kit word (Codex review of #873).
_DATA_STATE = re.compile(r"""data-state=(?:"([^"]*)"|'([^']*)')""")
_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.S)
_JINJA_OUTPUT = re.compile(r"\{\{.*?\}\}", re.S)
_JINJA_BLOCK = re.compile(r"\{%.*?%\}", re.S)
_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", re.S | re.I)
_NON_JS_TYPE = re.compile(r'type\s*=\s*["\'](?!text/javascript|module)')


def _text(source: Union[Path, str]) -> str:
    return source.read_text() if isinstance(source, Path) else source


def bare_url_for_calls(template: Union[Path, str]) -> list[str]:
    """Return every ``url_for(...)`` in *template* not followed by ``.path``.

    Jinja comments are blanked first, so a call mentioned in a ``{# … #}``
    is not reported. Arguments may nest parentheses two deep
    (``day=fmt(d)``, ``day=f(g(d))``); a third level is not matched.

    Parameters
    ----------
    template : Path or str
        A template file, or its text.

    Returns
    -------
    list of str
        The offending call expressions, in order. Empty means clean.
    """
    text = _JINJA_COMMENT.sub(" ", _text(template))
    return [m.group(0) for m in _URL_FOR_BARE.finditer(text)]


def unknown_data_states(
    template: Union[Path, str], allowed: Iterable[str]
) -> list[str]:
    """Return the literal ``data-state`` values in *template* outside *allowed*.

    Parameters
    ----------
    template : Path or str
        A template or script file, or its text.
    allowed : iterable of str
        The vocabulary — typically ``STATES`` plus ``PANE_STATES`` from
        :mod:`geecs_web_theme`.

    A value rendered by Jinja (``data-state="{{ kit_state[s] }}"``) is not
    a literal and is skipped; the constant it reads from is pinned
    elsewhere. Everything else is judged exactly as written — case,
    whitespace and emptiness included — because the browser matches the
    attribute exactly too.

    Returns
    -------
    list of str
        The unknown values, in order of appearance, duplicates kept so a
        count is a count.
    """
    ok = set(allowed)
    out = []
    for m in _DATA_STATE.finditer(_text(template)):
        value = m.group(1) if m.group(1) is not None else m.group(2)
        if "{{" in value or "{%" in value:
            continue
        if value not in ok:
            out.append(value)
    return out


def inline_scripts(template: Union[Path, str]) -> list[str]:
    """Return the inline JavaScript blocks of an HTML template, Jinja blanked.

    Jinja comments are removed first: ``{# … a <script> body … #}`` mentions
    the tag, and an extractor that does not blank comments starts a match
    inside one and swallows the real script after it. A ``<script>`` with a
    non-JavaScript ``type`` (``application/json``) is a data payload, not
    code, and is skipped — judged by the attribute, never by the body.
    ``{{ … }}`` becomes the string literal ``"jinja"`` and ``{% … %}`` an
    empty statement ``;``, so the result is the script's shape, not one
    render of it, and block tags in statement position — inline
    ``{% if x %}f(){% endif %}`` or on their own lines — still parse. A
    block tag inside an expression (an object literal, an array, the right
    side of an assignment) does not; keep those out of scripts. A ``{{ … }}`` inside a
    double-quoted JavaScript string does not (``""jinja""``); templates
    hand values to scripts through data attributes or single quotes.

    Parameters
    ----------
    template : Path or str
        A template file, or its text.

    Returns
    -------
    list of str
        One entry per non-empty inline script block, ready for
        :func:`javascript_syntax_error`.
    """
    text = _JINJA_COMMENT.sub(" ", _text(template))
    out = []
    for attrs, body in _SCRIPT.findall(text):
        if not body.strip() or _NON_JS_TYPE.search(attrs):
            continue
        out.append(_JINJA_BLOCK.sub(";", _JINJA_OUTPUT.sub('"jinja"', body)))
    return out


def node_available() -> bool:
    """Whether ``node`` is on the PATH, so a suite can skip rather than fail."""
    return shutil.which("node") is not None


def javascript_syntax_error(code: str) -> Optional[str]:
    """Return ``node --check``'s complaint about *code*, or ``None`` if it parses.

    Parameters
    ----------
    code : str
        JavaScript source.

    Returns
    -------
    str or None
        The parser's stderr, stripped, when the code does not parse.

    Raises
    ------
    RuntimeError
        If ``node`` is not available; check :func:`node_available` first.
    """
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is not on the PATH; check node_available() first")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        done = subprocess.run([node, "--check", path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    return None if done.returncode == 0 else done.stderr.strip()
