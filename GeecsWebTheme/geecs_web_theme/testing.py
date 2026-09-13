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

The CSS helpers below the HTML ones read stylesheets through **tinycss2**,
a real CSS parser, instead of regular expressions: comments, ``@media``
blocks, nested braces and quoted strings are already structure by the time
a helper looks at them, so a check reads like the rule it enforces and a
mistake fails loudly rather than matching nothing. tinycss2 is imported
lazily — the HTML helpers above stay standard-library only — and is
declared under the ``testing`` extra.

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
from collections.abc import Iterable, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "attribute_selector_values",
    "bare_url_for_calls",
    "classes_used",
    "colour_literals",
    "css_rules",
    "defined_tokens",
    "html_style_sources",
    "inline_scripts",
    "javascript_syntax_error",
    "js_colour_literals",
    "node_available",
    "referenced_tokens",
    "rule_selectors",
    "selector_parts",
    "styled_classes",
    "token_indirections",
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


# ----------------------------------------------------------------- CSS
# Everything below reads CSS through tinycss2. The HTML side uses the
# standard library's HTMLParser for the same reason: structure first,
# then a rule expressed over the structure.


def _tinycss2() -> Any:  # the module; typed Any because it is optional
    try:
        import tinycss2
    except ImportError as exc:  # pragma: no cover - the extra is missing
        raise RuntimeError(
            "the CSS guards need tinycss2: install geecs-web-theme[testing]"
        ) from exc
    return tinycss2


#: Named colours people actually reach for as values. The full CSS list would
#: flag words like ``tan`` or ``plum`` in an ident that is not a colour at all.
NAMED_COLOURS = frozenset(
    "white black red green blue gray grey orange yellow purple pink cyan magenta "
    "navy teal olive maroon silver lime aqua fuchsia tomato lightgray lightgrey "
    "darkgray darkgrey whitesmoke gold crimson".split()
)
_COLOUR_FUNCTIONS = frozenset({"rgb", "rgba", "hsl", "hsla"})
#: A hex colour inside a URL or data: string, percent-encoded or not.
_ENCODED_HEX = re.compile(r"(?:%23|#)[0-9a-fA-F]{3,8}\b")
_HEX_LENGTHS = frozenset({3, 4, 6, 8})
_HEX = re.compile(r"^[0-9a-fA-F]+$")


def css_rules(css: str) -> list[tuple[str, list[tuple[str, list[Any]]]]]:
    """Return every qualified rule as ``(selector, declarations)``, flattened.

    ``@media`` and other conditional at-rules are descended into at any
    depth (``@media { @supports { .a{} } }`` yields ``.a``), so a rule
    inside one counts like any other; ``@keyframes`` bodies are skipped
    (their ``0%`` stops are not selectors); an at-rule whose body is only
    declarations (``@font-face``) yields nothing. **Native nesting** is
    flattened too: a rule nested inside ``.a{…}`` is reported with the
    selector ``.a &:hover`` — parent first, so a scoping check that reads
    the front of the selector sees the parent's scope; comma lists on
    either side cross-multiply, and declarations sitting directly inside a
    nested at-rule (``.a{ @media (x){ color:red } }``) belong to ``.a``.
    Comments are dropped by the parser. Each declaration is
    ``(name, value_tokens)``.

    Parameters
    ----------
    css : str
        Stylesheet text.

    Returns
    -------
    list of (str, list of (str, list))
        The serialized selector and its declarations, in source order.
    """
    tc = _tinycss2()
    out: list[tuple[str, list[tuple[str, list[Any]]]]] = []

    def compose(parent: str, own: str) -> str:
        """Every parent part × every own part: ``.a, .b`` under ``.k`` → ``.k .a, .k .b``."""
        if not parent:
            return own
        return ", ".join(
            f"{p} {o}" for p in selector_parts(parent) for o in selector_parts(own)
        )

    def block(content: Any, selector: str) -> None:
        """Walk a ``{}`` body that may hold declarations and nested rules."""
        decls: list[tuple[str, list[Any]]] = []
        nested: list[Any] = []
        for item in tc.parse_blocks_contents(content, skip_whitespace=True):
            if item.type == "declaration":
                decls.append((item.name, item.value))
            elif item.type in ("qualified-rule", "at-rule"):
                nested.append(item)
        out.append((selector, decls))
        walk(nested, selector)

    def walk(rules: Sequence[Any], parent: str = "") -> None:
        for rule in rules:
            if rule.type == "qualified-rule":
                block(rule.content, compose(parent, tc.serialize(rule.prelude).strip()))
            elif rule.type == "at-rule" and rule.content is not None:
                if rule.lower_at_keyword == "keyframes":
                    continue
                if parent:
                    # Nested inside a rule: `.a{ @media (x){ color:red; &:hover{} } }`
                    # — bare declarations still belong to `.a`.
                    block(rule.content, parent)
                    continue
                inner = tc.parse_rule_list(rule.content, skip_whitespace=True)
                if any(r.type in ("qualified-rule", "at-rule") for r in inner):
                    walk(inner, parent)

    walk(tc.parse_stylesheet(css, skip_comments=True, skip_whitespace=True))
    return out


def selector_parts(selector: str) -> list[str]:
    """Split a selector list on its TOP-LEVEL commas only.

    ``.kit :is(.panel, .well)`` is one selector; a plain ``split(",")``
    would cut it inside the parentheses. The tokenizer hands functions and
    blocks back as single tokens, so a comma seen here is a real separator.
    """
    tc = _tinycss2()
    parts: list[str] = []
    current: list[Any] = []
    for tok in tc.parse_component_value_list(selector):
        if tok.type == "literal" and tok.value == ",":
            parts.append(tc.serialize(current).strip())
            current = []
        else:
            current.append(tok)
    parts.append(tc.serialize(current).strip())
    return [p for p in parts if p]


def rule_selectors(css: str) -> list[str]:
    """Return every selector in *css*, one entry per top-level comma-separated part."""
    return [part for selector, _ in css_rules(css) for part in selector_parts(selector)]


def _is_root_selector(selector: str) -> bool:
    """Whether every part of *selector* is ``:root`` plus attribute filters only.

    ``:root[data-theme="x"]`` qualifies; ``:root .foo`` does not — a token
    defined on a descendant is a component's private literal, not a palette
    entry.
    """
    tc = _tinycss2()
    for part in selector_parts(selector):
        tokens = [
            t for t in tc.parse_component_value_list(part) if t.type != "whitespace"
        ]
        if len(tokens) < 2 or tokens[0].type != "literal" or tokens[0].value != ":":
            return False
        if tokens[1].type != "ident" or tokens[1].lower_value != "root":
            return False
        if any(t.type != "[] block" for t in tokens[2:]):
            return False
        # whitespace between :root and a block means a descendant selector
        if re.search(r":root\s+\S", part.strip()):
            return False
    return True


def _colour_tokens(
    tokens: Iterable[Any], *, in_var_fallback: bool = False
) -> list[str]:
    """Return the colour literals among *tokens*, descending into functions.

    A ``var(--x)`` use is not a literal, but a ``var(--x, #fff)`` *fallback*
    is, so the arguments after the first comma of a ``var()`` are judged.
    Shadows and scrims over black are ground-free and exempt — see
    :func:`colour_literals`.
    """
    found: list[str] = []
    for tok in tokens:
        if tok.type == "hash":
            if len(tok.value) in _HEX_LENGTHS and _HEX.match(tok.value):
                found.append("#" + tok.value)
        elif tok.type == "ident":
            if tok.lower_value in NAMED_COLOURS:
                found.append(tok.value)
        elif tok.type == "function":
            name = tok.lower_name
            if name == "var":
                args = list(tok.arguments)
                comma = next(
                    (
                        i
                        for i, a in enumerate(args)
                        if a.type == "literal" and a.value == ","
                    ),
                    None,
                )
                if comma is not None:
                    found.extend(
                        _colour_tokens(args[comma + 1 :], in_var_fallback=True)
                    )
            elif name in _COLOUR_FUNCTIONS:
                text = _tinycss2().serialize([tok])
                if not _ground_free(text):
                    found.append(text)
            else:
                found.extend(
                    _colour_tokens(tok.arguments, in_var_fallback=in_var_fallback)
                )
        elif tok.type == "url":
            # an unquoted data: URL carrying SVG with fill='%23ff00ff'
            if _ENCODED_HEX.search(tok.value):
                found.append("url(…%23…)")
        elif tok.type == "string":
            # the quoted form of the same
            if _ENCODED_HEX.search(tok.value):
                found.append("'…%23…'")
        elif tok.type == "error" and getattr(tok, "kind", "") == "bad-url":
            # url(data:…fill='%23fff'…) unquoted, with a quote inside, is not
            # parseable CSS at all; the browser drops it and so would a check
            # that skipped errors. Report it — quoting the URL fixes both.
            found.append("bad-url (unparseable url(); quote it)")
        elif tok.type in ("() block", "[] block", "{} block"):
            found.extend(_colour_tokens(tok.content, in_var_fallback=in_var_fallback))
    return found


_GROUND_FREE = re.compile(r"^rgba\(\s*0\s*,\s*0\s*,\s*0\s*,")


def _ground_free(text: str) -> bool:
    """Whether a colour function is opacity over plain black.

    Shadows and scrims read on every ground and have no token to take; a
    coloured glow — or a tinted black — is a colour and must go through a
    token (the palettes' own ``--shadow`` definitions live in ``:root`` and
    are exempt on that ground).
    """
    return bool(_GROUND_FREE.match(text.replace(" ", "")))


def _allowed_part(part: str, marks: tuple[str, ...]) -> bool:
    """Whether one comma-part of a selector is covered by an allowlist entry.

    An entry matches the part exactly, or as a prefix followed by a
    descendant, a pseudo-class or a further class (``img.plot`` covers
    ``img.plot:hover`` and ``img.plot .x``) — never by extending the class
    name itself, so it does not cover ``img.plotwrap`` or ``img.plot-x``.
    """
    for mark in marks:
        if part == mark:
            return True
        if part.startswith(mark) and part[len(mark)] in " :.>[":
            return True
    return False


def colour_literals(css: str, *, allowed: Iterable[str] = ()) -> list[str]:
    """Return every colour literal in *css* outside a ``:root`` token definition.

    Rules, in the order they are applied:

    - a ``--token: value`` declaration inside a ``:root…`` rule is the one
      legitimate literal and is skipped;
    - a ``--local: #fff`` inside any other rule is a hidden literal;
    - a ``var(--x)`` use is not a literal; its fallback argument is;
    - ``rgba()`` over black or ink-black (shadows, scrims) is exempt;
    - a rule every one of whose selector parts is covered by an *allowed*
      entry is skipped — the caller's allowlist, each entry with a stated
      reason; ``img.plot`` covers ``img.plot`` and ``img.plot .x``, not
      ``img.plotwrap``, not ``img.plot-x``, and not the other half of
      ``img.plot, .other``.

    Parameters
    ----------
    css : str
        Stylesheet text.
    allowed : iterable of str, optional
        Selectors (exact, or a prefix at a class boundary) whose rules may
        carry a literal.

    Returns
    -------
    list of str
        ``"selector: property: literal"`` per finding, in source order.
    """
    marks = tuple(allowed)
    out: list[str] = []
    for selector, decls in css_rules(css):
        if marks and all(
            _allowed_part(part, marks) for part in selector_parts(selector)
        ):
            continue
        root = _is_root_selector(selector)
        for name, value in decls:
            if root and name.startswith("--"):
                continue
            for lit in _colour_tokens(value):
                out.append(f"{selector}: {name}: {lit}")
    return out


class _StyleSources(HTMLParser):
    """Collect the CSS an HTML document carries outside a stylesheet file."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.css: list[str] = []
        self.scripts: list[str] = []
        self._in: Optional[str] = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in ("style", "script"):
            self._in = tag
            self._buf = []
        for name, value in attrs:
            if value is None:
                continue
            # A Jinja fragment is not a literal, but the rest of the value
            # still is: style="color:#ff00ff; width:{{ w }}px" carries one.
            residue = _JINJA_BLOCK.sub(" ", _JINJA_OUTPUT.sub("jinja", value))
            if name == "style":
                self.css.append(f"x{{{residue}}}")
            elif name in ("fill", "stroke"):
                self.css.append(f"x{{{name}:{residue}}}")

    def handle_endtag(self, tag: str) -> None:
        if tag == self._in:
            body = "".join(self._buf)
            (self.css if tag == "style" else self.scripts).append(body)
            self._in = None

    def handle_data(self, data: str) -> None:
        if self._in:
            self._buf.append(data)


def html_style_sources(html: Union[Path, str]) -> tuple[list[str], list[str]]:
    """Return ``(css_chunks, script_bodies)`` an HTML template carries.

    CSS is the contents of every ``<style>`` element plus each ``style="…"``
    and SVG ``fill``/``stroke`` attribute, wrapped as a rule so
    :func:`colour_literals` can judge it. Script bodies are returned
    separately for :func:`js_colour_literals`. Jinja comments are removed
    first and a Jinja-rendered attribute value is not a literal.
    """
    text = _JINJA_COMMENT.sub(" ", _text(html))
    parser = _StyleSources()
    parser.feed(text)
    parser.close()
    return parser.css, parser.scripts


#: Double-quoted, single-quoted and template-literal strings. A nested
#: template literal inside a ``${…}`` interpolation ends the outer match
#: early — accepted; the guarded pages build HTML with backticks and put
#: colours nowhere near an interpolation.
_JS_STRING = re.compile(
    r'"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'|`((?:[^`\\]|\\.)*)`'
)


def js_colour_literals(js: str) -> list[str]:
    """Return the string literals in *js* whose whole value is a CSS colour.

    ``el.style.color = "orange"``, ``cssText = "color:#ff00ff"``,
    ``setProperty("color", "#fff")`` and a template-literal HTML builder
    all put a colour in a string; so does a Plotly layout. Each string literal is parsed as a CSS value and
    reported when it contains a colour literal — a ``#now`` element id is
    not one (``now`` is not hex), a ``"#abc"`` would be. Comments are not
    special-cased: a colour in a comment is a string only if quoted.
    """
    tc = _tinycss2()
    out: list[str] = []
    for m in _JS_STRING.finditer(js):
        s = next(g for g in m.groups() if g is not None)
        if not s or "{{" in s:
            continue
        # A bare value, or a declaration list ("color:#fff;font:x").
        tokens = tc.parse_component_value_list(s, skip_comments=True)
        if _colour_tokens(tokens):
            out.append(s)
    return out


#: Token references that are STRING patterns rather than CSS structure — a
#: script reading a token, a server-emitted sentinel, and CSS text a script
#: builds (``cssText = "background:var(--surface-2)"``), which the old
#: raw-text guard saw and a parser over stylesheet files alone does not.
_TOKEN_REF_TEXT = re.compile(r'getPropertyValue\(\s*["\'`](--[\w-]+)|\$tok:(--[\w-]+)')
#: Only for text that is NOT a stylesheet: in CSS the parser finds var()
#: uses, and a token named in a CSS comment is not a reference.
_TOKEN_REF_JS_VAR = re.compile(r"var\(\s*(--[\w-]+)")


def referenced_tokens(source: Union[Path, str], *, css: bool = True) -> set[str]:
    """Return every ``--token`` a file references.

    In CSS that is every ``var(--x)`` (found through the parser, at any
    nesting). In scripts and Python it is ``getPropertyValue("--x")``, the
    ``$tok:--x`` sentinels a server emits for the page to resolve, and any
    ``var(--x)`` inside CSS text a script builds — string patterns, not
    CSS structure, so a small regex is the honest tool.
    """
    text = _text(source)
    names: set[str] = {
        g for m in _TOKEN_REF_TEXT.finditer(text) for g in m.groups() if g
    }
    if not css:
        names |= set(_TOKEN_REF_JS_VAR.findall(text))
    if css:

        def walk(tokens: Iterable[Any]) -> None:
            for tok in tokens:
                if tok.type == "function":
                    if tok.lower_name == "var":
                        first = next(
                            (a for a in tok.arguments if a.type == "ident"), None
                        )
                        if first is not None:
                            names.add(first.value)
                    walk(tok.arguments)
                elif tok.type in ("() block", "[] block", "{} block"):
                    walk(tok.content)

        for _, decls in css_rules(text):
            for _, value in decls:
                walk(value)
    return names


def token_indirections(css: str) -> set[str]:
    """Return the ``--local`` tokens whose value is exactly one ``var(--x)``.

    ``.tone-ok{--tone:var(--ok)}`` is a surface's own indirection over a
    theme token, so ``var(--tone)`` elsewhere is not an undefined reference.
    """
    out: set[str] = set()
    for _, decls in css_rules(css):
        for name, value in decls:
            if not name.startswith("--"):
                continue
            real = [v for v in value if v.type != "whitespace"]
            if (
                len(real) == 1
                and real[0].type == "function"
                and real[0].lower_name == "var"
            ):
                out.add(name)
    return out


def defined_tokens(css: str) -> dict[str, set[str]]:
    """Return ``{selector: {--token, …}}`` for every rule that defines tokens."""
    out: dict[str, set[str]] = {}
    for selector, decls in css_rules(css):
        names = {n for n, _ in decls if n.startswith("--")}
        if names:
            out.setdefault(selector, set()).update(names)
    return out


def styled_classes(*css_texts: str) -> set[str]:
    """Return every class name any selector in the given stylesheets mentions.

    Descends into ``:is()`` / ``:where()`` and attribute blocks, so
    ``.kit :is(.panel, .well)`` styles ``panel`` and ``well``; it does NOT
    descend into ``:not()`` / ``:has()``, which name a class without
    painting it.
    """
    tc = _tinycss2()
    out: set[str] = set()

    def walk(tokens: Iterable[Any]) -> None:
        prev = None
        for tok in tokens:
            if (
                tok.type == "ident"
                and prev is not None
                and prev.type == "literal"
                and prev.value == "."
            ):
                out.add(tok.value)
            elif tok.type == "function":
                # :is()/:where() style their arguments; :not()/:has() name a
                # class WITHOUT styling it — `.a:not(.dim)` paints no `.dim`.
                if tok.lower_name not in ("not", "has"):
                    walk(tok.arguments)
            elif tok.type in ("() block", "[] block", "{} block"):
                walk(tok.content)
            prev = tok

    for css in css_texts:
        for selector, _ in css_rules(css):
            walk(tc.parse_component_value_list(selector))
    return out


def attribute_selector_values(
    css: str, attribute: str, *, on_class: Optional[str] = None
) -> set[str]:
    """Return the values ``[attribute="…"]`` takes in *css*, optionally on one class.

    ``attribute_selector_values(kit, "data-state", on_class="chip")`` is the
    set of states the chip colours — pinned against ``STATES`` both ways.
    """
    tc = _tinycss2()
    out: set[str] = set()
    for part in rule_selectors(css):
        tokens = tc.parse_component_value_list(part)
        prev = None
        for tok in tokens:
            if tok.type == "[] block":
                inner = [t for t in tok.content if t.type != "whitespace"]
                if (
                    len(inner) >= 3
                    and inner[0].type == "ident"
                    and inner[0].value == attribute
                    and inner[-1].type in ("string", "ident")
                ):
                    if on_class is None or (
                        prev is not None
                        and prev.type == "ident"
                        and prev.value == on_class
                    ):
                        out.add(inner[-1].value)
            prev = tok
    return out


class _ClassCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.classes: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        for name, value in attrs:
            if name == "class" and value:
                # A Jinja-rendered fragment is not a literal class name.
                literal = _JINJA_BLOCK.sub(" ", _JINJA_OUTPUT.sub(" ", value))
                self.classes.update(literal.split())


def classes_used(html: Union[Path, str]) -> set[str]:
    """Return every class name an HTML document puts on an element.

    Jinja-rendered class fragments are skipped; with :func:`styled_classes`
    this is the "every class on the page is styled by something" guard.
    """
    parser = _ClassCollector()
    parser.feed(_JINJA_COMMENT.sub(" ", _text(html)))
    parser.close()
    return parser.classes
