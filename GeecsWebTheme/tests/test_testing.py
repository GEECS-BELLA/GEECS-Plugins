"""The template-guard helpers find what they exist to find, and only that."""

from __future__ import annotations

import pytest

from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import (
    bare_url_for_calls,
    inline_scripts,
    javascript_syntax_error,
    node_available,
    unknown_data_states,
)


def test_bare_url_for_is_found_and_path_form_is_not() -> None:
    text = (
        "<link href=\"{{ url_for('s', path='a.css').path }}\">"
        "<script src=\"{{ url_for('s', path='a.js') }}\"></script>"
        "{{ url_for('_page', day=fmt(d)) }}"
        "{{ url_for('_page', day=f(g(d))) }}"
        "{# {{ url_for('commented', x=1) }} — a comment is not a call #}"
    )
    assert bare_url_for_calls(text) == [
        "url_for('s', path='a.js')",
        "url_for('_page', day=fmt(d))",
        "url_for('_page', day=f(g(d)))",
    ]


def test_bare_url_for_nesting_limit_is_two_levels() -> None:
    """Three levels of parentheses are NOT matched — the documented limit.

    Pinned so a template that needs deeper nesting fails this test's
    expectation loudly when someone extends the pattern, rather than the
    guard quietly changing what it covers.
    """
    assert bare_url_for_calls("{{ url_for('a', x=f(g(h(1)))) }}") == []


def test_unknown_data_states_against_the_kit_vocabulary() -> None:
    text = (
        '<span data-state="ok"></span><div data-state="denied"></div>'
        "<span data-state='no_data'></span><i data-state=\"ok\"></i>"
    )
    assert unknown_data_states(text, {**STATES, **PANE_STATES}) == ["no_data"]
    assert unknown_data_states(text, STATES) == ["denied", "no_data"]


def test_unknown_data_states_judges_malformed_values_and_skips_jinja() -> None:
    """Case, whitespace and emptiness are unknown words too; Jinja is not a literal.

    The first cut matched only values that already looked like a kit word,
    so ``"FAILED"``, ``"ok "`` and ``""`` — each an uncoloured chip in the
    browser — were never reported (Codex review of #873).
    """
    text = (
        '<i data-state="FAILED"></i><i data-state="ok "></i><i data-state=""></i>'
        "<i data-state='Running'></i>"
        '<i data-state="{{ kit_state[s.status] }}"></i>'
        '<i data-state="{% if x %}ok{% else %}failed{% endif %}"></i>'
        '<i data-state="ok"></i>'
    )
    assert unknown_data_states(text, STATES) == ["FAILED", "ok ", "", "Running"]


def test_inline_scripts_skip_src_json_and_jinja_comments() -> None:
    text = (
        "{# a <script>inside a comment</script> must not start a match #}"
        '<script src="x.js"></script>'
        '<script type="application/json">{"a": 1}</script>'
        "<script>var a = {{ value }}; {% if x %}b(){% endif %}</script>"
        "<script>   </script>"
    )
    blanked = inline_scripts(text)
    assert blanked == ['var a = "jinja"; ;b();']
    if node_available():
        # "ready for javascript_syntax_error" has to be true of the fixture.
        assert javascript_syntax_error(blanked[0]) is None


@pytest.mark.skipif(not node_available(), reason="node not on the PATH")
def test_javascript_syntax_error_reports_only_broken_code() -> None:
    assert javascript_syntax_error("var a = 1; function f() { return a; }") is None
    err = javascript_syntax_error("var a = ;")
    assert err and "SyntaxError" in err


def test_javascript_syntax_error_without_node_is_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("geecs_web_theme.testing.shutil.which", lambda _: None)
    assert not node_available()
    with pytest.raises(RuntimeError):
        javascript_syntax_error("1")


# ------------------------------------------------------------ CSS helpers


def test_classes_used_and_styled_classes_pair_up() -> None:
    from geecs_web_theme.testing import classes_used, styled_classes

    html = '<div class="panel {{ extra }} chip">{# <i class="ghost"> #}</div><b class="lone"></b>'
    css = ".kit .panel{x:1} .chip[data-state=ok], .kit .well > .dim{x:1} @media (a){.tight{x:1}}"
    assert classes_used(html) == {"panel", "chip", "lone"}
    assert styled_classes(css) == {"kit", "panel", "chip", "well", "dim", "tight"}
    assert classes_used(html) - styled_classes(css) == {"lone"}


def test_attribute_selector_values_by_class() -> None:
    from geecs_web_theme.testing import attribute_selector_values

    css = (
        '.kit .chip[data-state="ok"], .kit .chip[data-state="failed"]{x:1}'
        ".kit .dot[data-state='ok']{x:1} .kit .state[data-state=denied]{x:1}"
    )
    assert attribute_selector_values(css, "data-state", on_class="chip") == {
        "ok",
        "failed",
    }
    assert attribute_selector_values(css, "data-state", on_class="dot") == {"ok"}
    assert attribute_selector_values(css, "data-state") == {"ok", "failed", "denied"}


def test_css_helpers_name_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from geecs_web_theme import testing

    real = builtins.__import__

    def no_tinycss2(name, *a, **k):
        if name == "tinycss2":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tinycss2)
    with pytest.raises(RuntimeError, match="geecs-web-theme\\[testing\\]"):
        testing.colour_literals("a{color:#fff}")


def test_token_indirection_map_reads_structure_not_formatting() -> None:
    """One ``var(--x)`` per local token, whatever the spacing; lists split."""
    from geecs_web_theme.testing import token_indirection_map

    css = """
    .tone-ok{--tone:var(--ok)}
    .tone-warn , .other { --tone : var( --warn ) ; color: var(--tone) }
    /* two vars is a computation, not an indirection */
    .tone-bad { --tone: var(--a) var(--b) }
    .tone-none { --tone: red }
    /* one function, but not var(): a computation over a token, not an indirection */
    .tone-mix { --tone: color-mix(in srgb, var(--ok), white) }
    @media (min-width: 40em) { .tone-deep { --tone: var(--deep) } }
    """
    got = token_indirection_map(css)
    assert got == {
        ".tone-ok": {"--tone": "--ok"},
        ".tone-warn": {"--tone": "--warn"},
        ".other": {"--tone": "--warn"},
        ".tone-deep": {"--tone": "--deep"},
    }
