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
