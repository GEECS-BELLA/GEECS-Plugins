"""The body renderer: the one thing that reads ``body_md``, and the safety boundary.

An agent writes into the same field a human does, so what comes out of
here is what the sanitiser lets through — pinned case by case.
"""

from __future__ import annotations

from geecs_logbook.render import render_markdown


class TestSanitiser:
    """Nothing executable survives, whoever wrote it."""

    def test_raw_html_is_escaped_not_rendered(self) -> None:
        """Inline HTML in markdown is text, never markup."""
        out = render_markdown("<script>alert(1)</script> and <b>bold</b>")
        assert "<script" not in out and "<b>" not in out
        assert "&lt;script&gt;" in out

    def test_javascript_and_data_links_are_dropped(self) -> None:
        """Only http(s)/mailto/relative links keep their href."""
        out = render_markdown(
            "[x](javascript:alert(1)) [y](data:text/html,hi) [z](https://a.b/)"
        )
        # markdown-it refuses the link outright (the text stays literal);
        # nothing ever reaches an href.
        assert 'href="javascript' not in out and 'href="data' not in out
        assert out.count("<a ") == 1 and 'href="https://a.b/"' in out
        assert 'rel="noopener noreferrer"' in out

    def test_image_event_handlers_never_emerge(self) -> None:
        """An image with an onerror handler loses the handler."""
        out = render_markdown(
            '<img src=x onerror="alert(1)">\n\n![ok](attachments/abc/a.png)'
        )
        # The raw tag is text (escaped); the only real <img> is the markdown one.
        assert out.count("<img") == 1 and "&lt;img" in out
        assert 'src="attachments/abc/a.png"' in out

    def test_attachment_base_cannot_inject_attributes(self) -> None:
        """A hostile base is escaped before it lands in src=."""
        out = render_markdown(
            "![p](attachments/abc/a.png)", attachment_base='/x" onload="e'
        )
        assert '" onload="' not in out  # the quote never closes the attribute
        assert "&quot;" in out


class TestMarkdownFeatures:
    """What the composer advertises actually renders."""

    def test_tables_strikethrough_and_task_lists(self) -> None:
        """Tables, ~~strike~~ and task-list checkboxes survive sanitising."""
        out = render_markdown(
            "| a | b |\n|---|---|\n| 1 | 2 |\n\n~~gone~~\n\n- [ ] open\n- [x] done\n"
        )
        assert "<table>" in out and "<s>gone</s>" in out
        assert out.count('type="checkbox"') == 2
        assert "checked" in out and "disabled" in out

    def test_callouts(self) -> None:
        """A marker-led blockquote becomes a labelled callout."""
        out = render_markdown("> [!WARNING]\n> jet pressure drifting\n")
        assert '<div class="callout callout-warning">' in out
        assert '<div class="callout-label">warning</div>' in out
        assert "jet pressure drifting" in out
        assert "[!WARNING]" not in out

    def test_plain_blockquote_is_left_alone(self) -> None:
        """No marker, no callout."""
        out = render_markdown("> just a quote\n")
        assert "<blockquote>" in out and "callout" not in out

    def test_attachment_links_rewritten_only_with_a_base(self) -> None:
        """Relative on disk; served path on the page."""
        body = "![p](attachments/abc/a.png) [pdf](attachments/abc/n.pdf)"
        offline = render_markdown(body)
        assert offline.count('="attachments/abc/') == 2
        # The base REPLACES the ``attachments/`` prefix: it names the serving
        # route, whose own path already carries the day and the scope.
        served = render_markdown(
            body, attachment_base="/log/attachments/2026-09-11/Scan005/"
        )
        assert served.count('="/log/attachments/2026-09-11/Scan005/abc/') == 2
        assert "attachments/abc" not in served.replace("/Scan005/abc", "")

    def test_empty_body_is_empty(self) -> None:
        """No text, no markup."""
        assert render_markdown("") == ""


class TestImageGrid:
    """Consecutive images become one grid; a lone image does not."""

    def test_two_or_more_images_in_a_row_grid(self) -> None:
        """Pasted screenshots one after another render side by side."""
        out = render_markdown(
            "![a](attachments/x/a.png)\n\n![b](attachments/x/b.png)\n\n![c](attachments/x/c.png)\n"
        )
        assert out.count('<div class="figgrid">') == 1
        assert out.count("<figure><img") == 3
        assert "<p><img" not in out

    def test_a_single_image_stays_a_paragraph(self) -> None:
        """One figure is a figure, not a grid of one."""
        out = render_markdown("text\n\n![a](attachments/x/a.png)\n\nmore")
        assert "figgrid" not in out and "<p><img" in out

    def test_a_paragraph_breaks_the_run(self) -> None:
        """Images separated by prose are two runs, not one."""
        out = render_markdown(
            "![a](x/a.png)\n\n![b](x/b.png)\n\ncaption\n\n![c](x/c.png)\n\n![d](x/d.png)"
        )
        assert out.count('<div class="figgrid">') == 2

    def test_grid_survives_the_link_rewrite(self) -> None:
        """The serving base is applied inside the grid too."""
        out = render_markdown(
            "![a](attachments/x/a.png)\n\n![b](attachments/x/b.png)",
            attachment_base="/log/attachments",
        )
        assert out.count('src="/log/attachments/x/') == 2 and "figgrid" in out


class TestMath:
    """Equations: parsed and marked here, typeset in the browser by math.js."""

    def test_inline_math_is_marked_and_escaped(self) -> None:
        """The TeX survives as text, `<` included, inside the inline mark."""
        out = render_markdown("energy $E = \\gamma m c^2 < 1$ here")
        assert '<span class="math-inline">E = \\gamma m c^2 &lt; 1</span>' in out
        assert "$" not in out

    def test_a_display_block(self) -> None:
        """``$$`` on lines of its own is a block of its own."""
        out = render_markdown("$$\n\\int_0^1 x\\,dx\n$$\n\nafter")
        assert '<div class="math-display">\\int_0^1 x\\,dx</div>' in out
        assert "<p>after</p>" in out

    def test_double_dollar_inside_a_paragraph_is_a_span(self) -> None:
        """Display mode still, but a <div> inside <p> is not HTML."""
        out = render_markdown("see $$a<b$$ there")
        assert '<p>see <span class="math-display">a&lt;b</span> there</p>' in out

    def test_money_is_not_math(self) -> None:
        """Pandoc's rules: no space inside the dollars, no digit against them."""
        out = render_markdown("cost $5 and $10 each; $ x $ spaced; 1$x$2")
        assert "math-" not in out
        assert "$5 and $10 each; $ x $ spaced; 1$x$2" in out

    def test_an_escaped_dollar_is_a_dollar(self) -> None:
        """``\\$`` is how a note writes a price next to an equation."""
        out = render_markdown("\\$5 flat, $E$ ok")
        assert "$5 flat" in out and '<span class="math-inline">E</span>' in out

    def test_math_in_code_stays_code(self) -> None:
        """A code span is literal, dollars included."""
        out = render_markdown("`$x$` in code")
        assert "<code>$x$</code>" in out and "math-" not in out

    def test_markup_inside_math_never_emerges(self) -> None:
        """TeX is text to the page; a tag written between the dollars stays so."""
        out = render_markdown('$<img src=x onerror="alert(1)">$')
        assert "<img" not in out and "&lt;img" in out
        assert 'class="math-inline"' in out

    def test_an_authored_mark_is_text(self) -> None:
        """Only the renderer writes the marks; typing one gets escaped HTML."""
        out = render_markdown('<span class="math-inline">\\rule{9em}{9em}</span>')
        assert '<span class="math-inline"' not in out and "&lt;span" in out

    def test_the_marks_are_the_only_classes_the_sanitiser_admits(self) -> None:
        """The second lock, exercised directly: any other class is stripped."""
        import nh3

        from geecs_logbook.render import _ATTRIBUTES, _CLASSES, _TAGS

        out = nh3.clean(
            '<span class="math-inline evil">a</span><div class="math-inline">b</div>'
            '<p class="callout">c</p>',
            tags=_TAGS,
            attributes=_ATTRIBUTES,
            allowed_classes=_CLASSES,
        )
        assert '<span class="math-inline">a</span>' in out
        assert 'math-inline">b' not in out  # a div carries display marks only
        assert "callout" not in out
