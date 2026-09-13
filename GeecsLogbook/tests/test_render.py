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
