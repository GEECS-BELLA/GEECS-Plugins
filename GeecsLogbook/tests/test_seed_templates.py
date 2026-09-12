"""Seed templates: files in, buttons out, and the closed colour vocabulary."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from geecs_logbook import seed_templates
from geecs_logbook.seed_templates import (
    TONES,
    SeedTemplates,
    load_templates,
    parse_template,
)

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples" / "logbook_templates"
CSS = HERE.parent / "geecs_logbook" / "static" / "scanlog.css"


class TestParse:
    """One file's text becomes one template."""

    def test_header_and_body(self) -> None:
        """Every header key is read; the body is the prefill with its tag."""
        t = parse_template(
            "laser",
            "---\nlabel: Laser\ncolour: ok\nbook: ops\norder: 10\n---\n### Laser\n#laser\n",
        )
        assert (t.label, t.colour, t.book, t.order) == ("Laser", "ok", "ops", 10)
        assert t.body == "### Laser\n#laser\n"
        assert t.offered_in("ops") and not t.offered_in("scans")

    def test_defaults_without_a_header(self) -> None:
        """A bare file is all body, offered in both books, accent-coloured."""
        t = parse_template("shift_handover", "**State:**\n")
        assert t.label == "Shift Handover"
        assert (t.colour, t.book, t.order) == ("accent", "both", 100)
        assert t.body == "**State:**\n"
        assert t.offered_in("ops") and t.offered_in("scans")

    def test_unknown_colour_falls_back_not_through(self, caplog) -> None:
        """A literal or a typo never reaches the page: it becomes accent, logged."""
        t = parse_template("x", "---\ncolour: #ff0000\n---\nbody\n")
        assert t.colour == "accent"
        assert "not a theme tone" in caplog.text

    def test_unknown_book_and_bad_order_are_tolerated(self) -> None:
        """Bad header values cost the button nothing but its metadata."""
        t = parse_template("x", "---\nbook: nope\norder: soon\n---\n")
        assert (t.book, t.order) == ("both", 100)
        assert t.body == ""

    def test_unclosed_header_is_all_body(self) -> None:
        """A lone --- fence is not a header; nothing is silently dropped."""
        t = parse_template("x", "---\nlabel: Not a header\nreal text\n")
        assert t.label == "X"
        assert "real text" in t.body and "label: Not a header" in t.body

    def test_color_spelling_accepted(self) -> None:
        """Both spellings of the colour key are read."""
        assert parse_template("x", "---\ncolor: warn\n---\n").colour == "warn"


class TestLoad:
    """A directory of files, sorted for the button row."""

    def test_examples_load_and_sort(self) -> None:
        """The shipped examples parse; order then label decides the row."""
        found = load_templates(EXAMPLES)
        assert [t.name for t in found] == [
            "laser",
            "result",
            "handover",
            "fault",
            "maintenance",
        ]
        assert {t.colour for t in found} <= set(TONES)
        assert all("#" + t.name in t.body for t in found)

    def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        """No directory, no buttons, no error."""
        assert load_templates(tmp_path / "nope") == []

    def test_reserved_stems_are_refused(self, tmp_path: Path, caplog) -> None:
        """blank.md would put a chip on every hand-typed entry; it is skipped."""
        for name in ("blank", "scan_note", "day_intro", "laser"):
            (tmp_path / f"{name}.md").write_text(f"#{name}\n")
        assert [t.name for t in load_templates(tmp_path)] == ["laser"]
        assert caplog.text.count("is reserved") == 3

    def test_unusable_stems_are_skipped(self, tmp_path: Path, caplog) -> None:
        """A stem that cannot be a template name is skipped with a warning."""
        (tmp_path / "ok.md").write_text("fine\n")
        (tmp_path / ".hidden.md").write_text("no\n")
        (tmp_path / ("x" * 70 + ".md")).write_text("too long\n")
        (tmp_path / "notes.txt").write_text("not markdown\n")
        assert [t.name for t in load_templates(tmp_path)] == ["ok"]
        assert caplog.text.count("not usable") == 2


class TestSeedTemplates:
    """The live set: loaded at start, refreshed off the request path."""

    def test_none_directory_means_nothing_ever(self) -> None:
        """Without a directory the set is empty and stays that way."""
        seeds = SeedTemplates(None)
        assert seeds.current() == [] and seeds.for_book("ops") == []
        assert seeds.by_name() == {}

    def test_first_load_is_synchronous(self, tmp_path: Path) -> None:
        """The process starts with its buttons."""
        (tmp_path / "laser.md").write_text("---\nbook: ops\n---\n#laser\n")
        seeds = SeedTemplates(tmp_path)
        assert [t.name for t in seeds.for_book("ops")] == ["laser"]
        assert seeds.for_book("scans") == []
        assert seeds.by_name()["laser"].body == "#laser\n"

    def test_a_new_file_arrives_after_the_interval_in_the_background(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale set is served as-is while one thread re-reads the directory."""
        (tmp_path / "laser.md").write_text("#laser\n")
        seeds = SeedTemplates(tmp_path)
        (tmp_path / "fault.md").write_text("#fault\n")
        assert [t.name for t in seeds.current()] == ["laser"]  # fresh: no re-read
        monkeypatch.setattr(seed_templates, "REFRESH_INTERVAL_S", 0.0)
        assert [t.name for t in seeds.current()] == ["laser"]  # served stale
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if len(seeds.current()) == 2:
                break
            time.sleep(0.02)
        assert [t.name for t in seeds.current()] == ["fault", "laser"]

    def test_a_failed_refresh_keeps_the_last_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The share going away costs the buttons nothing."""
        (tmp_path / "laser.md").write_text("#laser\n")
        seeds = SeedTemplates(tmp_path)

        def boom(directory: Path) -> list:
            raise OSError("share gone")

        monkeypatch.setattr(seed_templates, "load_templates", boom)
        monkeypatch.setattr(seed_templates, "REFRESH_INTERVAL_S", 0.0)
        seeds.current()
        deadline = time.monotonic() + 5
        while seeds._refreshing and time.monotonic() < deadline:
            time.sleep(0.02)
        assert [t.name for t in seeds.current()] == ["laser"]


def test_page_seeds_carry_the_reserved_names() -> None:
    """The macro hides the chip for exactly the names the loader refuses."""
    from geecs_logbook.seed_templates import RESERVED_NAMES

    page = SeedTemplates(None).for_page("ops")
    assert page.quiet == RESERVED_NAMES and page.buttons == [] and page.prefill == {}


def test_every_tone_has_a_css_rule() -> None:
    """The vocabulary a file may name is exactly what the stylesheet maps.

    A tone without a ``.tone-<name>`` rule would render with no colour at
    all; a rule without a tone would be dead. Both directions are pinned.
    """
    rules = set(
        re.findall(r"\.tone-([\w-]+)\s*\{--tone:var\(--([\w-]+)\)\}", CSS.read_text())
    )
    assert {name for name, _ in rules} == set(TONES)
    assert all(name == token for name, token in rules)
