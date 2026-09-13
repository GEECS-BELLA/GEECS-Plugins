"""The scan-status vocabulary, and how it maps onto the shared kit."""

from __future__ import annotations

from typing import get_args

from geecs_web_theme import STATES

from geecs_logbook.models import KIT_STATE, ScanStatus


class TestKitStateMapping:
    """``KIT_STATE`` is the seam between our statuses and the kit's."""

    def test_every_scan_status_is_mapped(self) -> None:
        """A new ScanStatus must decide its severity, not default to none.

        An unmapped status renders ``data-state=""``, which matches no kit
        rule and still produces a plausible-looking neutral chip — it
        survives review and the browser alike.
        """
        assert set(get_args(ScanStatus)) == set(KIT_STATE)

    def test_every_target_is_a_real_kit_state(self) -> None:
        """Every value names a state ``kit.css`` actually colours."""
        unknown = sorted(set(KIT_STATE.values()) - set(STATES))
        assert not unknown, f"{unknown} are not kit states; kit knows {sorted(STATES)}"

    def test_the_colour_each_status_had_is_the_colour_it_keeps(self) -> None:
        """Adoption must not repaint a status. This is the whole pin.

        Before the kit, ``scanlog.css`` gave each status a colour family,
        and two of them are load-bearing decisions recorded in this
        package's CLAUDE.md under "Status is reported, not inferred":

        - ``incomplete`` (empty ``ScanEndInfo``) was **neutral grey**. It is
          the most common state on the real share — 37 of 49 ScanInfo files
          across four sampled days — and classifying it as a warning
          "painted most of a day amber". It is an absence of information.
        - ``unknown`` (a non-empty ``ScanEndInfo`` we cannot read) was
          **amber**. Something was written and we cannot interpret it, which
          is the genuinely suspicious case.

        So the mapping is deliberately NOT the identity on those two names:
        the logbook's ``unknown`` is the kit's ``degraded``, and the
        logbook's ``incomplete`` is the kit's ``unknown``. A first cut of
        this mapping had them the other way round, silently swapping the two
        severities, and a weaker version of this test passed.
        """
        neutral, amber = {"unknown", "queued"}, {"degraded"}
        assert KIT_STATE["success"] == "ok"
        assert KIT_STATE["failed"] == "failed"
        assert KIT_STATE["aborted"] in amber
        assert KIT_STATE["incomplete"] in neutral, (
            "empty ScanEndInfo is the common case and must stay neutral — "
            "see CLAUDE.md, 'Status is reported, not inferred'"
        )
        assert KIT_STATE["unknown"] in amber, (
            "an unreadable ScanEndInfo is the suspicious case and keeps amber"
        )

    def test_no_failure_ever_reads_as_success(self) -> None:
        """The one merge that must never happen."""
        for status in ("failed", "aborted", "incomplete", "unknown"):
            assert KIT_STATE[status] != "ok"
