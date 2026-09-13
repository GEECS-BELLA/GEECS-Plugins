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

    def test_severity_is_preserved_where_it_matters(self) -> None:
        """The mapping may merge colours, never severities.

        ``aborted`` and ``incomplete`` share ``degraded`` on purpose — both
        finished with less than was asked, and the chip still writes which
        one it was. What must never happen is a failure reading as success
        or vice versa.
        """
        assert KIT_STATE["success"] == "ok"
        assert KIT_STATE["failed"] == "failed"
        assert KIT_STATE["aborted"] != "ok"
        assert KIT_STATE["incomplete"] != "ok"
