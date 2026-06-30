"""Tests for the local fake-hardware RunEngine sandbox."""

from __future__ import annotations

import pytest

from geecs_bluesky.testing import run_fake_step_scan

pytestmark = pytest.mark.fake_server


def test_run_fake_step_scan_collects_expected_documents() -> None:
    """The sandbox should run a complete RunEngine scan without lab services."""
    result = run_fake_step_scan(positions=(0.0, 1.0), shots_per_step=2)

    assert result.start_doc["sandbox"] is True
    assert result.start_doc["plan_name"] == "geecs_step_scan"
    assert result.stop_doc["exit_status"] == "success"

    events = result.event_docs
    assert len(events) == 4

    first_data = events[0]["data"]
    assert "sandbox_motor-position" in first_data
    assert "sandbox_detector-signal" in first_data
    assert "sandbox_detector-shot_id" in first_data
    assert first_data["bin_number"] == 1
    assert first_data["shot_index_in_bin"] == 1

    assert [event["data"]["bin_number"] for event in events] == [1, 1, 2, 2]
    assert [event["data"]["scan_event_index"] for event in events] == [1, 2, 3, 4]
