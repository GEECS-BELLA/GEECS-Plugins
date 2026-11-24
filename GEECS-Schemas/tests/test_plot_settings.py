"""PlotterSettings: a set keeps what it does not know, and refuses what is out of range."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, PlotterSettings, ScanRef
from geecs_schemas.plot_settings import MAX_CAMERAS, MAX_Y_LISTS


def test_the_kind_is_an_operator_file_in_the_experiment_folder() -> None:
    assert SCHEMA_REGISTRY["plot_settings"] is PlotterSettings
    assert PlotterSettings.tier == "operator"
    assert (
        KIND_STORAGE["plot_settings"].path_pattern()
        == "scanner_configs/experiments/<experiment>/plot_settings/<name>.yaml"
    )


def test_the_empty_document_is_every_default() -> None:
    settings = PlotterSettings.model_validate({})
    assert settings == PlotterSettings()
    assert settings.y_count == 1 and settings.image_zscale == "shared"


def test_unknown_fields_are_kept_at_every_level() -> None:
    """A newer portal's fields survive a load and a dump here (a rollback)."""
    doc = {
        "y": "charge",
        "added_later": {"any": "thing"},
        "extra_scans": [{"day": "2026-11-22", "scan": 4, "label": "ref"}],
        "image_more": [{"device": "UC_Side", "newer_knob": 3}],
    }
    dumped = PlotterSettings.model_validate(doc).model_dump(mode="json")
    assert dumped["added_later"] == {"any": "thing"}
    assert dumped["extra_scans"][0]["label"] == "ref"
    assert dumped["image_more"][0]["newer_knob"] == 3
    assert PlotterSettings.model_validate(dumped).model_dump(mode="json") == dumped


def test_a_choice_this_version_does_not_know_still_loads() -> None:
    settings = PlotterSettings.model_validate(
        {"y_fit": "lorentzian", "tab": "waterfall"}
    )
    assert (settings.y_fit, settings.tab) == ("lorentzian", "waterfall")


def test_days_read_from_yaml_dates_and_strings_alike() -> None:
    by_text = PlotterSettings.model_validate({"day": "2026-11-23", "scan": 12})
    by_date = PlotterSettings.model_validate({"day": date(2026, 11, 23), "scan": 12})
    assert by_text == by_date
    assert ScanRef(day=date(2026, 11, 22), scan=4) in {
        ScanRef(day="2026-11-22", scan=4)
    }


@pytest.mark.parametrize(
    "doc",
    [
        {"schema_version": 2},
        {"y_count": MAX_Y_LISTS + 1},
        {"y_count": 0},
        {"image_more": [{}] * MAX_CAMERAS},
        {"image_tile_max": 10},
        {"image_centre": 0},
        {"extra_scans": [{"day": "2026-11-22", "scan": 0}]},
        {"image_channels": []},
        {"image_more": [{"channels": []}]},
        {"image_channels": [-1]},
        {"image_more": [{"channels": [16]}]},
        {"image_channel": "two"},
    ],
)
def test_out_of_range_values_are_refused(doc: dict) -> None:
    with pytest.raises(ValidationError):
        PlotterSettings.model_validate(doc)


def test_a_set_saved_with_one_channel_reads_as_that_channels_list() -> None:
    settings = PlotterSettings.model_validate(
        {"image_channel": 2, "image_more": [{"channel": 3}, {}]}
    )
    assert settings.image_channels == [2]
    assert [c.channels for c in settings.image_more] == [[3], [0]]
    dumped = settings.model_dump()
    assert "image_channel" not in dumped and "channel" not in dumped["image_more"][0]


def test_a_set_naming_both_keeps_the_list() -> None:
    settings = PlotterSettings.model_validate(
        {"image_channel": 2, "image_channels": [0, 1]}
    )
    assert settings.image_channels == [0, 1]


def test_an_old_channel_written_as_text_still_reads() -> None:
    assert PlotterSettings.model_validate({"image_channel": "2"}).image_channels == [2]
