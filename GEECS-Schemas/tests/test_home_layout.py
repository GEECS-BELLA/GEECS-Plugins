"""HomeLayout: the launcher document names words and stems, never a URL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geecs_schemas import HomeLayout


def test_a_layout_round_trips() -> None:
    doc = {
        "areas": [{"title": "Bench", "screens": ["sim_bench", "control-center"]}],
        "device_pages": [{"title": "Cameras", "families": ["ad_camera_chain"]}],
        "links": ["scanner", "config_editor"],
    }
    layout = HomeLayout.model_validate(doc)
    assert HomeLayout.model_validate(layout.model_dump()) == layout


@pytest.mark.parametrize(
    "doc",
    [
        {"links": ["portal", "portal"]},
        {"links": ["http://elsewhere"]},
        {"areas": [{"title": "A", "screens": ["../x"]}]},
        {"device_pages": [{"title": "P", "families": ["Not-A-Family"]}]},
        {"url": "http://x"},
    ],
)
def test_the_document_refuses_urls_and_bad_names(doc: dict) -> None:
    with pytest.raises(ValidationError):
        HomeLayout.model_validate(doc)
