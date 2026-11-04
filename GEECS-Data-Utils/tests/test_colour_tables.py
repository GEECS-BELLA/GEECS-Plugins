"""GEECSplotter's LabVIEW colour tables, by name through the PNG renderer."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from geecs_data_utils.io import colour_tables
from geecs_data_utils.io.display_png import safe_cmap, to_display_png

pytest.importorskip("matplotlib")


@pytest.mark.parametrize("name", sorted(colour_tables.TABLES))
def test_each_table_is_whole_colours(name: str) -> None:
    text = colour_tables.TABLES[name]
    assert len(text) % 6 == 0 and len(text) // 6 > 200
    int(text, 16)  # hex throughout


def test_the_ends_are_the_labview_tables() -> None:
    """enhance low runs white to red; white blue yellow white to pale yellow."""
    low = colour_tables.colours("enhance_low")
    assert low[0] == (1.0, 1.0, 1.0) and low[-1] == (1.0, 0.0, 0.0)
    wby = colour_tables.colours("white_blue_yellow")
    assert wby[0] == pytest.approx((0xFA / 255, 0xFA / 255, 0xFA / 255))
    assert wby[-1] == pytest.approx((1.0, 0xFD / 255, 0xC5 / 255))


@pytest.mark.parametrize("name", sorted(colour_tables.TABLES))
def test_the_renderer_knows_them_by_name(name: str) -> None:
    assert safe_cmap(name) == name
    ramp = np.linspace(0.0, 1.0, 64).reshape(1, 64)
    drawn = np.asarray(
        Image.open(io.BytesIO(to_display_png(ramp, cmap=name, plo=0, phi=100)))
    )
    assert drawn.shape == (1, 64, 3)
    first = tuple(round(c * 255) for c in colour_tables.colours(name)[0])
    assert tuple(drawn[0, 0]) == first


def test_registering_twice_is_harmless() -> None:
    colour_tables.register()
    colour_tables.register()
    assert safe_cmap("enhance_low") == "enhance_low"
