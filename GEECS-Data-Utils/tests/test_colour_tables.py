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


def test_a_picture_with_its_colour_bar_is_wider_and_keeps_its_pixels() -> None:
    from geecs_data_utils.io.display_png import to_display_png

    y, x = np.indices((60, 80))
    image = (x + y).astype(float)
    plain = np.asarray(Image.open(io.BytesIO(to_display_png(image, plo=0, phi=100))))
    barred = np.asarray(
        Image.open(io.BytesIO(to_display_png(image, plo=0, phi=100, colour_bar=True)))
    )
    assert plain.shape == (60, 80)
    assert barred.shape[1] > 80 and barred.shape[0] >= 60
    top = (barred.shape[0] - 60) // 2
    np.testing.assert_array_equal(barred[top : top + 60, :80, 0], plain)


def test_the_grid_bar_runs_low_to_high_left_to_right() -> None:
    from geecs_data_utils.io.display_png import colour_bar_png

    bar = np.asarray(
        Image.open(io.BytesIO(colour_bar_png("gray", (0.0, 10.0), length=200)))
    )
    row = bar[2, :, 0].astype(int)  # across the ramp, white margins each side
    ramp = row[np.argmin(row) :][:200]
    assert ramp[0] == 0 and ramp[-1] == 255
    assert np.all(np.diff(ramp) >= 0)
