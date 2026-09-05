"""``RendererOptions`` (GEECS-Schemas) must be accepted, field for field, by the renderer configs here.

The typed ``scan.renderer`` section lives in the schema package; the
renderers' own config models (with their defaults) stay in ScanAnalysis.
This pins the coupling so a field added on one side cannot silently be
dropped on the other.
"""

from geecs_schemas.analysis import RendererOptions

from scan_analysis.analyzers.renderers.config import (
    Image2DRendererConfig,
    Line1DRendererConfig,
)

OPTIONS = set(RendererOptions.model_fields)


def test_line_renderer_accepts_every_non_camera_option():
    assert OPTIONS - RendererOptions.CAMERA_ONLY <= set(
        Line1DRendererConfig.model_fields
    )


def test_image_renderer_accepts_every_non_line_option():
    assert OPTIONS - RendererOptions.LINE_ONLY <= set(
        Image2DRendererConfig.model_fields
    )


def test_renderer_config_fields_all_exist_on_the_options():
    assert set(Line1DRendererConfig.model_fields) <= OPTIONS
    assert set(Image2DRendererConfig.model_fields) <= OPTIONS


def test_unset_options_leave_renderer_defaults_alone():
    kwargs = RendererOptions(cmap="viridis").as_kwargs()
    line = Line1DRendererConfig(**kwargs)
    image = Image2DRendererConfig(**kwargs)
    assert line.cmap == image.cmap == "viridis"
    assert line.colormap_mode == Line1DRendererConfig().colormap_mode
    assert image.figsize == Image2DRendererConfig().figsize
