"""Rendering modules for scan analysis visualization.

This package provides renderer classes that handle visualization of scan analysis results.
Renderers are used by scan analyzers to create plots, images, and animations from analyzed data.

Configuration classes are also provided for type-safe renderer configuration.
"""

from .base_renderer import BaseRenderer
from .image_2d_renderer import Image2DRenderer
from .line_1d_renderer import Line1DRenderer
from .config import (
    BaseRendererConfig,
    Line1DRendererConfig,
    Image2DRendererConfig,
    RenderContext,
)

__all__ = [
    "BaseRenderer",
    "Image2DRenderer",
    "Line1DRenderer",
    "BaseRendererConfig",
    "Line1DRendererConfig",
    "Image2DRendererConfig",
    "RenderContext",
]
