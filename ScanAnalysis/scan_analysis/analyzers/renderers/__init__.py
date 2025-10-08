"""Rendering modules for scan analysis visualization.

This package provides renderer classes that handle visualization of scan analysis results.
Renderers are used by scan analyzers to create plots, images, and animations from analyzed data.
"""

from .base_renderer import BaseRenderer
from .image_2d_renderer import Image2DRenderer

__all__ = ["BaseRenderer", "Image2DRenderer"]
