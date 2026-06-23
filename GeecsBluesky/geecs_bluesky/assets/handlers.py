"""Handlers that load native GEECS external assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from geecs_data_utils.io.images import read_imaq_image


class GeecsCameraImageHandler:
    """Load one native GEECS camera image from a resource path.

    This is intentionally a thin I/O adapter: it resolves the file path and
    delegates decoding to :func:`geecs_data_utils.io.images.read_imaq_image`.
    Scientific feature extraction belongs in ImageAnalysis / future analysis
    adapters, not in the external-asset handler.
    """

    def __init__(
        self,
        resource_path: str | Path,
        *,
        root: str | Path | None = None,
    ) -> None:
        """Create a handler for a resource path.

        Parameters
        ----------
        resource_path:
            Absolute path to the image, or path relative to *root*.
        root:
            Optional storage root used when *resource_path* is relative.
        """
        path = Path(resource_path)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        self.path = path

    def __call__(self) -> np.ndarray:
        """Return the image payload as a NumPy array."""
        return read_imaq_image(self.path)
