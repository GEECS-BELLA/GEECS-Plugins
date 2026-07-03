"""Handlers that load native GEECS external assets."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from geecs_data_utils.io.images import read_imaq_image


class GeecsPathBackedHandler:
    """Base handler for one native file path recorded as an external asset."""

    def __init__(
        self,
        resource_path: str | Path,
        *,
        root: str | Path | None = None,
        data_key: str | None = None,
        **resource_kwargs: object,
    ) -> None:
        """Resolve a Resource document path into a locally visible path."""
        path = Path(resource_path)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        self.path = path
        self.data_key = data_key
        self.resource_kwargs = resource_kwargs


class GeecsCameraImageHandler(GeecsPathBackedHandler):
    """Load one native GEECS camera image from a resource path.

    This is intentionally a thin I/O adapter: it resolves the file path and
    delegates decoding to :func:`geecs_data_utils.io.images.read_imaq_image`.
    Scientific feature extraction belongs in ImageAnalysis / future analysis
    adapters, not in the external-asset handler.
    """

    def __call__(self) -> np.ndarray:
        """Return the image payload as a NumPy array."""
        return read_imaq_image(self.path)


class GeecsTextArrayHandler(GeecsPathBackedHandler):
    """Load one native text array asset as a NumPy array."""

    def __call__(self) -> np.ndarray:
        """Return whitespace-delimited numeric text as a NumPy array."""
        return np.genfromtxt(self.path, skip_header=_text_header_rows(self.path))


def _text_header_rows(path: Path) -> int:
    """Return one when the first non-empty line is not numeric data."""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            return 0 if _NUMERIC_LINE_RE.match(stripped) else 1
    return 0


_NUMERIC_LINE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
