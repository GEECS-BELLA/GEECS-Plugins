"""
General camera image analyzer specialized for HIMG data.

This module defines :class:`HIMGWithAveraging`, a thin subclass of
:class:`scan_analysis.analyzers.common.array2D_scan_analysis.Array2DScanAnalyzer`
that customizes *no‑scan* post‑processing to write an average image to a TSV file.
For parameter scans, the post‑processing hooks are intentionally left as no‑ops.

Notes
-----
- This class assumes that, for a no‑scan, a stack of processed images is available
  under ``self.data['images']`` (shape ``(N, H, W)``). That container must be
  populated by the parent workflow or the provided :class:`ImageAnalyzer`.
- When a no‑scan is detected, the average image is written to
  ``<scan_folder>/<device>/average_phase.tsv`` as a tab‑separated, headerless file.
"""

# %% imports
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    pass

import logging

from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalyzer

PRINT_TRACEBACK = True


class HIMGWithAveraging(Array2DScanAnalyzer):
    """
    Minimal specialization of :class:`Array2DScanAnalyzer` for HIMG data.

    This subclass overrides the *no‑scan* post‑processing step to export the
    average processed image as a TSV file (useful for downstream phase/density
    pipelines). Parameter‑scan post‑processing is intentionally left unimplemented.

    Notes
    -----
    - Input images and upstream analysis are still handled by the parent
      :class:`Array2DScanAnalyzer` pipeline (parallel loading, per‑shot analysis,
      optional batch processing).
    - The average is computed with ``np.mean(self.data['images'], axis=0)``. The
      presence and shape of ``self.data['images']`` are assumed and not validated here.
    """

    def _postprocess_noscan(self) -> None:
        """
        Write the average processed image to ``average_phase.tsv`` for a no scan.

        The average is computed across the first axis of ``self.data['images']`` and
        written as a tab‑separated, headerless table in the device's data directory.

        Side Effects
        ------------
        - Create or overwrite ``<scan_folder><device>average_phase.tsv``.

        Warnings
        --------
        Logs a warning and returns if no images are available.

        """
        # Compute the average image from the processed images.
        avg_image = np.mean(self.data["images"], axis=0)

        if avg_image is None:
            logging.warning("No images available to process in _postprocess_noscan.")
            return

        df = pd.DataFrame(avg_image)
        df.to_csv(
            self.path_dict["data_img"] / "average_phase.tsv",
            sep="\t",
            index=False,
            header=False,
        )

    def _postprocess_scan_parallel(self) -> None:
        """
        Post‑processing hook for parameter scans (parallel variant).

        Currently a no‑op. Override in a subclass if HIMG parameter‑scan outputs
        (e.g., per‑bin averages or montages) are desired.
        """
        pass

    def _postprocess_scan_interactive(self) -> None:
        """
        Post‑processing hook for parameter scans (interactive/sequential variant).

        Currently a no‑op. Override in a subclass if interactive plotting or custom
        per‑bin visualizations are needed.
        """
        pass
