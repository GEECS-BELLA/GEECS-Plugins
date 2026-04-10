"""Line stitcher for composite spectrometers.

Stitches data files from several devices (e.g., magcam1, magcam2, ...) that
each cover a different portion of a shared physical axis (e.g., energy in MeV).
The master device is configured normally via a line_config YAML; sibling device
files are resolved from the master file path using the GEECS naming convention::

    <scan_dir>/<device>/<device>_NNN.tsv

All files are concatenated and sorted by x before being handed to the standard
processing pipeline (ROI, interpolation, statistics).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from image_analysis.data_1d_utils import read_1d_data
from image_analysis.offline_analyzers.line_analyzer import LineAnalyzer
from image_analysis.types import Array1D

logger = logging.getLogger(__name__)


class LineStitcher(LineAnalyzer):
    """Line analyzer that stitches data files from multiple devices.

    Parameters
    ----------
    line_config_name : str
        Name of the line configuration for the master device.
    sibling_devices : list of str
        Device names to load in addition to the master. Order determines
        how ties are broken when energies overlap, but the data is always
        sorted by x after concatenation.
    metric_suffix : str, optional
        Passed through to LineAnalyzer for metric naming.
    """

    def __init__(
        self,
        line_config_name: str,
        sibling_devices: List[str],
        metric_suffix: Optional[str] = None,
    ):
        super().__init__(line_config_name, metric_suffix)
        self.sibling_devices = sibling_devices

    def load_image(self, file_path: Path) -> Array1D:
        """Load and concatenate data from the master device and all siblings.

        Parameters
        ----------
        file_path : Path
            Path to the master device file, e.g.
            ``/scan_dir/magcam1/magcam1_001.tsv``.

        Returns
        -------
        Array1D
            Nx2 array (x, y) combining all devices, sorted by x.

        Raises
        ------
        FileNotFoundError
            If any sibling file does not exist. The shot is then skipped by
            the scan analyzer.
        """
        file_path = Path(file_path)
        master_device = file_path.parent.name
        filename = file_path.name
        base_dir = file_path.parent.parent
        data_config = self.line_config.data_loading

        # Master file — also sets self.data_metadata
        master_result = read_1d_data(file_path, data_config)
        self.data_metadata = {
            "x_units": master_result.x_units,
            "y_units": master_result.y_units,
            "x_label": master_result.x_label,
            "y_label": master_result.y_label,
        }
        segments = [master_result.data]

        # Sibling files
        for device in self.sibling_devices:
            sibling_filename = filename.replace(master_device, device, 1)
            sibling_path = base_dir / device / sibling_filename
            if not sibling_path.exists():
                raise FileNotFoundError(
                    f"Missing data file for device '{device}': {sibling_path}"
                )
            segments.append(read_1d_data(sibling_path, data_config).data)

        # Concatenate and sort by x
        combined = np.concatenate(segments, axis=0)
        combined = combined[combined[:, 0].argsort()]

        logger.info(
            "Loaded %d segments (%d total points) from master '%s', shot '%s'",
            len(segments),
            len(combined),
            master_device,
            filename,
        )

        return combined
