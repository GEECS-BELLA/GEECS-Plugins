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
from typing import Dict, List, Optional

import numpy as np

from image_analysis.data_1d_utils import read_1d_data
from image_analysis.offline_analyzers.line_analyzer import LineAnalyzer
from image_analysis.types import Array1D, ImageAnalyzerResult

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
        self._device_in_filename: Optional[str] = None

    def load_image(self, file_path: Path) -> Array1D:
        """Load and concatenate data from the master device and all siblings.

        Parameters
        ----------
        file_path : Path
            Path to the master device file, e.g.
            /scan_dir/magcam1/magcam1_001.tsv.

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

        # Directory names may have a channel suffix not present in filenames
        # (e.g. dir "HTT-C23_1_MagSpec1-interpSpec", file "..._HTT-C23_1_MagSpec1_004.tsv").
        # Find the longest hyphen-stripped prefix of master_device that appears in the stem.
        stem = file_path.stem
        device_in_filename = master_device
        while device_in_filename not in stem:
            if "-" not in device_in_filename:
                device_in_filename = master_device  # fallback: no stripping
                break
            device_in_filename = device_in_filename.rsplit("-", 1)[0]
        dir_suffix = master_device[len(device_in_filename) :]  # e.g. "-interpSpec"

        # Cache for use in _save_stitched_output
        self._device_in_filename = device_in_filename

        # Sibling files
        for device in self.sibling_devices:
            sibling_in_file = (
                device[: -len(dir_suffix)]
                if dir_suffix and device.endswith(dir_suffix)
                else device
            )
            sibling_filename = filename.replace(device_in_filename, sibling_in_file, 1)
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

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Analyze stitched line data and save the result as a TSV."""
        result = super().analyze_image(image=image, auxiliary_data=auxiliary_data)

        file_path = auxiliary_data.get("file_path") if auxiliary_data else None
        if file_path is not None:
            try:
                self._save_stitched_output(result, Path(file_path))
            except Exception as e:
                logger.warning("Failed to save stitched output for %s: %s", file_path, e)

        return result

    def _save_stitched_output(
        self, result: ImageAnalyzerResult, file_path: Path
    ) -> None:
        """Save the stitched lineout as a TSV alongside the per-camera interpSpec files.

        Output goes to {scan_dir}/{line_config_name}/{stem}.tsv where the
        device name in the stem is replaced with the line config name, matching
        the naming convention used by the individual camera interpSpec directories.
        """
        if result.line_data is None:
            return

        device_in_filename = self._device_in_filename or file_path.parent.name
        output_name = self.line_config.name
        new_stem = file_path.stem.replace(device_in_filename, output_name, 1)

        output_dir = file_path.parent.parent / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = result.metadata or {}
        x_label = metadata.get("x_label", "X")
        x_units = metadata.get("x_units", "")
        y_label = metadata.get("y_label", "Y")
        y_units = metadata.get("y_units", "")
        x_header = f"{x_label} [{x_units}]" if x_units else x_label
        y_header = f"{y_label} [{y_units}]" if y_units else y_label

        output_path = output_dir / f"{new_stem}.tsv"
        np.savetxt(
            str(output_path),
            result.line_data,
            delimiter="	",
            header=f"{x_header}	{y_header}",
            comments="",
        )
        logger.info("Saved stitched lineout to %s", output_path)
