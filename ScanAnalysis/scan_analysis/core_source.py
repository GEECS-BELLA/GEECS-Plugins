"""Resolve v2 scan inputs to the shared readers, without legacy analyzers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from geecs_data_utils.io.array1d import Data1DConfig, read_1d_data
from geecs_data_utils.io.images import read_imaq_image
from geecs_data_utils.io.scan_stack import ShotRef, read_shot
from geecs_data_utils.shot_files import map_shot_files
from geecs_schemas.analysis import AnalysisDiagnostic
from geecs_schemas.analysis.processing_1d import Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig


@dataclass(frozen=True)
class V2ShotSource:
    """Resolved shot references and a snapshotted native-reader configuration.

    Native arrays remain in their original dtype until the v2 evaluator has
    performed its required scaling. The source never caches per-shot arrays or
    creates folders. ``references`` retains ShotRef frame indices unchanged.
    """

    data_dir: Path
    references: Mapping[int, Path]
    line_loading_json: str | None = None

    def __post_init__(self) -> None:
        """Snapshot references so edits to a caller's map cannot rebind reads."""
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))

    def load(self, shot: int) -> np.ndarray:
        """Read one mapped shot; missing references and reader failures propagate."""
        path = self.references[shot]
        if self.line_loading_json is not None:
            loading = Data1DConfig.model_validate_json(self.line_loading_json)
            return read_1d_data(path, loading).data
        if isinstance(path, ShotRef):
            return read_shot(path)
        return read_imaq_image(path)


def prepare_source(
    document: AnalysisDiagnostic, scan_folder: Path, rows: pd.DataFrame
) -> V2ShotSource:
    """Resolve a completed scan using the existing v2 wrapper's source rules.

    Keep device identity separate from the optional folder/file-device override.
    Camera and line default suffixes remain .png and .csv, respectively. Stack
    preference is explicit; pva_stack traces require it and never fall back to
    per-shot files. File mapping reads only identities, not frame arrays. Hosts
    must wait for scan completion before resolving HDF5 stacks over SMB.
    """
    config = document.image
    if not isinstance(config, (CameraConfig, Line1DConfig)):
        raise ValueError("A camera or line input configuration is required")
    folder = Path(scan_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Scan folder does not exist: {folder}")
    file_device = document.scan.device or document.name
    if file_device in {".", ".."} or "/" in file_device or "\\" in file_device:
        raise ValueError("Input device must name one scan subfolder")
    device_dir = folder / file_device
    loading_json = None
    stacks_only = False
    default_tail = ".png"
    if isinstance(config, Line1DConfig):
        loading = Data1DConfig.model_validate(
            config.data_loading.model_dump(mode="json")
        )
        loading_json = loading.model_dump_json()
        stacks_only = loading.data_type == "pva_stack"
        default_tail = ".csv"
    references = map_shot_files(
        device_dir,
        rows,
        device=document.name,
        file_tail=document.scan.file_tail
        if document.scan.file_tail is not None
        else default_tail,
        prefer_stack=document.scan.data_format == "device_hdf5",
        stacks_only=stacks_only,
        file_device=file_device,
    )
    return V2ShotSource(device_dir, references, loading_json)
