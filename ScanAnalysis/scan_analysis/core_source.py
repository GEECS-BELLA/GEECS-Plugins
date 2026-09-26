"""Resolve a recipe's scan inputs to the shared readers, without legacy analyzers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from geecs_data_utils.io.array1d import Data1DConfig, read_1d_data
from geecs_data_utils.io.images import read_imaq_image
from geecs_data_utils.io.scan_stack import ShotRef, open_stack, read_frame, read_shot
from geecs_data_utils.shot_files import map_shot_files

from scan_analysis.core_recipe import AnalysisDocument, ScanRecipe, scan_recipe


@dataclass(frozen=True)
class V2ShotSource:
    """Resolved shot references and a snapshotted native-reader configuration.

    Native arrays remain in their original dtype until the v2 evaluator has
    performed its required scaling. The source never caches per-shot arrays or
    creates folders. ``references`` retains ShotRef frame indices unchanged.

    The source is the run's loader: callable per shot, and a context manager
    that keeps **one handle per capture stack open for the whole run** — each
    ``open`` is several protocol round trips over SMB, and a camera scan reads
    every frame of one file. Outside the context every read opens and closes
    the stack itself, as before. Handles never travel: the source pickles
    without them (a pooled run enters it once in every worker), and they are
    closed when the context exits, however it exits. Trace stacks
    (``pva_stack``) still open per read inside the shared 1-D reader.
    """

    data_dir: Path
    references: Mapping[int, Path]
    line_loading_json: str | None = None
    _stacks: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _entered: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Snapshot references so edits to a caller's map cannot rebind reads."""
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))

    def __getstate__(self) -> dict:
        """Pickle the references as a plain dict and never an open handle."""
        return {
            "data_dir": self.data_dir,
            "references": dict(self.references),
            "line_loading_json": self.line_loading_json,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore a closed source with its read-only reference view."""
        object.__setattr__(self, "data_dir", state["data_dir"])
        object.__setattr__(
            self, "references", MappingProxyType(dict(state["references"]))
        )
        object.__setattr__(self, "line_loading_json", state["line_loading_json"])
        object.__setattr__(self, "_stacks", {})
        object.__setattr__(self, "_entered", False)

    def __enter__(self) -> V2ShotSource:
        """Start keeping stack handles open across reads."""
        object.__setattr__(self, "_entered", True)
        return self

    def __exit__(self, *exc: object) -> None:
        """Close every stack this run opened, on success, error or cancellation."""
        object.__setattr__(self, "_entered", False)
        stacks = dict(self._stacks)
        self._stacks.clear()
        for handle in stacks.values():
            handle.close()

    @property
    def open_stacks(self) -> tuple[Path, ...]:
        """The stacks currently held open by this run (for inspection and tests)."""
        return tuple(Path(name) for name in self._stacks)

    def __call__(self, shot: int) -> np.ndarray:
        """The loader protocol: :meth:`load`."""
        return self.load(shot)

    def load(self, shot: int) -> np.ndarray:
        """Read one mapped shot; missing references and reader failures propagate."""
        path = self.references[shot]
        if self.line_loading_json is not None:
            loading = Data1DConfig.model_validate_json(self.line_loading_json)
            return read_1d_data(path, loading).data
        if isinstance(path, ShotRef):
            if not self._entered:
                return read_shot(path)
            name = str(path)
            stack = self._stacks.get(name)
            if stack is None:
                stack = self._stacks[name] = open_stack(path)
            return read_frame(stack, path.shot_index)
        return read_imaq_image(path)


def _resolved(recipe: ScanRecipe | AnalysisDocument) -> ScanRecipe:
    return recipe if isinstance(recipe, ScanRecipe) else scan_recipe(recipe)


def source_directory(recipe: ScanRecipe | AnalysisDocument, scan_folder: Path) -> Path:
    """Validate a preexisting scan and resolve its single device subfolder."""
    folder = Path(scan_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Scan folder does not exist: {folder}")
    file_device = _resolved(recipe).folder
    if file_device in {".", ".."} or "/" in file_device or "\\" in file_device:
        raise ValueError("Input device must name one scan subfolder")
    return folder / file_device


def prepare_source(
    recipe: ScanRecipe | AnalysisDocument, scan_folder: Path, rows: pd.DataFrame
) -> V2ShotSource:
    """Resolve a completed scan using the existing v2 wrapper's source rules.

    Keep device identity separate from the optional folder/file-device override.
    Camera and line default suffixes remain .png and .csv, respectively. Stack
    preference is explicit; pva_stack traces require it and never fall back to
    per-shot files. File mapping reads only identities, not frame arrays. Hosts
    must wait for scan completion before resolving HDF5 stacks over SMB.
    """
    spec = _resolved(recipe)
    device_dir = source_directory(spec, scan_folder)
    loading_json = None
    stacks_only = False
    default_tail = ".png"
    if spec.line:
        loading = Data1DConfig.model_validate_json(spec.line_loading_json)
        loading_json = loading.model_dump_json()
        stacks_only = loading.data_type == "pva_stack"
        default_tail = ".csv"
    references = map_shot_files(
        device_dir,
        rows,
        device=spec.device,
        file_tail=spec.file_tail if spec.file_tail is not None else default_tail,
        prefer_stack=spec.prefer_stack,
        stacks_only=stacks_only,
        file_device=device_dir.name,
    )
    return V2ShotSource(device_dir, references, loading_json)
