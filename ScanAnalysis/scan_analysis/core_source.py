"""Resolve a recipe's scan inputs to the shared readers, without legacy analyzers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from geecs_data_utils.io.array1d import Data1DConfig, read_1d_data
from geecs_data_utils.io.himg_stack import HIMG_SUFFIX
from geecs_data_utils.io.images import read_imaq_image
from geecs_data_utils.io.scan_stack import (
    ShotRef,
    find_stack_file,
    open_stack,
    read_frame,
    read_shot,
)
from geecs_data_utils.shot_files import StackMappingUnavailable, map_shot_files

from scan_analysis.core_recipe import AnalysisDocument, ScanRecipe, scan_recipe

logger = logging.getLogger(__name__)

#: How the message that refuses an unconverted HASO scan names the way out.
STACK_HINT = (
    "convert the scan first with the himg_to_stack analyzer "
    "(HasoLift_stack in the Data Portal's Analysis tab)"
)


def stack_required(device_dir: Path) -> Path:
    """The device folder's capture stack, or the refusal every ``.himg`` reader gives.

    ``.himg`` frames enter the analysis core only through the device's
    stack (``<device>/<device>.h5``, written by ``himg_to_stack``): the
    per-shot files are never read by a run — not by the source, not by a
    dark scan's background loader, not by the WaveKit service that takes
    its sensor header from the stack. The refusal is a
    ``StackMappingUnavailable`` (a ``LookupError``), which the scan host
    reports as missing data.
    """
    stack = find_stack_file(Path(device_dir))
    if stack is None:
        raise StackMappingUnavailable(
            f"no capture stack in {device_dir}: {HIMG_SUFFIX} frames are read "
            f"from the stack only — {STACK_HINT}"
        )
    return stack


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

    ``siblings`` stitches a multi-device trace (the legacy ``LineStitcher``):
    one ``(folder, shot → reference)`` pair per sibling device, read with
    the same trace reader; each loaded shot is the input's trace joined with
    every sibling's same-shot trace and sorted by x, exactly as the legacy
    stitcher joined them. A shot a sibling lacks is stitched without it,
    with a warning.
    """

    data_dir: Path
    references: Mapping[int, Path]
    line_loading_json: str | None = None
    siblings: tuple[tuple[str, Mapping[int, Path]], ...] = ()
    _stacks: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _entered: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Snapshot references so edits to a caller's map cannot rebind reads."""
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))
        object.__setattr__(
            self,
            "siblings",
            tuple(
                (folder, MappingProxyType(dict(refs))) for folder, refs in self.siblings
            ),
        )

    def __getstate__(self) -> dict:
        """Pickle the references as a plain dict and never an open handle."""
        return {
            "data_dir": self.data_dir,
            "references": dict(self.references),
            "line_loading_json": self.line_loading_json,
            "siblings": [(folder, dict(refs)) for folder, refs in self.siblings],
        }

    def __setstate__(self, state: dict) -> None:
        """Restore a closed source with its read-only reference view."""
        object.__setattr__(self, "data_dir", state["data_dir"])
        object.__setattr__(
            self, "references", MappingProxyType(dict(state["references"]))
        )
        object.__setattr__(self, "line_loading_json", state["line_loading_json"])
        object.__setattr__(
            self,
            "siblings",
            tuple(
                (folder, MappingProxyType(dict(refs)))
                for folder, refs in state["siblings"]
            ),
        )
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
            data = read_1d_data(path, loading).data
            if not self.siblings:
                return data
            segments = [data]
            for folder, refs in self.siblings:
                if shot not in refs:
                    logger.warning(
                        "Shot %s: no %s trace; stitching the available segments "
                        "without it",
                        shot,
                        folder,
                    )
                    continue
                segments.append(read_1d_data(refs[shot], loading).data)
            combined = np.concatenate(segments, axis=0)
            # The legacy stitcher's sort (numpy's default kind): same order
            # of equal-x samples, so the processed trace is identical.
            return combined[combined[:, 0].argsort()]
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
    per-shot files, and neither do ``.himg`` frames (a HASO device): the
    per-shot vendor files are read only through the stack ``himg_to_stack``
    writes, so an unconverted scan is refused (``StackMappingUnavailable``)
    with the way out named. File mapping reads only identities, not frame
    arrays. Hosts must wait for scan completion before resolving HDF5
    stacks over SMB.
    """
    spec = _resolved(recipe)
    device_dir = source_directory(spec, scan_folder)
    loading_json = None
    stacks_only = False
    default_tail = ".png"
    prefer_stack = spec.prefer_stack
    if spec.line:
        loading = Data1DConfig.model_validate_json(spec.line_loading_json)
        loading_json = loading.model_dump_json()
        stacks_only = loading.data_type == "pva_stack"
        default_tail = ".csv"
    file_tail = spec.file_tail if spec.file_tail is not None else default_tail
    if not spec.line and file_tail == HIMG_SUFFIX:
        stack_required(device_dir)
        prefer_stack = stacks_only = True
    references = map_shot_files(
        device_dir,
        rows,
        device=spec.device,
        file_tail=file_tail,
        prefer_stack=prefer_stack,
        stacks_only=stacks_only,
        file_device=device_dir.name,
    )
    siblings = tuple(
        (folder, _sibling_references(spec, folder, scan_folder, rows, stacks_only))
        for folder in spec.siblings
    )
    return V2ShotSource(device_dir, references, loading_json, siblings)


def _sibling_references(
    spec: ScanRecipe,
    folder: str,
    scan_folder: Path,
    rows: pd.DataFrame,
    stacks_only: bool,
) -> dict[int, Path]:
    """Map one sibling device's shots the way the input's are mapped.

    The sibling's device name is its folder with the input's folder suffix
    removed (``X-interpSpec`` for input device ``Y`` in folder
    ``Y-interpSpec`` is device ``X``), so its own timestamp column joins its
    own files. A missing sibling folder, or a stack-only sibling without a
    stack, maps no shots (warned once); the input's shots are then stitched
    without it.
    """
    if folder in {".", ".."} or "/" in folder or "\\" in folder:
        raise ValueError("A sibling must name one scan subfolder")
    suffix = (
        spec.folder[len(spec.device) :]
        if spec.folder != spec.device and spec.folder.startswith(spec.device)
        else ""
    )
    device = folder[: -len(suffix)] if suffix and folder.endswith(suffix) else folder
    directory = Path(scan_folder) / folder
    if not directory.is_dir():
        logger.warning(
            "Sibling folder %s does not exist; stitching without it", directory
        )
        return {}
    try:
        return map_shot_files(
            directory,
            rows,
            device=device,
            file_tail=spec.file_tail if spec.file_tail is not None else ".csv",
            prefer_stack=spec.prefer_stack,
            stacks_only=stacks_only,
            file_device=folder,
        )
    except StackMappingUnavailable as exc:
        # A stack-only sibling with no stack is a missing sibling, not a
        # missing input: the run stitches without it.
        logger.warning(
            "Sibling %s maps no shots (%s); stitching without it", folder, exc
        )
        return {}
