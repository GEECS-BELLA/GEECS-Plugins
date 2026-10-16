"""Save core scan products under the sibling analysis tree, using the legacy file names.

Two exceptions to "under the analysis tree": a measure's per-shot sidecar
table (the FROG retrieval's lineouts) is written beside the shot's raw file,
where the legacy analyzer wrote it and where follow-on analyzers read it
(a recipe input's ``folder``). It never creates a directory. And a measure's
*sibling outputs* (the magspec ``-interp`` image and ``-interpSpec``
spectrum, a stitcher's input) are written per shot into
``<output name><suffix>/`` folders beside the device's raw folder in the
scan folder, named by the raw file's stem, as the legacy analyzer wrote
them: the suffix folder is the one directory created, never the scan folder.

A measure's *shot store* (:class:`ShotStore`, the ``haso`` measure's
wavefront products) is the per-shot product that stays under the analysis
tree: one HDF5 per scan and recipe holding every single-shot measurement's
frame and extras, appended as the run streams.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pandas as pd
from geecs_data_utils.io.images import write_png16
from geecs_data_utils.io.scan_stack import ShotRef
from geecs_data_utils.scan_info import analysis_folder_for, parse_scan_folder_name
from geecs_analysis.registry import (
    SiblingOutput,
    SummaryOutput,
    summary_definition,
    summary_output,
)
from geecs_analysis.render import RenderError, single

from scan_analysis.core_products import ProductPlan
from scan_analysis.core_recipe import ScanRecipe


# ---------------------------------------------------------------------------
# Output-name contract (the inverse of the product file names written below)
# ---------------------------------------------------------------------------

_BIN_OUTPUT = re.compile(r"_(\d+)_processed(?:_visual)?\.[A-Za-z0-9]+$")
_SUMMARY_MARKERS = ("_average_processed", "_averaged_image_grid.", "_summary_")


def parse_output_filename(name: str) -> tuple[str, int | None]:
    """Classify a product file by its name: ``(kind, bin)``.

    The sink writes per-bin files as ``{device}_{bin}_processed.h5`` /
    ``{device}_{bin}_processed_visual.png``, the scan-level average with
    identifier ``average``, the 2D grid as ``{device}_averaged_image_grid.png``,
    the 1D summary as ``{device}_summary_{mode}.png`` (a ``scalar_fit``'s
    JSON beside it), and the noscan animation as a ``.gif`` — the legacy
    renderers' shapes, kept. Consumers that present these outputs (the data
    portal's Analysis tab) classify through THIS function so the naming
    stays a one-place contract: change a name here and its parser together
    (``tests/test_renderer_output_names.py`` pins the pairing).

    Parameters
    ----------
    name : str
        A filename or path (only the basename is inspected).

    Returns
    -------
    (kind, bin)
        ``("bin", n)`` for per-bin outputs, ``("summary", None)`` for
        scan-level figures, ``("other", None)`` for anything else.
    """
    base = Path(name).name
    match = _BIN_OUTPUT.search(base)
    if match:
        return "bin", int(match.group(1))
    if any(marker in base for marker in _SUMMARY_MARKERS) or base.lower().endswith(
        ".gif"
    ):
        return "summary", None
    return "other", None


def _component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path component")
    return value


@dataclass(frozen=True)
class SavedProducts:
    """Written files, notable display figures, and explicit rendering omissions."""

    files: tuple[Path, ...]
    display_files: tuple[Path, ...]
    notes: tuple[str, ...]


def _destination(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Output file escapes its analysis directory")
    return path


def product_directory(spec: ScanRecipe, scan_folder: Path) -> Path:
    """The analyzer directory every product of ``spec`` goes under (not created).

    ``analysis/ScanNNN/<output_name>/Array2DScanAnalyzer`` (``Array1D…`` for
    a trace recipe); an empty output_name falls back to the device, as the
    legacy wrapper did.
    """
    output = _component(spec.output_name or spec.device, "Output name")
    root = analysis_folder_for(scan_folder, validate=True)
    target = (
        root / output / ("Array1DScanAnalyzer" if spec.line else "Array2DScanAnalyzer")
    )
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output directory escapes its scan analysis folder")
    return target


#: The shot store's frame dtype: the measures that fill one (WaveKit) compute
#: in float32, and the store keeps that precision rather than doubling it.
SHOT_STORE_DTYPE = "f4"


class ShotStore:
    """One HDF5 per scan and recipe of every single-shot measurement's products.

    ``shots`` (int64) lists the shot numbers in the order they arrived;
    ``frame`` is ``(N, …)`` float32, one measurement frame per row; each
    extra is ``extras/<key>`` of the same layout (a pupil arrives as 0/1).
    Rows are one chunk each, gzip level 4, so a follow-on recipe reads one
    shot at a time. The file is written as ``<name>.part`` — created
    exclusively, so a second run storing the same scan refuses instead of
    racing this one to the rename — and renamed into place by
    :meth:`close` once the run ends without a store error, so a reader
    never finds a half-written store under the final name; a run that
    stores nothing leaves no file. The directory is created on the first
    shot (under the analysis tree; never under ``scans/``).

    The first shot fixes every dataset's shape and the set of extras; a
    later shot that disagrees is a ``ValueError`` (the store is then
    discarded by the host and the run goes on without it).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self._handle = None
        self.count = 0

    def add(self, shot: int, measurement) -> None:
        """Append one shot's frame and extras."""
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.part.exists():
                raise OSError(
                    f"{self.part} exists: another run is storing this scan, or one "
                    "died mid-run; remove it once nothing is running"
                )
            self._handle = h5py.File(self.part, "w-")
            self._handle.create_dataset(
                "shots", shape=(0,), maxshape=(None,), dtype="i8"
            )
            self._create("frame", measurement.frame.data)
            for key, extra in measurement.extras.items():
                self._create(f"extras/{key}", extra.data)
        handle = self._handle
        expected = set(handle["extras"]) if "extras" in handle else set()
        if set(measurement.extras) != expected:
            raise ValueError(
                f"Shot {shot}: extras {sorted(measurement.extras)} differ from "
                f"the store's {sorted(expected)}"
            )
        self._append("frame", shot, measurement.frame.data)
        for key, extra in measurement.extras.items():
            self._append(f"extras/{key}", shot, extra.data)
        shots = handle["shots"]
        shots.resize((self.count + 1,))
        shots[self.count] = int(shot)
        self.count += 1

    def _create(self, name: str, data: np.ndarray) -> None:
        self._handle.create_dataset(
            name,
            shape=(0, *data.shape),
            maxshape=(None, *data.shape),
            chunks=(1, *data.shape),
            dtype=SHOT_STORE_DTYPE,
            compression="gzip",
            compression_opts=4,
        )

    def _append(self, name: str, shot: int, data: np.ndarray) -> None:
        dataset = self._handle[name]
        if data.shape != dataset.shape[1:]:
            raise ValueError(
                f"Shot {shot}: {name} shape {data.shape} differs from the "
                f"store's {dataset.shape[1:]}"
            )
        dataset.resize((self.count + 1, *dataset.shape[1:]))
        dataset[self.count] = data

    def close(self, *, keep: bool = True) -> Path | None:
        """Finish the store: rename it into place (``keep``) or discard it.

        Returns the store's path when a file was kept, else ``None``. An
        ``OSError`` from the final flush or the rename (a full or flaky
        share) discards the part file as far as it can and propagates, so
        the host can keep the run's other products.
        """
        handle, self._handle = self._handle, None
        if handle is None:
            return None
        try:
            handle.close()
            if keep and self.count:
                os.replace(self.part, self.path)
                return self.path
        except OSError:
            with contextlib.suppress(OSError):
                self.part.unlink(missing_ok=True)
            raise
        self.part.unlink(missing_ok=True)
        return None

    def __enter__(self) -> ShotStore:
        """Use as a context: the store is discarded if the block raises."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Keep the store on a clean exit only."""
        self.close(keep=exc_type is None)


def shot_store_path(spec: ScanRecipe, scan_folder: Path, name: str) -> Path:
    """Where a recipe's shot store goes: ``<device>_<name>.h5`` in its analyzer directory."""
    device = _component(spec.device, "Diagnostic name")
    _component(name, "Shot store name")
    return _destination(product_directory(spec, scan_folder), f"{device}_{name}.h5")


def draw_product(
    measurement, figure, *, position: float | None = None, position_label: str = ""
):
    """The per-frame draw of every shot and bin product: ``single`` with the recipe's figure.

    A bin product on a scan is titled by its position (``label = value``);
    a shot product, or a preview, carries no title. The editor's frame
    preview (``core_preview.preview_frame``) makes this same call.
    """
    style = figure
    if position is not None and position_label:
        title = f"{position_label} = {position:.3f}"
        style = style.model_copy(update={"axes": {**style.axes, "title": title}})
    return single(measurement, style)


def draw_summary(options, measurements, positions, label: str, figure) -> SummaryOutput:
    """The summary draw: the registered kind's layout over the products it consumes.

    ``options`` is one of the recipe's summary entries; the kind's own
    function draws ``measurements`` at ``positions`` under ``label``. The
    result is normalised to a ``SummaryOutput`` (the figure, plus any
    scan-level scalars and notes the kind computed). The sink and the
    editor's summary preview (``core_preview.preview_summary``) make this
    same call.
    """
    return summary_output(
        summary_definition(options).function(
            list(measurements), list(positions), label, options, figure
        )
    )


def summary_scalars_document(
    kind: str, position_label: str, output: SummaryOutput
) -> str:
    """The JSON sidecar of a summary's scan-level scalars.

    ``{"kind", "position_label", "scalars", "notes"}``; a nonfinite value is
    written as ``null`` (strict JSON has no NaN), its reason in ``notes``.
    """
    scalars = {
        key: value if math.isfinite(value) else None
        for key, value in output.scalars.items()
    }
    document = {
        "kind": kind,
        "position_label": position_label,
        "scalars": scalars,
        "notes": list(output.notes),
    }
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def _save_figure(fig, path: Path) -> None:
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        fig.clear()


def save_products(  # noqa: C901, PLR0912, PLR0915
    plan: ProductPlan, spec: ScanRecipe, scan_folder: Path
) -> SavedProducts:
    """Write the HDF5/PNG products; disabled saves perform no writes.

    Every single product stores its data; each bin's figure is the recipe's
    per-frame draw titled by position. The scan-level figures are the
    recipe's ``summaries`` in order, each drawn by its registered kind from
    the products it consumes (the ordered panels, or the noscan average) and
    skipped without a note when this run produced none of those. Summary
    figures are the display files, except a trace's averaged figure: the
    legacy line wrapper never listed it, and the route comparison holds the
    core to that. A kind that computes scan-level scalars (``scalar_fit``)
    also gets ``<stem>.json`` beside its PNG (:func:`summary_scalars_document`;
    not a display file), and its notes join the returned notes. The logical device names files;
    output_name selects the analyzer directory. HDF5 retains the legacy
    dataset name, dtype and gzip level. Rendering errors omit only their
    figure and are returned as notes; data/write errors propagate. Scalar
    persistence is independent of this sink and of ``save``.
    """
    if not spec.save or not (plan.singles or plan.summary):
        return SavedProducts((), (), plan.notes)
    device = _component(spec.device, "Diagnostic name")
    line = spec.line
    target = product_directory(spec, scan_folder)
    for product in plan.singles:
        _component(str(product.identifier), "Product identifier")
    target.mkdir(parents=True, exist_ok=True)
    files, display, notes = [], [], list(plan.notes)
    average = None
    for product in plan.singles:
        stem = f"{device}_{product.identifier}_processed"
        data_path = _destination(target, f"{stem}.h5")
        frame = product.measurement.frame
        data = (
            frame.as_trace().astype(spec.recipe.storage_dtype) if line else frame.data
        )
        with h5py.File(data_path, "w") as handle:
            handle.create_dataset(
                "data" if line else "image",
                data=data,
                compression="gzip",
                compression_opts=4,
            )
        files.append(data_path)
        if product.identifier == "average":
            # The noscan average is a summary: the ``average`` kind draws it.
            average = product
            continue
        try:
            fig = draw_product(
                product.measurement,
                spec.figure,
                position=product.position,
                position_label=plan.position_label,
            )
        except RenderError as exc:
            notes.append(f"Skipped {stem}_visual.png: {exc}")
            continue
        path = _destination(target, f"{stem}_visual.png")
        _save_figure(fig, path)
        files.append(path)
    for options in spec.summaries:
        definition = summary_definition(options)
        if definition.consumes == "average":
            if average is None:
                continue
            panels = (average,)
        else:
            if not plan.summary:
                continue
            panels = plan.summary
        try:
            output = draw_summary(
                options,
                [p.measurement for p in panels],
                [p.position for p in panels],
                plan.position_label,
                spec.figure,
            )
        except RenderError as exc:
            notes.append(f"Skipped {definition.filename}: {exc}")
            continue
        stem = f"{device}_{definition.filename}"
        path = _destination(target, f"{stem}.png")
        _save_figure(output.figure, path)
        files.append(path)
        if not (line and definition.consumes == "average"):
            display.append(path)
        if output.scalars:
            sidecar = _destination(target, f"{stem}.json")
            sidecar.write_text(
                summary_scalars_document(options.kind, plan.position_label, output)
            )
            files.append(sidecar)
        notes.extend(f"{definition.filename}: {note}" for note in output.notes)
    return SavedProducts(tuple(files), tuple(display), tuple(notes))


def _shot_location(
    reference: Path, shot: int, file_numbers: Mapping[int, int] | None = None
) -> tuple[Path, str]:
    """The folder a shot's file sits in and the stem that names it.

    A per-shot file ``X.png`` is its own stem. A frame of a capture stack
    ``scans/ScanNNN/Dev/Dev.h5`` takes the legacy shot-number stem
    ``ScanNNN_Dev_<number:03d>`` — a name data-utils' shot mapping resolves.
    In a scan-mode run (``file_numbers`` given: the announced shots,
    ``core_scan.file_numbers``) the number is the shot's announced one, as
    the mapping reads it back; a shot without one raises ``ValueError``
    rather than take a name that maps to another row. Otherwise it is the
    shot.
    """
    if isinstance(reference, ShotRef):
        stack = Path(str(reference))
        scan = stack.parent.parent.name
        if parse_scan_folder_name(scan) is None:
            raise ValueError(f"Stack {stack} is not under a scans/ScanNNN folder")
        if file_numbers is None:
            number = shot
        elif shot in file_numbers:
            number = file_numbers[shot]
        else:
            raise ValueError(
                f"Shot {shot} has no announced shot; a shot-number name "
                "would not map back to it"
            )
        return stack.parent, f"{scan}_{stack.parent.name}_{number:03d}"
    reference = Path(reference)
    return reference.parent, reference.stem


def shot_table_path(
    reference: Path,
    shot: int,
    name: str,
    file_numbers: Mapping[int, int] | None = None,
) -> Path:
    """Where one shot's sidecar table goes: beside the shot's own file.

    A per-shot file ``X.png`` gets ``X_<name>.tsv``. A frame of a capture
    stack ``scans/ScanNNN/Dev/Dev.h5`` gets the legacy shot-number name
    ``ScanNNN_Dev_<number:03d>_<name>.tsv`` beside the stack (the number
    as ``_shot_location`` picks it), one file per shot — a name data-utils'
    shot mapping resolves, so a follow-on recipe reading these tables (its
    input ``folder``) finds them.
    """
    _component(name, "Sidecar name")
    folder, stem = _shot_location(reference, shot, file_numbers)
    return folder / f"{stem}_{name}.tsv"


def write_sibling_outputs(
    measurement,
    outputs: tuple[SiblingOutput, ...],
    reference: Path,
    shot: int,
    output_name: str,
    file_numbers: Mapping[int, int] | None = None,
) -> list[Path]:
    """Write one shot's sibling outputs; return the files written.

    Each goes to ``<scan folder>/<output_name><suffix>/<stem>.png|.tsv``,
    the stem the shot's raw file's (``_shot_location``) — byte for byte the
    legacy magspec writer's per-shot files (a capture-stack frame gets the
    legacy shot-number stem; the legacy writer named every stack frame by
    the stack's own stem, one file overwritten per shot): ``png16`` is the data times the output's scale,
    clipped to 0..65535 and truncated to uint16; ``tsv`` is ``np.savetxt``
    of the axis and the samples under a ``<label> [<unit>]`` header. The
    scan folder must already exist (a missing one raises
    ``FileNotFoundError`` and nothing is created); the suffix folder is
    created with ``exist_ok`` and no parents. Written whatever ``save``
    says, as the legacy analyzer did.
    """
    folder, stem = _shot_location(reference, shot, file_numbers)
    scan_folder = folder.parent
    if not scan_folder.is_dir():
        raise FileNotFoundError(
            f"Scan folder {scan_folder} is not visible; refusing to create "
            "sibling output folders"
        )
    written = []
    for output in outputs:
        directory = scan_folder / _component(
            f"{output_name}{output.suffix}", "Sibling output folder"
        )
        if output.source == "frame":
            frame = measurement.frame
        elif output.source in measurement.extras:
            frame = measurement.extras[output.source]
        else:
            raise ValueError(f"The measurement has no {output.source!r} extra")
        if output.format == "png16":
            if frame.data.ndim != 2:
                raise ValueError(f"{output.suffix}: a PNG is written from an image")
            image = np.clip(frame.data * output.scale, 0, 65535).astype(np.uint16)
            directory.mkdir(exist_ok=True)
            path = directory / f"{stem}.png"
            write_png16(path, image)
        else:
            if frame.data.ndim != 1:
                raise ValueError(f"{output.suffix}: a TSV is written from a trace")
            axis = frame.axes[0]
            directory.mkdir(exist_ok=True)
            path = directory / f"{stem}.tsv"
            np.savetxt(
                str(path),
                np.column_stack([axis.values, frame.data]),
                delimiter="\t",
                header=f"{axis.label} [{axis.unit}]\t{frame.label} [{frame.unit}]",
                comments="",
            )
        written.append(path)
    return written


def write_shot_table(
    measurement,
    reference: Path,
    shot: int,
    name: str,
    file_numbers: Mapping[int, int] | None = None,
) -> Path:
    """Write a measurement's extras as one tab-separated table beside the shot.

    Each distinct coordinate axis is written once, as ``<label>_<unit>``,
    before the first extra sampled on it; each extra is a column named by
    its key; shorter columns are padded with NaN. For the FROG retrieval this
    is the legacy ``*_retrieved_lineouts.tsv`` layout exactly (``time_fs``,
    ``temporal_intensity``, ``temporal_phase``, ``wavelength_nm``,
    ``spectral_intensity``, ``spectral_phase``). The shot's directory must
    already exist; nothing is created but the file.
    """
    path = shot_table_path(reference, shot, name, file_numbers)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Shot directory does not exist: {path.parent}")
    columns: dict[str, np.ndarray] = {}
    for key, frame in measurement.extras.items():
        if frame.data.ndim != 1:
            raise ValueError(f"Extra {key} is not a trace; it cannot be a column")
        axis = frame.axes[0]
        axis_name = f"{axis.label}_{axis.unit}" if axis.label else f"{key}_x"
        if axis_name not in columns:
            columns[axis_name] = axis.values
        elif not np.array_equal(columns[axis_name], axis.values):
            raise ValueError(f"Extras disagree on the {axis_name} coordinates")
        if key in columns:
            raise ValueError(f"Extra {key} collides with an axis column")
        columns[key] = frame.data
    length = max(len(values) for values in columns.values())
    table = pd.DataFrame(
        {
            key: np.pad(
                np.asarray(values, dtype=np.float64),
                (0, length - len(values)),
                constant_values=np.nan,
            )
            for key, values in columns.items()
        }
    )
    table.to_csv(path, sep="\t", index=False)
    return path
