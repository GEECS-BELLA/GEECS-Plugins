"""Save core scan products under the sibling analysis tree, using the legacy file names.

One exception to "under the analysis tree": a measure's per-shot sidecar
table (the FROG retrieval's lineouts) is written beside the shot's raw file,
where the legacy analyzer wrote it and where follow-on analyzers read it
(a recipe input's ``folder``). It never creates a directory.

A measure's *shot store* (:class:`ShotStore`, the ``haso`` measure's
wavefront products) is the per-shot product that stays under the analysis
tree: one HDF5 per scan and recipe holding every single-shot measurement's
frame and extras, appended as the run streams.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from geecs_data_utils.io.scan_stack import ShotRef
from geecs_analysis.registry import summary_definition
from geecs_analysis.render import RenderError, single

from scan_analysis.core_products import ProductPlan
from scan_analysis.core_recipe import ScanRecipe


def _component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path component")
    return value


def analysis_directory(scan_folder: Path) -> Path:
    """Resolve the sibling analysis folder without creating any directories."""
    scan_folder = Path(scan_folder)
    if not scan_folder.is_dir():
        raise FileNotFoundError(scan_folder)
    if scan_folder.parent.name != "scans":
        raise ValueError("Expected an existing scans/ScanNNN folder")
    target = scan_folder.parent.parent / "analysis" / scan_folder.name
    if target.resolve().is_relative_to(scan_folder.parent.resolve()):
        raise ValueError("Analysis outputs cannot point into the raw scans tree")
    return target


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
    root = analysis_directory(scan_folder)
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
            try:
                self.part.unlink(missing_ok=True)
            except OSError:
                pass
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


def draw_summary(options, measurements, positions, label: str, figure):
    """The summary draw: the registered kind's layout over the products it consumes.

    ``options`` is one of the recipe's summary entries; the kind's own
    function draws ``measurements`` at ``positions`` under ``label``. The
    sink and the editor's summary preview (``core_preview.preview_summary``)
    make this same call.
    """
    return summary_definition(options).function(
        list(measurements), list(positions), label, options, figure
    )


def _save_figure(fig, path: Path) -> None:
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        fig.clear()


def save_products(
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
    core to that. The logical device names files;
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
            fig = draw_summary(
                options,
                [p.measurement for p in panels],
                [p.position for p in panels],
                plan.position_label,
                spec.figure,
            )
        except RenderError as exc:
            notes.append(f"Skipped {definition.filename}: {exc}")
            continue
        path = _destination(target, f"{device}_{definition.filename}.png")
        _save_figure(fig, path)
        files.append(path)
        if not (line and definition.consumes == "average"):
            display.append(path)
    return SavedProducts(tuple(files), tuple(display), tuple(notes))


def shot_table_path(reference: Path, shot: int, name: str) -> Path:
    """Where one shot's sidecar table goes: beside the shot's own file.

    A per-shot file ``X.png`` gets ``X_<name>.tsv``. A frame of a capture
    stack ``scans/ScanNNN/Dev/Dev.h5`` gets the legacy shot-number name
    ``ScanNNN_Dev_<shot:03d>_<name>.tsv`` beside the stack, one file per
    shot — a name data-utils' shot mapping resolves, so a follow-on recipe
    reading these tables (its input ``folder``) finds them.
    """
    _component(name, "Sidecar name")
    if isinstance(reference, ShotRef):
        stack = Path(str(reference))
        scan = stack.parent.parent.name
        if not re.fullmatch(r"Scan\d{3,}", scan):
            raise ValueError(f"Stack {stack} is not under a scans/ScanNNN folder")
        return stack.parent / f"{scan}_{stack.parent.name}_{shot:03d}_{name}.tsv"
    reference = Path(reference)
    return reference.parent / f"{reference.stem}_{name}.tsv"


def write_shot_table(measurement, reference: Path, shot: int, name: str) -> Path:
    """Write a measurement's extras as one tab-separated table beside the shot.

    Each distinct coordinate axis is written once, as ``<label>_<unit>``,
    before the first extra sampled on it; each extra is a column named by
    its key; shorter columns are padded with NaN. For the FROG retrieval this
    is the legacy ``*_retrieved_lineouts.tsv`` layout exactly (``time_fs``,
    ``temporal_intensity``, ``temporal_phase``, ``wavelength_nm``,
    ``spectral_intensity``, ``spectral_phase``). The shot's directory must
    already exist; nothing is created but the file.
    """
    path = shot_table_path(reference, shot, name)
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
