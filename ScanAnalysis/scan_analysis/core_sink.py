"""Save core scan products under the sibling analysis tree, using the legacy file names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
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
    # An empty output_name falls back to the device, as the legacy wrapper did.
    output = _component(spec.output_name or spec.device, "Output name")
    line = spec.line
    root = analysis_directory(scan_folder)
    target = root / output / ("Array1DScanAnalyzer" if line else "Array2DScanAnalyzer")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output directory escapes its scan analysis folder")
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
        style = spec.figure
        if product.position is not None and plan.position_label:
            title = f"{plan.position_label} = {product.position:.3f}"
            style = style.model_copy(update={"axes": {**style.axes, "title": title}})
        try:
            fig = single(product.measurement, style)
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
            fig = definition.function(
                [p.measurement for p in panels],
                [p.position for p in panels],
                plan.position_label,
                options,
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
