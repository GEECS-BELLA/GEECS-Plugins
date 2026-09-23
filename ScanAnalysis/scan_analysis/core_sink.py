"""Save core scan products under the sibling analysis tree, using v2 file names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
from geecs_analysis.compat.v2 import V2Recipe
from geecs_analysis.compat.v2_render import image_grid_v2, single_v2, waterfall_v2
from geecs_analysis.render import RenderError
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.core_products import ProductPlan


def _component(value: str, label: str, *, allow_empty: bool = False) -> str:
    if (
        (not value and not allow_empty)
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
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


def save_products(
    plan: ProductPlan, recipe: V2Recipe, document: AnalysisDiagnostic, scan_folder: Path
) -> SavedProducts:
    """Write legacy HDF5/PNG products; disabled saves perform no writes.

    The logical device names files; output_name selects the analyzer directory.
    HDF5 retains the legacy dataset name, dtype and gzip level. Rendering errors
    omit only their figure and are returned as notes; data/write errors propagate.
    Scalar persistence is independent of this sink and of scan.save.
    """
    if not document.scan.save or not (plan.singles or plan.summary):
        return SavedProducts((), (), plan.notes)
    device = _component(document.name, "Diagnostic name")
    output = _component(document.effective_output_name, "Output name", allow_empty=True)
    line = recipe.input_kind == "line"
    root = analysis_directory(scan_folder)
    target = root / output / ("Array1DScanAnalyzer" if line else "Array2DScanAnalyzer")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output directory escapes its scan analysis folder")
    for product in plan.singles:
        _component(str(product.identifier), "Product identifier")
    target.mkdir(parents=True, exist_ok=True)
    files, display, notes = [], [], list(plan.notes)
    options = document.scan.renderer
    for product in plan.singles:
        stem = f"{device}_{product.identifier}_processed"
        data_path = _destination(target, f"{stem}.h5")
        frame = product.measurement.frame
        data = frame.as_trace().astype(recipe.storage_dtype) if line else frame.data
        with h5py.File(data_path, "w") as handle:
            handle.create_dataset(
                "data" if line else "image",
                data=data,
                compression="gzip",
                compression_opts=4,
            )
        files.append(data_path)
        title = (
            f"{plan.position_label} = {product.position:.3f}"
            if product.position is not None and plan.position_label
            else None
        )
        try:
            fig = single_v2(product.measurement, options, title=title)
        except RenderError as exc:
            notes.append(f"Skipped {stem}_visual.png: {exc}")
            continue
        path = _destination(target, f"{stem}_visual.png")
        try:
            fig.savefig(path, bbox_inches="tight")
        finally:
            fig.clear()
        files.append(path)
        if not line and product.identifier == "average":
            display.append(path)
    if plan.summary:
        measurements = [p.measurement for p in plan.summary]
        positions = [p.position for p in plan.summary]
        suffix = "summary_waterfall" if line else "averaged_image_grid"
        try:
            fig = (
                waterfall_v2(measurements, positions, plan.position_label, options)
                if line
                else image_grid_v2(measurements, positions, options)
            )
        except RenderError as exc:
            notes.append(f"Skipped {suffix}: {exc}")
        else:
            path = _destination(target, f"{device}_{suffix}.png")
            try:
                fig.savefig(path, bbox_inches="tight")
            finally:
                fig.clear()
            files.append(path)
            display.append(path)
    return SavedProducts(tuple(files), tuple(display), tuple(notes))
