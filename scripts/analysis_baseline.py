"""Capture and compare beam/line numerical baselines for the analysis migration.

Run from the repository's root Poetry environment::

    poetry run python scripts/analysis_baseline.py capture \
        --config /configs/beam.yaml --output /tmp/beam.npz /data/shot.png
    poetry run python scripts/analysis_baseline.py compare /tmp/beam.npz /tmp/new.npz

Inputs are explicit native files, read through the existing analyzer's loader;
scan discovery and scan.data_format are deliberately not involved. Only beam
and line are supported. Analysis receives in-memory arrays through the same
write-free seam used by the portal and optimizer. No scan runner is invoked.
Recipes with scan-dependent backgrounds, file backgrounds or vignette maps
are refused until the harness fingerprints those additional dependencies.

Archives contain numeric arrays plus JSON metadata (no pickle), including the
resolved v2 recipe and ordered input hashes. The future backend can construct
Snapshot/Result directly without importing ImageAnalysis. Comparison is exact
by default; explicit --rtol/--atol apply to scalars and processed arrays. This
first harness covers numerical outputs, not figures, s-files or scan binning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class Result:
    """Backend-independent numerical output from one analysis invocation."""

    data: NDArray
    scalars: dict[str, float]


@dataclass
class Snapshot:
    """Ordered numerical results and the exact workload that produced them."""

    recipe: str
    inputs: tuple[str, ...]
    mode: str
    results: tuple[Result, ...]


def capture(
    config: Path,
    inputs: Sequence[Path],
    mode: str = "per_shot",
    *,
    backend: str = "legacy",
) -> Snapshot:
    """Run either beam/line backend without passing input paths to analysis."""
    from image_analysis.config import create_image_analyzer, load_diagnostic
    from image_analysis.ephemeral import run_document_ephemeral

    if backend not in {"legacy", "core"}:
        raise ValueError(f"Unknown analysis backend: {backend}")
    if not inputs:
        raise ValueError("At least one input is required")
    diagnostic = load_diagnostic(config)
    if diagnostic.analyzer.kind not in {"beam", "line"}:
        raise ValueError("Only beam and line numerical baselines are supported")
    background = diagnostic.image.background
    vignette = getattr(diagnostic.image, "vignette", None)
    if (
        diagnostic.scan.background_source is not None
        or (background is not None and background.method == "from_file")
        or (vignette is not None and vignette.method == "map_file")
    ):
        raise ValueError("External processing dependencies are not fingerprinted yet")
    if mode not in {"per_shot", "per_bin"}:
        raise ValueError(f"Unknown analysis mode: {mode}")
    if mode == "per_bin" and diagnostic.analyzer.kind != "beam":
        raise ValueError("per_bin currently models camera-frame optimization only")
    compiled = None
    if backend == "core":
        from geecs_analysis.compat.v2 import compile_v2

        compiled = compile_v2(diagnostic)
    # Keep the SAME reader for both backends: this harness isolates analysis
    # migration from the separate source/reader migration.
    loader = create_image_analyzer(diagnostic)
    frames = []
    identities = []
    for path in inputs:
        # Hash before and after loading so a changing acquisition file cannot
        # silently label one array with a different file's fingerprint.
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(loader.load_image(path))
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise ValueError(f"Input changed while reading: {path.name}")
        identities.append(f"{path.name}:{after}")
    if mode == "per_bin":
        frames = [np.mean(np.stack(frames), axis=0)]
    if backend == "core":
        from geecs_analysis.compat.v2 import analyze_v2

        measured = [analyze_v2(frame, compiled) for frame in frames]
        results = tuple(
            Result(
                result.frame.as_trace()
                if result.frame.data.ndim == 1
                else result.frame.data,
                dict(result.scalars),
            )
            for result in measured
        )
    else:
        results = tuple(
            Result(np.array(result.get_primary_data(), copy=True), dict(result.scalars))
            for result in run_document_ephemeral(diagnostic, frames)
        )
    return Snapshot(
        recipe=diagnostic.model_dump_json(),
        inputs=tuple(identities),
        mode=mode,
        results=results,
    )


def save(snapshot: Snapshot, output: Path) -> None:
    """Write a new archive in an existing directory, never inside a raw scan."""
    if not snapshot.inputs or not snapshot.results:
        raise ValueError("Cannot save an empty baseline")
    resolved = output.resolve()
    if any(
        parent.parent.name == "scans" and re.fullmatch(r"Scan\d+", parent.name)
        for parent in (resolved, *resolved.parents)
    ):
        raise ValueError("Baseline archives must be outside scans/ScanNNN")
    arrays = {}
    scalar_keys = []
    for index, result in enumerate(snapshot.results):
        if result.data.dtype.kind not in "biuf":
            raise ValueError("Processed data must be real numeric arrays")
        keys = sorted(result.scalars)
        scalar_keys.append(keys)
        arrays[f"data_{index}"] = result.data
        arrays[f"scalars_{index}"] = np.asarray(
            [result.scalars[key] for key in keys], dtype=np.float64
        )
    manifest = {
        "version": 1,
        "recipe": snapshot.recipe,
        "inputs": snapshot.inputs,
        "mode": snapshot.mode,
        "scalar_keys": scalar_keys,
    }
    arrays["manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
    # Exclusive creation prevents an accidental rerun from replacing the
    # reference. No parent directories are created by this tool.
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load(path: Path) -> Snapshot:
    """Read a numerical snapshot without enabling numpy pickle loading."""
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest["version"] != 1:
            raise ValueError("Unsupported baseline archive version")
        results = []
        for index, keys in enumerate(manifest["scalar_keys"]):
            scalars = archive[f"scalars_{index}"]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate scalar keys in baseline")
            results.append(
                Result(
                    archive[f"data_{index}"].copy(),
                    dict(zip(keys, map(float, scalars), strict=True)),
                )
            )
        return Snapshot(
            manifest["recipe"],
            tuple(manifest["inputs"]),
            manifest["mode"],
            tuple(results),
        )


def compare(
    reference: Snapshot, candidate: Snapshot, *, rtol: float = 0, atol: float = 0
) -> list[str]:
    """Return actionable differences, refusing mismatched workloads and NaN parity."""
    if any(not math.isfinite(value) or value < 0 for value in (rtol, atol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    differences = []
    if not reference.results or not candidate.results:
        return ["empty baseline cannot establish parity"]
    for field in ("recipe", "inputs", "mode"):
        if getattr(reference, field) != getattr(candidate, field):
            differences.append(f"workload {field} differs")
    if len(reference.results) != len(candidate.results):
        differences.append("result count differs")
    if differences:
        return differences

    def check_array(label: str, expected: NDArray, actual: NDArray) -> None:
        if expected.shape != actual.shape:
            differences.append(f"{label}: shape {actual.shape} != {expected.shape}")
        elif not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
            differences.append(f"{label}: non-finite values require investigation")
        else:
            if rtol == 0 and atol == 0:
                # Mixed int/float numpy comparisons can also round large ints.
                # Python scalar equality preserves their exact numeric values.
                equal = np.array_equal(
                    actual if actual.dtype == expected.dtype else actual.astype(object),
                    expected
                    if actual.dtype == expected.dtype
                    else expected.astype(object),
                )
            else:
                equal = np.allclose(actual, expected, rtol=rtol, atol=atol)
            if not equal:
                differences.append(
                    f"{label}: values differ (rtol={rtol:g}, atol={atol:g})"
                )

    for index, (expected, actual) in enumerate(
        zip(reference.results, candidate.results, strict=True)
    ):
        prefix = f"result {index}"
        check_array(f"{prefix} data", expected.data, actual.data)
        missing = expected.scalars.keys() - actual.scalars.keys()
        extra = actual.scalars.keys() - expected.scalars.keys()
        if missing or extra:
            differences.append(
                f"{prefix} scalar keys: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for key in sorted(expected.scalars.keys() & actual.scalars.keys()):
            check_array(
                f"{prefix} scalar {key}",
                np.asarray(expected.scalars[key]),
                np.asarray(actual.scalars[key]),
            )
    return differences


def main(argv: Sequence[str] | None = None) -> int:
    """Capture a reference or compare a candidate, returning nonzero on differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser(
        "capture", help="capture beam/line outputs from either backend"
    )
    record.add_argument("--backend", choices=("legacy", "core"), default="legacy")
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--mode", choices=("per_shot", "per_bin"), default="per_shot")
    record.add_argument("inputs", nargs="+", type=Path)
    diff = commands.add_parser("compare", help="compare two numerical archives")
    diff.add_argument("reference", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument("--rtol", type=float, default=0)
    diff.add_argument("--atol", type=float, default=0)
    args = parser.parse_args(argv)
    if args.command == "capture":
        snapshot = capture(args.config, args.inputs, args.mode, backend=args.backend)
        save(snapshot, args.output)
        print(f"Captured {len(snapshot.results)} results to {args.output}")
        return 0
    differences = compare(
        load(args.reference), load(args.candidate), rtol=args.rtol, atol=args.atol
    )
    print("\n".join(differences) if differences else "Numerical baselines match")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
