r"""``geecs-wavekit-doctor`` — check, prepare and self-test a host's WaveKit install.

The ``haso`` measure runs Imagine Optic's WaveKit SDK out of process
(:mod:`image_analysis.algorithms.haso_wavekit`). This command walks the
install the way the runbook (``docs/analysis/haso.md``) describes it and
says what is missing, in order:

1. the ``[Paths] wavekit_*`` keys of ``config.ini`` and the share tree
   they point at (``wavekit_py/``, ``dlls/x64/``, the Windows Python with
   its numpy, the sensor configurations);
2. on Linux, the launcher: Wine's version (pinned to the one the port
   was verified on — a newer Wine may fix ``fetestexcept`` and free the
   numpy pin, but nothing here has been run on one) and the 64-bit
   prefix it names, created if absent, with the common application-data
   directory the engine extracts its per-sensor DLL into;
3. the licence-free self-test: the SDK's own HASO3 sample (config and
   image under ``Examples/DATAS``) through the worker — every install can
   run it, lab files or not;
4. the golden reference: every ``reference/<set>/`` beside the configs
   that holds a ``reference.json`` (the settings that produced its
   sidecars) is recomputed and compared in float32 with the Windows
   results — raw phase, processed phase and intensity must be exactly
   equal.

Nothing here touches a scan folder or the repository. The share's
``Examples/OUT_FILES`` is not written either: the sample runs through the
worker into a temporary directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from geecs_data_utils.io.himg import read_himg
from image_analysis.algorithms.haso_wavekit import CONFIG_KEYS, HasoWaveKit

__all__ = ["WINE_VERSION", "Report", "compare_reference_set", "main", "wine_prefix"]

#: The Wine release the port was verified on (bit-identical to Windows on
#: 26_0310 Scan012). Its C runtime lacks ``fetestexcept``, which is why the
#: SDK's Python 3.8 + numpy 1.19 pair is used; another release is a warning.
WINE_VERSION = "6.0.3"

#: Where the engine writes the per-sensor DLL it extracts from a ``.dat``:
#: the prefix's common application-data folder, which Wine places under
#: ``ProgramData`` (a Windows 7+ prefix) or ``users/Public/Application
#: Data`` (an XP-style one). Both are created; the engine fails with
#: "cannot open file for writing" when its one is missing.
CORE_ENGINE_DIRS = (
    Path("drive_c") / "ProgramData" / "Imagine Optic" / "Core engine",
    Path("drive_c")
    / "users"
    / "Public"
    / "Application Data"
    / "Imagine Optic"
    / "Core engine",
)

#: The SDK's licence-free sample: a HASO3 configuration and one image.
SAMPLE_CONFIG = Path("Examples") / "DATAS" / "config_file_haso.dat"
SAMPLE_IMAGE = Path("Examples") / "DATAS" / "data_image.himg"

#: The reference products, ``<stem>_<name>.tsv`` beside each ``<stem>.himg``.
REFERENCE_PRODUCTS = {
    "raw": "raw_phase",
    "postprocessed": "processed_phase",
    "intensity": "intensity",
}


@dataclass
class Report:
    """What the doctor found: lines of ``[ OK ]`` / ``[WARN]`` / ``[FAIL]``."""

    lines: list[str] = field(default_factory=list)
    failures: int = 0
    warnings: int = 0

    def ok(self, text: str) -> None:
        """Record a passed check."""
        self.lines.append(f"[ OK ] {text}")

    def warn(self, text: str) -> None:
        """Record a finding that does not stop the install from working."""
        self.warnings += 1
        self.lines.append(f"[WARN] {text}")

    def fail(self, text: str) -> None:
        """Record a finding that does."""
        self.failures += 1
        self.lines.append(f"[FAIL] {text}")


def wine_prefix(launcher: Sequence[str]) -> Optional[Path]:
    """The ``WINEPREFIX`` a launcher sets (``env WINEPREFIX=... wine``), else Wine's default.

    ``None`` when the launcher does not run Wine at all (a Windows host).
    """
    tokens = list(launcher)
    if not any(Path(t).name.startswith("wine") for t in tokens):
        return None
    for token in tokens:
        if token.startswith("WINEPREFIX="):
            return Path(token[len("WINEPREFIX=") :]).expanduser()
    return Path(os.environ.get("WINEPREFIX", "~/.wine")).expanduser()


def _wine_version(launcher: Sequence[str]) -> Optional[str]:
    """``wine --version``'s release (``6.0.3``), or ``None`` when it cannot run."""
    wine = next((t for t in launcher if Path(t).name.startswith("wine")), None)
    if wine is None:
        return None
    try:
        out = subprocess.run(
            [wine, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = out.stdout.strip()  # "wine-6.0.3 (Ubuntu 6.0.3~repack-1)"
    return text.split()[0].removeprefix("wine-") if text else None


def _check_config(report: Report) -> Optional[dict]:
    from geecs_data_utils import GeecsPathsConfig

    config = GeecsPathsConfig()
    values = {key: getattr(config, key, None) for key in CONFIG_KEYS}
    missing = [k for k in CONFIG_KEYS if k != "wavekit_launcher" and values[k] is None]
    if missing:
        report.fail(
            f"config.ini [Paths] {', '.join(missing)}: not set, or the path does not exist"
        )
        return None
    report.ok(
        "config.ini names the SDK, the Windows Python and the sensor configurations"
    )
    sdk, python, configs = (
        Path(values["wavekit_sdk_path"]),
        Path(values["wavekit_python_path"]),
        Path(values["wavekit_configs_path"]),
    )
    for path, what in (
        (sdk / "wavekit_py" / "__init__.py", "the SDK's Python bindings"),
        (sdk / "dlls" / "x64", "the SDK's 64-bit DLLs"),
        (python, "the Windows Python"),
        (python.parent / "numpy", "numpy inside the Windows Python"),
        (sdk / SAMPLE_CONFIG, "the vendor sample configuration"),
        (sdk / SAMPLE_IMAGE, "the vendor sample image"),
    ):
        if path.exists():
            report.ok(f"{what}: {path}")
        else:
            report.fail(f"{what} missing: {path}")
    sensors = sorted(p.name for p in configs.glob("*.dat"))
    if sensors:
        report.ok(f"sensor configurations in {configs}: {', '.join(sensors)}")
    else:
        report.fail(f"no sensor .dat in {configs}")
    launcher = tuple(shlex.split(values["wavekit_launcher"] or ""))
    if launcher:
        report.ok(f"launcher: {' '.join(launcher)}")
    elif platform.system() != "Windows":
        report.fail(
            "wavekit_launcher is not set: on Linux the Windows Python needs one "
            "(env WINEDEBUG=-all WINEPREFIX=<64-bit prefix> wine)"
        )
    else:
        report.ok("no launcher: the Windows Python runs directly")
    return {"sdk": sdk, "python": python, "configs": configs, "launcher": launcher}


def _check_wine(report: Report, launcher: Sequence[str], *, create: bool) -> None:
    prefix = wine_prefix(launcher)
    if prefix is None:
        return
    version = _wine_version(launcher)
    if version is None:
        report.fail("wine does not run (is wine64 installed and on PATH?)")
        return
    if version == WINE_VERSION:
        report.ok(f"wine {version} (the verified release)")
    else:
        report.warn(
            f"wine {version}: the port was verified on {WINE_VERSION} only (its C "
            "runtime lacks fetestexcept, hence the SDK's Python 3.8 + numpy 1.19); "
            "run the reference check below before trusting this release"
        )
    system = prefix / "system.reg"
    if not system.exists():
        if not create:
            report.fail(
                f"Wine prefix {prefix} does not exist (run without --no-create)"
            )
            return
        env = {
            **os.environ,
            "WINEPREFIX": str(prefix),
            "WINEARCH": "win64",
            "WINEDEBUG": "-all",
        }
        try:
            subprocess.run(
                ["wineboot", "-u"],
                env=env,
                capture_output=True,
                timeout=600,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            report.fail(f"could not create the Wine prefix {prefix}: {exc}")
            return
        report.ok(f"created the 64-bit Wine prefix {prefix}")
    else:
        report.ok(f"Wine prefix {prefix}")
    arch = ""
    try:
        arch = next(
            (
                line
                for line in system.read_text(errors="replace").splitlines()
                if "#arch=" in line
            ),
            "",
        )
    except OSError:
        pass
    if arch and "win64" not in arch:
        report.fail(
            f"{prefix} is not a 64-bit prefix ({arch.strip()}): WaveKit's DLLs are x64"
        )
    for engine_dir in (prefix / d for d in CORE_ENGINE_DIRS):
        if engine_dir.is_dir():
            report.ok(f"engine directory {engine_dir}")
        elif create:
            engine_dir.mkdir(parents=True, exist_ok=True)
            report.ok(f"created the engine directory {engine_dir}")
        else:
            report.fail(
                f"engine directory missing: {engine_dir} (the engine cannot extract its DLL)"
            )


def _sample_test(report: Report, paths: dict) -> None:
    sdk = paths["sdk"]
    try:
        header, pixels = read_himg(sdk / SAMPLE_IMAGE)
        engine = HasoWaveKit(
            sdk,
            paths["python"],
            (sdk / SAMPLE_CONFIG).parent,
            header,
            launcher=paths["launcher"],
        )
        result = engine.compute(
            pixels,
            sensor_config=SAMPLE_CONFIG.name,
            lift=False,
            start_subpupil=(16, 16),
            denoising_strength=1.0,
        )
    except Exception as exc:  # noqa: BLE001 — every failure is a finding here
        report.fail(f"vendor sample (licence-free) failed: {exc}")
        return
    report.ok(
        f"vendor sample: HASO serial {result.image_serial}, slopes grid "
        f"{result.processed_phase.shape}, {result.timings.get('written', '?')} s"
    )


def _read_tsv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t", dtype=np.float32, ndmin=2)


def compare_reference_set(
    directory: Path, engine: HasoWaveKit, settings: dict
) -> list[tuple[str, str, bool, str]]:
    """Recompute every ``<stem>.himg`` of a reference set and compare its products.

    Returns ``(stem, product, equal, detail)`` per comparison; equality is
    ``np.array_equal`` in float32 with NaN equal to NaN.
    """
    mask = settings.get("mask")
    filters = settings.get("filters", {})
    flags = (
        filters.get("tilt_x", True),
        filters.get("tilt_y", True),
        filters.get("curvature", True),
        filters.get("astigmatism_0", True),
        filters.get("astigmatism_45", True),
        filters.get("others", False),
    )
    rows = []
    for image in sorted(directory.glob("*.himg")):
        stem = image.stem
        _, pixels = read_himg(image)
        result = engine.compute(
            pixels,
            sensor_config=settings["sensor_config"],
            mask=None
            if mask is None
            else (mask["top"], mask["bottom"], mask["left"], mask["right"]),
            filters=flags,
            wavelength_nm=settings.get("wavelength_nm", 800.0),
            start_subpupil=tuple(settings.get("start_subpupil", (87, 64))),
            zonal_prefs=tuple(settings.get("zonal_prefs", (100, 500, 1e-6))),
        )
        for suffix, attribute in REFERENCE_PRODUCTS.items():
            expected_path = directory / f"{stem}_{suffix}.tsv"
            if not expected_path.is_file():
                rows.append((stem, suffix, False, f"{expected_path.name} missing"))
                continue
            expected = _read_tsv(expected_path)
            got = np.asarray(getattr(result, attribute), dtype=np.float32)
            if got.shape != expected.shape:
                rows.append(
                    (stem, suffix, False, f"shape {got.shape} != {expected.shape}")
                )
                continue
            equal = np.array_equal(got, expected, equal_nan=True)
            differing = int(
                np.sum(~((got == expected) | (np.isnan(got) & np.isnan(expected))))
            )
            rows.append(
                (stem, suffix, equal, "" if equal else f"{differing} values differ")
            )
    return rows


def _reference_test(report: Report, paths: dict, reference: Path) -> None:
    sets = sorted(p for p in reference.glob("*/reference.json"))
    if not sets:
        report.warn(
            f"no reference set (a reference.json under {reference}); parity not checked"
        )
        return
    for settings_path in sets:
        directory = settings_path.parent
        settings = json.loads(settings_path.read_text())
        images = sorted(directory.glob("*.himg"))
        if not images:
            report.fail(f"{directory.name}: no .himg to recompute")
            continue
        try:
            header, _ = read_himg(images[0])
            engine = HasoWaveKit(
                paths["sdk"],
                paths["python"],
                paths["configs"],
                header,
                launcher=paths["launcher"],
            )
            rows = compare_reference_set(directory, engine, settings)
        except Exception as exc:  # noqa: BLE001 — every failure is a finding here
            report.fail(f"{directory.name}: {exc}")
            continue
        bad = [r for r in rows if not r[2]]
        if bad:
            for stem, product, _, detail in bad:
                report.fail(f"{directory.name}/{stem} {product}: {detail}")
        else:
            report.ok(
                f"{directory.name}: {len(images)} shots, raw phase / processed phase / "
                "intensity float32-exact against the Windows results"
            )


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        prog="geecs-wavekit-doctor",
        description="Check, prepare and self-test this host's WaveKit (HASO) install.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Directory of reference sets (each a folder with reference.json, .himg "
            "files and their Windows *_raw/_postprocessed/_intensity.tsv); default: "
            "the 'reference' folder beside wavekit_configs_path."
        ),
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Do not recompute the reference sets.",
    )
    parser.add_argument(
        "--skip-sample",
        action="store_true",
        help="Do not run the vendor's licence-free sample.",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Only report a missing Wine prefix or engine directory; never create them.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every check; exit 1 on any failure."""
    args = build_parser().parse_args(argv)
    report = Report()
    paths = _check_config(report)
    if paths is not None:
        _check_wine(report, paths["launcher"], create=not args.no_create)
        if report.failures == 0 and not args.skip_sample:
            _sample_test(report, paths)
        if report.failures == 0 and not args.skip_reference:
            reference = args.reference or paths["configs"].parent / "reference"
            _reference_test(report, paths, reference)
    for line in report.lines:
        print(line)
    print(
        f"{'FAIL' if report.failures else 'OK'}: {report.failures} failure(s), "
        f"{report.warnings} warning(s)"
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
