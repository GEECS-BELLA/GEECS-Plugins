"""A scan's per-shot analysis outputs: ``{day}/analysis/ScanNNN/<folder>/`` as a source.

An analysis run writes processed per-shot products beside the scan, under
the day's analysis tree: ``analysis/Scan075/py-magspec-interpSpec/`` holds
``Scan075_py-magspec-interpSpec_005.txt``, shot 5's spectrum. A folder may
hold several **series**, one per stem (``py-ebeam/`` holds ``py-ebeam_NNN.png``,
``py-ebeamX_NNN.txt`` and ``py-ebeamY_NNN.txt``). A series is a source like a
device folder: one file per shot, ``Scan{NNN}_{stem}_{shot}.{ext}``, numbered
the way a device's legacy files are (the row's shot, or the shot a
scan-mode run announced).

Its name, wherever a device name goes, is ``analysis:<folder>`` (the
folder's own series: the one named after the folder, else the first) or
``analysis:<folder>/<series>``. The name is checked against the analysis
folder's listing and then the output folder's, so a name from a URL can
never reach outside the scan's analysis folder.

Strictly read-only: a scan with no analysis folder has no outputs, and
nothing is ever created (the repo's scan-folder invariant).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from geecs_data_utils.scan_info import analysis_folder_for, parse_scan_folder_name

logger = logging.getLogger(__name__)

#: What starts an analysis output's name, so it is never a device folder's.
OUTPUT_PREFIX = "analysis:"

#: ``Scan075_py-magspec-interpSpec_005.txt``: scan, stem, shot, extension.
_SHOT_FILE = re.compile(
    r"^Scan(?P<scan>\d+)_(?P<stem>.+)_(?P<shot>\d+)\.(?P<ext>[A-Za-z0-9]+)$"
)

#: The image extensions a series' default prefers, when a stem has several.
_IMAGE_EXTS = ("png", "tif", "tiff")


@dataclass(frozen=True)
class OutputSeries:
    """One series of per-shot files in an analysis output folder."""

    #: The output folder: ``{day}/analysis/ScanNNN/<folder>``.
    folder: Path
    #: The stem between the scan and the shot number (``py-ebeamX``).
    stem: str
    #: The files' extension, lower case, without the dot.
    ext: str
    #: The scan number as the file names write it (``"075"``).
    scan_text: str
    #: The shot number's width in the file names (``3``: ``005``).
    width: int
    #: The series' full name (``analysis:py-ebeam/py-ebeamX``).
    name: str

    def shot_file(self, shot: int) -> Path:
        """The file that holds *shot* (it may not exist)."""
        return self.folder / (
            f"Scan{self.scan_text}_{self.stem}_{shot:0{self.width}d}.{self.ext}"
        )

    def files(self) -> list[Path]:
        """The series' files, sorted by name (one listing of its folder)."""
        return sorted(
            path
            for path in _listing(self.folder)
            if (found := _parse(path.name)) is not None
            and found[:2] == (self.stem, self.ext)
        )


def is_output_name(name: str) -> bool:
    """Whether *name* names an analysis output rather than a device folder."""
    return name.startswith(OUTPUT_PREFIX)


def output_label(name: str) -> str:
    """How a picker shows *name*: ``analysis · py-ebeam › py-ebeamX``; a device as itself."""
    if not is_output_name(name):
        return name
    folder, _, series = name[len(OUTPUT_PREFIX) :].partition("/")
    return f"analysis · {folder}" + (f" › {series}" if series else "")


def _analysis_folder(scan_folder: Path) -> Optional[Path]:
    """The scan's analysis folder (path only), or ``None`` for an odd scan folder."""
    try:
        return analysis_folder_for(scan_folder)
    except ValueError:
        return None


def _listing(folder: Path) -> list[Path]:
    """*folder*'s entries; empty when it is not there or cannot be listed."""
    try:
        return list(folder.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("cannot list %s: %s", folder, exc)
        return []


def output_dir(scan_folder: Path, name: str) -> Optional[Path]:
    """The output folder *name* names, by path alone: no listing, no check it exists.

    For a cheap stamp (its modification time); a reader resolves the name
    (:func:`resolve_output`). ``None`` for a name that is not an output or
    whose folder part is not one plain path part.
    """
    if not is_output_name(name):
        return None
    folder_name = name[len(OUTPUT_PREFIX) :].partition("/")[0]
    analysis = _analysis_folder(scan_folder)
    if analysis is None or folder_name in ("", ".", "..") or "\\" in folder_name:
        return None
    return analysis / folder_name


def output_folders(scan_folder: Path) -> list[str]:
    """The scan's analysis output folders as source names (one listing, sorted).

    Every subfolder of ``analysis/ScanNNN`` is named, whatever it holds:
    looking inside each would cost a listing per folder over the share, so
    the picked one is looked into (:func:`output_series`).
    """
    folder = _analysis_folder(scan_folder)
    if folder is None:
        return []
    return sorted(
        OUTPUT_PREFIX + entry.name for entry in _listing(folder) if entry.is_dir()
    )


def _parse(filename: str) -> Optional[tuple[str, str, str, int]]:
    """``(stem, ext, scan text, shot width)`` of a per-shot file name, or ``None``."""
    match = _SHOT_FILE.match(filename)
    if match is None:
        return None
    return (
        match.group("stem"),
        match.group("ext").lower(),
        match.group("scan"),
        len(match.group("shot")),
    )


def _series_in(folder: Path, scan_number: int, base: str) -> list[OutputSeries]:
    """The series of *folder*'s files for scan *scan_number*, by name."""
    seen: dict[tuple[str, str], Counter] = {}
    for entry in _listing(folder):
        found = _parse(entry.name)
        if found is None or int(found[2]) != scan_number:
            continue
        stem, ext, scan_text, width = found
        seen.setdefault((stem, ext), Counter())[(scan_text, width)] += 1
    exts: Counter = Counter(stem for stem, _ in seen)
    series = []
    for (stem, ext), shapes in sorted(seen.items()):
        (scan_text, width), _ = shapes.most_common(1)[0]
        label = stem if exts[stem] == 1 else f"{stem}.{ext}"
        series.append(
            OutputSeries(folder, stem, ext, scan_text, width, f"{base}/{label}")
        )
    return series


def output_series(scan_folder: Path, name: str) -> list[OutputSeries]:
    """The series in the output folder *name* names (its folder part), by name.

    One listing of the analysis folder (the guard: the folder must be one
    of its subfolders) and one of the output folder. Empty for a name that
    is not an output, a folder that is not there, or one with no per-shot
    files of this scan.
    """
    if not is_output_name(name):
        return []
    folder_name = name[len(OUTPUT_PREFIX) :].partition("/")[0]
    if not folder_name or folder_name not in {
        n[len(OUTPUT_PREFIX) :] for n in output_folders(scan_folder)
    }:
        return []
    scan_number = parse_scan_folder_name(Path(scan_folder).name)
    analysis = _analysis_folder(scan_folder)
    if scan_number is None or analysis is None:
        return []
    return _series_in(analysis / folder_name, scan_number, OUTPUT_PREFIX + folder_name)


def _default(series: list[OutputSeries], folder_name: str) -> OutputSeries:
    """A folder's own series: the one named after it (an image first), else the first."""
    own = [s for s in series if s.stem == folder_name]
    if own:
        images = [s for s in own if s.ext in _IMAGE_EXTS]
        return (images or own)[0]
    return series[0]


def resolve_output(scan_folder: Path, name: str) -> Optional[OutputSeries]:
    """The series *name* names in this scan, or ``None`` when it has none.

    ``analysis:<folder>`` is the folder's own series (:func:`_default`),
    ``analysis:<folder>/<series>`` that one.
    """
    series = output_series(scan_folder, name)
    if not series:
        return None
    folder_name, _, label = name[len(OUTPUT_PREFIX) :].partition("/")
    if not label:
        return _default(series, folder_name)
    for one in series:
        if one.name == name:
            return one
    return None
