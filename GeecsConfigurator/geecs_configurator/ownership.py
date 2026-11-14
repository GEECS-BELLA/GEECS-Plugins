"""Which paths of the configs repo this services box owns: one configs repo, many facilities.

Several facilities share one configs repo and one GitHub App, each with its
own services box. A box **owns** the experiments named in
``GEECS_CONFIGURATOR_EXPERIMENTS`` (``scanner_configs/experiments/<name>/``)
and the facility folders named in ``GEECS_CONFIGURATOR_ANALYSIS_FOLDERS``
inside the shared analysis tree (``scan_analysis_configs/<kind folder>/<name>/``,
where the tree keeps one folder per facility). Everything else belongs to
another box, or to nobody (the maintainer's pull requests):

- **writes**: a save, a proposal (the maintainer tier's branch) and an inbox
  publish are made only at an owned path; anything else is refused before a
  worktree is staged (the save's 403), and the repository refuses it again
  at the commit;
- **the gate** judges only the owned documents a commit changes: another
  facility's commit, valid or not, fast-forwards the live checkout without
  judgement (that facility's box and the configs repo's CI judge it), and
  never locks this box's operators out;
- **the log** holds rows for owned paths only, so the apply seam never sees
  another facility's document.

A path is owned by its name below the root (git's spelling: ``/``-separated,
no ``.`` or ``..``) and, on disk, only when it also resolves inside the owned
folder: a symbolic link from an owned folder to another facility's file is not
an owned file. No owned path is ever implied: an empty setting owns nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from geecs_schemas import KIND_STORAGE, FileStorage
from geecs_schemas.storage import STORAGE_ROOT_PATHS

#: ``scanner_configs/experiments``: the folder holding one folder per experiment.
EXPERIMENTS_ROOT = STORAGE_ROOT_PATHS["experiment"].split("/<experiment>")[0]
#: The shared analysis tree's kind folders (``scan_analysis_configs/analyzers``, ...), each split per facility.
ANALYSIS_KIND_FOLDERS = tuple(sorted({
    f"{STORAGE_ROOT_PATHS['analysis']}/{s.folder}" for s in KIND_STORAGE.values()
    if isinstance(s, FileStorage) and s.root == "analysis" and s.folder
}))  # fmt: skip
_PATTERN_CHARS = set("*?[]")


class OwnershipError(ValueError):
    """An owned name that is not one folder's plain name."""


def parse_names(raw: str, key: str) -> tuple[str, ...]:
    """A comma-separated setting as folder names: blanks dropped, duplicates once; refuses a path or a pattern."""
    out: list[str] = []
    for name in (n.strip() for n in raw.split(",")):
        if not name:
            continue
        if "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
            raise OwnershipError(f"{key}: {name!r} is not a folder's name")
        if _PATTERN_CHARS & set(name):
            raise OwnershipError(f"{key}: {name!r} is a pattern; name each folder")
        if name not in out:
            out.append(name)
    return tuple(out)


def _plain(rel: str) -> PurePosixPath | None:
    """``rel`` as a relative path with no ``.``/``..``/empty part, else ``None``."""
    if not rel or rel.startswith("/") or "\\" in rel:
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    return PurePosixPath(rel)


@dataclass(frozen=True)
class Ownership:
    """The experiments and analysis facility folders one services box owns."""

    experiments: tuple[str, ...] = ()
    analysis_folders: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """Whether nothing is owned."""
        return not (self.experiments or self.analysis_folders)

    def prefixes(self) -> tuple[str, ...]:
        """Every owned folder below the root (``scanner_configs/experiments/<name>``, ``scan_analysis_configs/analyzers/<name>``, ...)."""
        own = [f"{EXPERIMENTS_ROOT}/{e}" for e in self.experiments]
        own += [
            f"{k}/{f}" for k in ANALYSIS_KIND_FOLDERS for f in self.analysis_folders
        ]
        return tuple(own)

    def _prefix_of(self, rel: str) -> str | None:
        path = _plain(rel)
        if path is None:
            return None
        for prefix in self.prefixes():
            if path.is_relative_to(prefix) and path != PurePosixPath(prefix):
                return prefix
        return None

    def owns(self, rel: str) -> bool:
        """Whether the path (below the root, git's spelling) is under an owned folder, by name alone."""
        return self._prefix_of(rel) is not None

    def owns_file(self, root: Path, rel: str) -> bool:
        """Whether ``rel`` is owned by name **and** resolves, on disk under ``root``, inside the same owned folder."""
        prefix = self._prefix_of(rel)
        if prefix is None:
            return False
        folder = (root / prefix).resolve()
        return (root / rel).resolve().is_relative_to(folder)

    def owned(self, paths: Iterable[str]) -> list[str]:
        """The owned paths among ``paths``, in order."""
        return [p for p in paths if self.owns(p)]

    def why_not(self, rel: str) -> str:
        """The refusal's words for a path this box does not own."""
        return f"{rel} is not this services box's to change: it owns {self.describe()}; change it from the box that owns it, or by pull request"  # fmt: skip

    def describe(self) -> str:
        """The owned set in words (``/healthz``, the page, a refusal)."""
        parts = []
        if self.experiments:
            parts.append("experiments " + ", ".join(self.experiments))
        if self.analysis_folders:
            parts.append("analysis folders " + ", ".join(self.analysis_folders))
        return "; ".join(parts) or "nothing"
