"""Scaffold a new experiment's configs-repo folder from empty documents.

``deploy/experiment.sh up`` calls this for an experiment the configs repo
has no folder for yet::

    python -m geecs_bluesky.config_scaffold EXPERIMENT [--experiments-root DIR]

It creates ``<experiments root>/<EXPERIMENT>/`` with the folders
:class:`~geecs_bluesky.config_resolver.ConfigsRepoResolver` reads (presets,
trigger profiles, the scan-variable catalog) and two empty, valid documents
— an empty :class:`~geecs_schemas.ScanVariables` catalog and the default
:class:`~geecs_schemas.ExperimentDefaults` — so the worker and the scanner
open over it and every list is empty rather than an error. An existing
folder is never touched (it is someone's configs), and the experiments root
must already exist: nothing above the new folder is created.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_core.configs_repo import scanner_configs_base
from geecs_schemas import ExperimentDefaults, ScanVariables


def scaffold_experiment(
    experiment: str, experiments_root: Path | None = None
) -> list[Path]:
    """Create *experiment*'s folder with empty documents; ``[]`` when it exists.

    Parameters
    ----------
    experiment:
        The DB experiment name — the folder name the resolver looks up.
    experiments_root:
        ``scanner_configs/experiments``; the configured one by default.

    Returns
    -------
    list of Path
        Every directory and file created, in creation order.

    Raises
    ------
    FileNotFoundError
        When the experiments root cannot be resolved or does not exist.
    ValueError
        When *experiment* is empty or not a single path component.
    """
    if experiment.strip() in ("", ".", "..") or Path(experiment).name != experiment:
        raise ValueError(f"not a single folder name: {experiment!r}")
    root = experiments_root if experiments_root is not None else scanner_configs_base()
    if root is None or not root.is_dir():
        raise FileNotFoundError(f"no configs experiments root: {root}")
    target = root / experiment
    if target.exists():
        return []
    created = [target]
    target.mkdir()  # never parents=True: the root is someone else's
    for folder in (
        ConfigsRepoResolver.PRESET_FOLDER,
        ConfigsRepoResolver.TRIGGER_FOLDER,
        ConfigsRepoResolver.SCAN_VARIABLES_FOLDER,
    ):
        (target / folder).mkdir()
        created.append(target / folder)
    documents = {
        target
        / ConfigsRepoResolver.SCAN_VARIABLES_FOLDER
        / "scan_variables.yaml": ScanVariables(variables={}),
        target / "experiment_defaults.yaml": ExperimentDefaults(),
    }
    for path, document in documents.items():
        path.write_text(
            yaml.safe_dump(document.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        created.append(path)
    return created


def main(argv: Sequence[str] | None = None) -> int:
    """Scaffold the named experiment's folder; print what was created."""
    parser = argparse.ArgumentParser(
        prog="python -m geecs_bluesky.config_scaffold",
        description="Create an experiment's configs-repo folder from empty documents.",
    )
    parser.add_argument("experiment", help="the DB experiment name")
    parser.add_argument("--experiments-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        created = scaffold_experiment(args.experiment, args.experiments_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"configs scaffold: {exc}", file=sys.stderr)
        return 2
    if not created:
        print(
            f"configs scaffold: {args.experiment!r} already has a folder — left as it is"
        )
    for path in created:
        print(f"created {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
