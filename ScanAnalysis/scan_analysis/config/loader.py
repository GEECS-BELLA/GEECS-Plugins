"""Load one analysis document (by stem or path) from the analysis-configs tree.

The document models are GEECS-Schemas' (``load_analysis_document`` dispatches
on the file's ``kind`` / ``schema_version``) and the file lookup is
GEECS-Data-Utils' (``read_diagnostic`` over the ``analyzers/`` tree); this
module is where the two meet with the client ``config.ini`` default root,
which neither package may read (data-utils cannot import the schemas; the
schemas cannot read ``config.ini``). Every explicit single-diagnostic run —
the group loader, the data portal's Analysis tab, MCP's worker, a
notebook — loads through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from geecs_schemas.analysis import AnalysisDocument, load_analysis_document
from pydantic import ValidationError

from geecs_data_utils.analysis_configs import read_diagnostic

__all__ = ["load_diagnostic"]


def load_diagnostic(
    name_or_path: Union[str, Path],
    *,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AnalysisDocument:
    """Load an analysis document (a recipe, a diagnostic or a ``.himg`` job) by stem or path.

    Parameters
    ----------
    name_or_path : str or Path
        Diagnostic ID (filename stem, unique across the ``analyzers/``
        tree) or an explicit path.
    config_dir : Path, optional
        Root of the scan-analysis configs tree (the parent of
        ``analyzers/``); defaults to
        ``ScanPaths.paths_config.scan_analysis_configs_path``.
    overrides : dict, optional
        Deep-merged into the raw YAML before validation (nested dicts
        key-by-key, everything else replaced wholesale), so a consumer can
        run a per-call variant without forking the file. Override typos
        surface exactly like a bad YAML.

    Returns
    -------
    AnalysisDocument
        The validated, fully typed document, as its ``schema_version`` /
        ``kind`` says, with ``source_id`` set to the file stem.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    KeyError
        If a stem is not present under ``analyzers/``.
    ValueError
        On invalid YAML or validation errors, or when no root is available.
    """
    diag_path, data = read_diagnostic(
        name_or_path,
        config_dir=(
            None
            if isinstance(name_or_path, Path)
            else _resolve_default_config_dir(config_dir)
        ),
        overrides=overrides,
    )
    try:
        document = load_analysis_document(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid diagnostic config at {diag_path}: {exc}") from exc
    document._source_id = diag_path.stem
    return document


def _resolve_default_config_dir(config_dir: Optional[Path]) -> Path:
    """Return ``config_dir`` if set, else the client ``config.ini``'s configs root."""
    if config_dir is not None:
        return Path(config_dir)
    try:
        from geecs_data_utils import ScanPaths

        root = ScanPaths.paths_config.scan_analysis_configs_path
    except Exception as exc:
        raise ValueError(
            "config_dir was not provided and no default could be "
            "resolved from ScanPaths.paths_config."
        ) from exc
    if root is None:
        raise ValueError(
            "config_dir was not provided and "
            "ScanPaths.paths_config.scan_analysis_configs_path is unset."
        )
    return Path(root)
