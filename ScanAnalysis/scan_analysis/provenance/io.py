"""
I/O operations for provenance files with file locking support.

This module provides thread/process-safe read and write operations
for provenance files following the Analysis Provenance Standard v0.1.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from scan_analysis.provenance.capture import (
    capture_code_version,
    capture_dependencies,
    get_current_user,
)
from scan_analysis.provenance.models import (
    AnalysisEntry,
    CodeVersion,
    ProvenanceFile,
    Software,
)

logger = logging.getLogger(__name__)

# Default timeout for acquiring file lock (seconds)
DEFAULT_LOCK_TIMEOUT = 60


def get_provenance_path(data_file: str | Path) -> Path:
    """
    Get the provenance file path for a given data file.

    Args:
        data_file: Path to the data file (e.g., s123.txt)

    Returns
    -------
        Path to the corresponding provenance file (e.g., s123.provenance.json)
    """
    data_path = Path(data_file)
    # Remove all suffixes and add .provenance.json
    stem = data_path.stem
    # Handle files with multiple extensions like .tar.gz
    while "." in stem:
        stem = Path(stem).stem
    return data_path.parent / f"{stem}.provenance.json"


def read_provenance(data_file: str | Path) -> ProvenanceFile | None:
    """
    Read provenance information for a data file.

    Args:
        data_file: Path to the data file

    Returns
    -------
        ProvenanceFile object, or None if no provenance file exists
    """
    provenance_path = get_provenance_path(data_file)

    # Also check for .yaml extension
    yaml_path = provenance_path.with_suffix(".yaml")

    if provenance_path.exists():
        try:
            content = provenance_path.read_text(encoding="utf-8")
            data = json.loads(content)
            return ProvenanceFile.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to read provenance file {provenance_path}: {e}")
            return None
    elif yaml_path.exists():
        try:
            import yaml

            content = yaml_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            return ProvenanceFile.model_validate(data)
        except ImportError:
            logger.warning("PyYAML not installed, cannot read .yaml provenance file")
            return None
        except Exception as e:
            logger.warning(f"Failed to read provenance file {yaml_path}: {e}")
            return None

    return None


def log_provenance(
    data_file: str | Path,
    columns_written: list[str],
    *,
    software_name: str | None = None,
    software_version: str | None = None,
    code_version: CodeVersion | None = None,
    auto_capture_code: bool = True,
    dependencies: dict[str, str] | None = None,
    auto_capture_deps: bool = True,
    config: dict[str, Any] | None = None,
    config_ref: str | None = None,
    notes: str | None = None,
    user: str | None = None,
    auto_capture_user: bool = True,
    lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
) -> bool:
    """
    Log a provenance entry for analysis that wrote to a data file.

    This function is thread/process-safe and uses file locking to prevent
    concurrent write conflicts.

    Args:
        data_file: Path to the data file that was written to
        columns_written: List of column names that were added or modified

    Keyword Args:
        software_name: Name of the software/tool
        software_version: Version string
        code_version: Pre-captured CodeVersion object
        auto_capture_code: If True and code_version is None, auto-capture git info
        dependencies: Dictionary of package versions
        auto_capture_deps: If True and dependencies is None, auto-capture package versions
        config: Configuration dictionary
        config_ref: Path to external configuration file
        notes: Human-readable notes
        user: Username (auto-captured if None and auto_capture_user is True)
        auto_capture_user: If True and user is None, auto-capture username
        lock_timeout: Seconds to wait for file lock

    Returns
    -------
        True if provenance was logged successfully, False otherwise
    """
    if not columns_written:
        logger.warning("log_provenance called with empty columns_written, skipping")
        return False

    provenance_path = get_provenance_path(data_file)
    lock_path = provenance_path.with_suffix(".provenance.json.lock")

    # Build the entry
    software = None
    if software_name:
        software = Software(name=software_name, version=software_version)

    # Auto-capture code version if requested
    if code_version is None and auto_capture_code:
        code_version = capture_code_version()

    # Auto-capture dependencies if requested
    if dependencies is None and auto_capture_deps:
        deps = capture_dependencies()
        if deps:
            dependencies = deps

    # Auto-capture user if requested
    if user is None and auto_capture_user:
        user = get_current_user()

    entry = AnalysisEntry(
        timestamp=datetime.now(timezone.utc),
        columns_written=columns_written,
        software=software,
        code_version=code_version,
        dependencies=dependencies,
        config=config,
        config_ref=config_ref,
        notes=notes,
        user=user,
    )

    # Use file locking for concurrent access safety
    lock = FileLock(lock_path, timeout=lock_timeout)

    try:
        with lock:
            # Read existing provenance or create new
            if provenance_path.exists():
                try:
                    content = provenance_path.read_text(encoding="utf-8")
                    data = json.loads(content)
                    provenance = ProvenanceFile.model_validate(data)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse existing provenance, creating new: {e}"
                    )
                    provenance = ProvenanceFile()
            else:
                provenance = ProvenanceFile()

            # Append the new entry
            provenance.append_entry(entry)

            # Write back
            output = provenance.model_dump(mode="json", exclude_none=True)
            provenance_path.write_text(
                json.dumps(output, indent=2, default=str), encoding="utf-8"
            )

            logger.debug(
                f"Logged provenance for {len(columns_written)} columns to {provenance_path}"
            )
            return True

    except Timeout:
        logger.warning(
            f"Could not acquire provenance file lock after {lock_timeout}s: {lock_path}"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to log provenance: {e}")
        return False


def log_provenance_for_scan_analyzer(
    data_file: str | Path,
    columns_written: list[str],
    analyzer_config: Any | None = None,
    image_analyzer_config: Any | None = None,
    notes: str | None = None,
) -> bool:
    """
    Convenience function for logging provenance from scan analyzers.

    This captures the standard information needed for scan_analysis provenance.

    Args:
        data_file: Path to the sfile
        columns_written: Columns that were written
        analyzer_config: The scan analyzer's config (Pydantic model or dict)
        image_analyzer_config: The image analyzer's config (if applicable)
        notes: Optional notes

    Returns
    -------
        True if successful
    """
    from importlib.metadata import PackageNotFoundError, version

    # Get scan_analysis version
    try:
        software_version = version("scan_analysis")
    except PackageNotFoundError:
        software_version = "unknown"

    # Build config dict
    config = {}
    if analyzer_config is not None:
        if hasattr(analyzer_config, "model_dump"):
            config["analyzer"] = analyzer_config.model_dump(mode="json")
        elif isinstance(analyzer_config, dict):
            config["analyzer"] = analyzer_config

    if image_analyzer_config is not None:
        if hasattr(image_analyzer_config, "model_dump"):
            config["image_analyzer"] = image_analyzer_config.model_dump(mode="json")
        elif isinstance(image_analyzer_config, dict):
            config["image_analyzer"] = image_analyzer_config

    return log_provenance(
        data_file=data_file,
        columns_written=columns_written,
        software_name="scan_analysis",
        software_version=software_version,
        config=config if config else None,
        notes=notes,
    )
