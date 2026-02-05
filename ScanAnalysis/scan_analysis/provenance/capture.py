"""
Utilities for capturing code version and environment information.

These functions automatically detect git repository state, package versions,
and other environment details for provenance logging.
"""

import logging
import subprocess
from pathlib import Path

from scan_analysis.provenance.models import CodeVersion

logger = logging.getLogger(__name__)

# Default packages to track for reproducibility
DEFAULT_TRACKED_PACKAGES = [
    "scan_analysis",
    "image_analysis",
    "geecs_data_utils",
    "numpy",
    "scipy",
    "pandas",
    "pydantic",
]


def capture_code_version(repo_path: Path | None = None) -> CodeVersion | None:
    """
    Capture git repository information for reproducibility.

    Args:
        repo_path: Path to the git repository. If None, uses the directory
                   containing this file (scan_analysis package).

    Returns
    -------
        CodeVersion object with repository info, or None if not a git repo.
    """
    if repo_path is None:
        # Default to the GEECS-Plugins repo (parent of ScanAnalysis)
        repo_path = Path(__file__).parent.parent.parent.parent

    try:
        # Check if this is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
        )
        if result.returncode != 0:
            logger.debug(f"Not a git repository: {repo_path}")
            return None

        # Get remote URL
        repository = _get_git_remote(repo_path)

        # Get commit hash
        commit = _get_git_commit(repo_path)

        # Get branch name
        branch = _get_git_branch(repo_path)

        # Check for uncommitted changes
        dirty = _is_git_dirty(repo_path)

        return CodeVersion(
            repository=repository,
            commit=commit,
            branch=branch,
            dirty=dirty,
        )

    except Exception as e:
        logger.warning(f"Failed to capture git information: {e}")
        return None


def _get_git_remote(repo_path: Path) -> str | None:
    """Get the origin remote URL."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_git_commit(repo_path: Path) -> str | None:
    """Get the current commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_git_branch(repo_path: Path) -> str | None:
    """Get the current branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            # HEAD means detached head state
            return branch if branch != "HEAD" else None
    except Exception:
        pass
    return None


def _is_git_dirty(repo_path: Path) -> bool:
    """Check if there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    # Assume dirty if we can't check
    return True


def capture_dependencies(packages: list[str] | None = None) -> dict[str, str]:
    """
    Capture versions of key Python packages.

    Args:
        packages: List of package names to capture. If None, uses DEFAULT_TRACKED_PACKAGES.

    Returns
    -------
        Dictionary mapping package names to version strings.
    """
    from importlib.metadata import PackageNotFoundError, version

    if packages is None:
        packages = DEFAULT_TRACKED_PACKAGES

    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            # Package not installed, skip it
            pass
        except Exception as e:
            logger.debug(f"Could not get version for {pkg}: {e}")

    return versions


def get_current_user() -> str | None:
    """Get the current username for provenance logging."""
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return None


def extract_config_from_analyzer(analyzer) -> dict | None:
    """
    Try to extract configuration from an analyzer using various patterns.

    This function attempts multiple patterns to find and serialize configuration
    from an analyzer object. It supports Pydantic models, plain dicts, and
    callable `get_config()` methods.

    Args:
        analyzer: An analyzer object that may have configuration attached.

    Returns
    -------
        A dictionary containing the serialized configuration, or None if no
        configuration could be extracted.

    Notes
    -----
    Supported patterns (tried in order):
    1. `analyzer.camera_config` - Pydantic model (common in ImageAnalyzer subclasses)
    2. `analyzer.config` - Generic config attribute (Pydantic or dict)
    3. `analyzer.get_config()` - Callable that returns config dict
    """
    config_dict = None

    # Pattern 1: camera_config (common in ImageAnalyzer subclasses)
    if hasattr(analyzer, "camera_config"):
        config = analyzer.camera_config
        if config is not None:
            if hasattr(config, "model_dump"):
                try:
                    config_dict = config.model_dump(mode="json")
                except Exception as e:
                    logger.debug(f"Failed to dump camera_config: {e}")
            elif isinstance(config, dict):
                config_dict = config

    # Pattern 2: Generic config attribute
    if config_dict is None and hasattr(analyzer, "config"):
        config = analyzer.config
        if config is not None:
            if hasattr(config, "model_dump"):
                try:
                    config_dict = config.model_dump(mode="json")
                except Exception as e:
                    logger.debug(f"Failed to dump config: {e}")
            elif isinstance(config, dict):
                config_dict = config

    # Pattern 3: get_config() method
    if config_dict is None and hasattr(analyzer, "get_config"):
        if callable(analyzer.get_config):
            try:
                config_dict = analyzer.get_config()
            except Exception as e:
                logger.debug(f"Failed to call get_config(): {e}")

    return config_dict
