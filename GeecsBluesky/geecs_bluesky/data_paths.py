r"""Local ↔ device-server data path mapping and asset roots.

GEECS device servers see the shared data root at a Windows path (typically
``Z:\data``); the machine running scans sees it at a local mount.  These
helpers translate scanner-owned save paths into the form devices need for
``localsavingpath``, and resolve the canonical/local roots used by external
asset Resource documents.  Config comes from
``~/.config/geecs_python_api/config.ini`` ``[Paths]``.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path, PureWindowsPath

logger = logging.getLogger(__name__)


CONFIG_PATH = Path.home() / ".config" / "geecs_python_api" / "config.ini"


def read_config_entry(
    section: str, key: str, config_path: Path | None = None
) -> str | None:
    """One ``config.ini`` value (``None`` when the file, section or key is absent)."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if section not in cfg:
        return None
    return cfg[section].get(key) or None


def pva_addr_tokens(raw: str | None) -> list[str]:
    """The hosts of a ``[pva]`` address-list value: space- or comma-separated, in order, de-duplicated."""
    tokens: list[str] = []
    for token in (raw or "").replace(",", " ").split():
        if token not in tokens:
            tokens.append(token)
    return tokens


def _read_paths_entry(key: str) -> str | None:
    return read_config_entry("Paths", key)


def read_device_server_data_base_path() -> str | None:
    """Read the data base path visible from GEECS device-server hosts."""
    return _read_paths_entry("geecs_device_server_data_base_path")


def read_local_data_base_path() -> str | None:
    """Read the scanner-local path for the shared GEECS data root."""
    return _read_paths_entry("GEECS_DATA_LOCAL_BASE_PATH")


def translate_save_path_for_device_server(
    save_path: str | Path,
    *,
    local_base_path: str | Path,
    device_server_base_path: str,
) -> str:
    """Translate a local scan path to the path understood by GEECS devices."""
    local_path = Path(save_path)
    local_base = Path(local_base_path)
    try:
        relative_path = local_path.relative_to(local_base)
    except ValueError:
        logger.warning(
            "Native save path %s is not under local base path %s; using local path",
            local_path,
            local_base,
        )
        return str(local_path)
    return str(PureWindowsPath(device_server_base_path, *relative_path.parts))


def _translate_to(remote_base_path: str | None, save_path: str, who: str) -> str:
    """Translate a local scan path onto *remote_base_path* (the local path if unknown)."""
    if not remote_base_path:
        return save_path
    try:
        from geecs_data_utils import ScanPaths
    except Exception:
        logger.warning(
            "Could not import geecs_data_utils; using local path for %s", who
        )
        return save_path
    local_base_path = getattr(
        getattr(ScanPaths, "paths_config", None), "base_path", None
    )
    if local_base_path is None:
        logger.warning(
            "ScanPaths.paths_config is not loaded; using local path for %s", who
        )
        return save_path
    return translate_save_path_for_device_server(
        save_path,
        local_base_path=local_base_path,
        device_server_base_path=remote_base_path,
    )


def device_server_save_path(save_path: str) -> str:
    """Return the path to send to device ``localsavingpath`` controls."""
    return _translate_to(read_device_server_data_base_path(), save_path, "the device")


def asset_resource_root_paths() -> tuple[str | None, str | None]:
    """Return ``(canonical_root, local_root)`` for external asset docs."""
    canonical_root = read_device_server_data_base_path()
    if not canonical_root:
        return None, None

    local_root = read_local_data_base_path()
    if local_root:
        return canonical_root, local_root

    try:
        from geecs_data_utils import ScanPaths
    except Exception:
        logger.warning(
            "Could not import geecs_data_utils; using scan-folder asset roots"
        )
        return None, None

    paths_config = getattr(ScanPaths, "paths_config", None)
    base_path = getattr(paths_config, "base_path", None)
    if base_path is None:
        logger.warning(
            "ScanPaths.paths_config is not loaded; using scan-folder asset roots"
        )
        return None, None
    return canonical_root, str(base_path)


def read_plugin_data_base_path() -> str | None:
    """The data root as the camera servers' **file plugin** sees it.

    The plugin runs as a Windows service (GeecsPvaGateway ``DEPLOYMENT.md``,
    session-0 rule 1): it cannot see the per-user mapped drive LabVIEW writes
    through, so it needs the UNC form of the same root
    (``[Paths] geecs_pva_plugin_data_base_path``).  Absent, the device-server
    path is used — right where the service can see that drive.
    """
    return _read_paths_entry("geecs_pva_plugin_data_base_path")


def plugin_save_path(save_path: str) -> str:
    """Return the path to send to a file plugin's ``FilePath`` control."""
    return _translate_to(
        read_plugin_data_base_path() or read_device_server_data_base_path(),
        save_path,
        "the file plugin",
    )
