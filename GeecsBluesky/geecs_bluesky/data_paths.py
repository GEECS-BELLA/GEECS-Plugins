r"""Local ↔ device-server data path mapping.

GEECS device servers see the shared data root at a Windows path (typically
``Z:\data``); the machine running scans sees it at a local mount.  These
helpers translate scanner-owned save paths into the form devices need for
``localsavingpath`` and for a file plugin's ``FilePath``.  Config comes from
``~/.config/geecs_python_api/config.ini`` ``[Paths]``.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath

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


def _local_base_path() -> str | None:
    """The data root as this host mounts it (``ScanPaths.paths_config``), or ``None``.

    ``GeecsPathsConfig`` records no base path when the share is not mounted
    at load, and ``ScanPaths`` loads it once at import — so a long-lived
    process that started before the mount (a service ordered only on the
    network) re-reads the config once per ask until a root appears, and
    heals without a restart.
    """
    try:
        from geecs_data_utils import ScanPaths
    except Exception:
        logger.warning(
            "Could not import geecs_data_utils; the local data root is unknown"
        )
        return None
    base = getattr(getattr(ScanPaths, "paths_config", None), "base_path", None)
    if base is None:
        try:
            ScanPaths.reload_paths_config()
        except Exception as exc:  # pragma: no cover - the loader logs its own error
            logger.debug("paths config reload: %s", exc)
        base = getattr(getattr(ScanPaths, "paths_config", None), "base_path", None)
    return base


def _translate_to(remote_base_path: str | None, save_path: str, who: str) -> str:
    """Translate a local scan path onto *remote_base_path* (the local path if unknown)."""
    if not remote_base_path:
        return save_path
    local_base_path = _local_base_path()
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


def read_tiled_host_data_base_path() -> str | None:
    """The data root as the **Tiled host** mounts it (``[Paths] geecs_tiled_host_data_base_path``).

    The Tiled writer registers each stream's Parquet table by the path the
    Tiled server reads it at (``geecs_bluesky.tiled_parquet``).  Absent means
    the Tiled host mounts the share where the writer does — the same box, or
    the same mount path — and the local path is the Tiled host's path.
    """
    return _read_paths_entry("geecs_tiled_host_data_base_path")


def translate_save_path_for_tiled_host(
    save_path: str | Path, *, local_base_path: str | Path, tiled_host_base_path: str
) -> str:
    """Translate a local scan path onto the Tiled host's (POSIX) mount of the share.

    Strict where :func:`translate_save_path_for_device_server` is lenient: a
    path outside the local data root has no Tiled-host form, and registering
    it by its local path would succeed at the catalog and fail at every read
    ("outside the readable storage area"), so it raises instead.
    """
    local_path = Path(save_path)
    try:
        relative = local_path.relative_to(Path(local_base_path))
    except ValueError as exc:
        raise ValueError(
            f"{local_path} is not under the local data root {local_base_path}; "
            "its Tiled-host path is unknown"
        ) from exc
    return PurePosixPath(tiled_host_base_path, *relative.parts).as_posix()


def tiled_host_path(save_path: str) -> str:
    """The path the Tiled host reads a file of the data share at.

    The local path when ``[Paths] geecs_tiled_host_data_base_path`` is unset.
    When it is set and the local data root is not known (the share was not
    mounted when ``geecs_data_utils`` loaded its config), this **refuses**
    rather than returning the local path: the registration would otherwise
    succeed and the table be unreadable from the Tiled host.
    """
    remote = read_tiled_host_data_base_path()
    if not remote:
        return save_path
    local_base_path = _local_base_path()
    if local_base_path is None:
        raise RuntimeError(
            "[Paths] geecs_tiled_host_data_base_path is set but the local data root "
            "is unknown (ScanPaths.paths_config not loaded) — refusing to register "
            f"{save_path} by a path the Tiled host cannot read"
        )
    return translate_save_path_for_tiled_host(
        save_path, local_base_path=local_base_path, tiled_host_base_path=remote
    )
