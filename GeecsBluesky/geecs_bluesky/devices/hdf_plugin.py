"""The worker's side of the PVA gateway's file plugin (#806).

Three small things, everything else is stock ophyd-async:

- :class:`GeecsHdfIO` — ``NDFileHDF5IO`` plus the three GEECS PVs the plugin
  adds (``Rewind``, ``WriteStatus``, ``WriteMessage``).  The prefix is
  minted by :func:`geecs_core.pv_naming.hdf_plugin_prefix`, the one place
  both sides agree on it.
- :class:`PluginPathProvider` — the per-detector ``PathProvider`` the stock
  ``ADHDFDataLogic`` calls.  It ignores the datakey (ophyd's lowercase name)
  and asks the shared :class:`~geecs_bluesky.plans.claim_scan.GeecsScanPathProvider`
  for ``ScanNNN/<GEECS device>/`` — the directory every analysis reader
  builds — then returns the **two paths** of one folder: the Windows path
  the plugin's ``FilePath`` receives (translated for the service that runs
  it, ``data_paths.plugin_save_path``) and the worker's ``file://`` URI the
  stream resource carries for Tiled.  The filename is the GEECS device name,
  so the stock template ``%s%s.h5`` yields ``<device>/<device>.h5`` — the
  file the read side (``geecs_data_utils.io.scan_stack``) looks for.
- :func:`file_plugin_hosts` — which camera servers serve the plugin
  (``config.ini [pva] file_plugin_addr_list``, falling back to
  ``[pva] addr_list``).  h5py is a bootstrap-time dependency on the camera
  servers, so the rollout is per box, and a camera on a box not yet rolled
  keeps LabVIEW-native saving.  The key goes when the fleet is rolled.
"""

from __future__ import annotations

import configparser
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Annotated as A

from ophyd_async.core import PathInfo, PathProvider, SignalR, SignalRW
from ophyd_async.core._path_providers import generate_directory_uri
from ophyd_async.epics.adcore import NDFileHDF5IO
from ophyd_async.epics.core import PvSuffix

from geecs_bluesky.data_paths import plugin_save_path


class GeecsHdfIO(NDFileHDF5IO):
    """``NDFileHDF5IO`` plus the GEECS PVs: the refire guard and the writer's status."""

    rewind: A[SignalRW[int], PvSuffix("Rewind")]
    write_status: A[SignalR[str], PvSuffix("WriteStatus")]
    write_message: A[SignalR[str], PvSuffix("WriteMessage")]


class PluginPathProvider(PathProvider):
    """``ScanNNN/<GEECS device>/`` as the plugin and Tiled each see it.

    Parameters
    ----------
    shared :
        The worker's run-scoped provider (called with the GEECS device name).
    device :
        The GEECS device name: the directory and the file stem.
    plugin_path :
        Worker path → the path the plugin's host can write (the UNC root of
        the data share); defaults to the ``config.ini`` mapping.
    """

    def __init__(
        self,
        shared: PathProvider,
        device: str,
        *,
        plugin_path: Callable[[str], str] = plugin_save_path,
    ) -> None:
        self._shared = shared
        self._device = device
        self._plugin_path = plugin_path

    def __call__(self, datakey_name: str | None = None) -> PathInfo:
        """The device directory this run: Windows path for ``FilePath``, URI for Tiled."""
        local = self._shared(self._device)
        local_dir = Path(local.directory_path)
        return PathInfo(
            directory_path=PureWindowsPath(self._plugin_path(str(local_dir))),
            filename=self._device,
            directory_uri=generate_directory_uri(local_dir),
        )


def file_plugin_hosts(config_path: Path | None = None) -> set[str] | None:
    """Camera-server IPs whose gateway serves the file plugin.

    ``[pva] file_plugin_addr_list`` when present, else ``[pva] addr_list``
    (the deployed gateways); ``None`` when neither key exists — the caller
    then treats no host as plugin-backed.
    """
    path = config_path or Path.home() / ".config" / "geecs_python_api" / "config.ini"
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    for key in ("file_plugin_addr_list", "addr_list"):
        raw = cfg.get("pva", key, fallback=None)
        if raw is not None:
            return {token for token in raw.replace(",", " ").split() if token}
    return None
