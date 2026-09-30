"""The worker's side of the PVA gateway's file plugin (#806).

Three small things; everything else is stock ophyd-async:

- :class:`GeecsHdfIO` — ``NDFileHDF5IO`` plus the three PVs the plugin
  adds (``Rewind``, ``WriteStatus``, ``WriteMessage``); the prefix comes
  from :func:`geecs_core.pv_naming.hdf_plugin_prefix`.
- :class:`PluginPathProvider` — the per-detector ``PathProvider`` the
  stock ``ADHDFDataLogic`` calls.  It asks the shared
  :class:`~geecs_bluesky.plans.claim_scan.GeecsScanPathProvider` for
  ``ScanNNN/<GEECS device>/`` and returns that folder's two paths: the
  Windows path the plugin's ``FilePath`` receives
  (``data_paths.plugin_save_path``) and the worker's ``file://`` URI for
  the stream resource.  The filename is the folder's name, so the primary
  stream writes ``<device>/<device>.h5`` and a second capture stream of
  the same device writes the sibling
  ``<device>-<variable>/<device>-<variable>.h5``, the layout the
  LabVIEW-native files use, so ``find_stack_file`` resolves both.  Two
  plugins must never share a path: each writer opens it with ``"w"``.
- :func:`file_plugin_hosts` — the camera servers that serve the plugin
  (``config.ini [pva] file_plugin_addr_list``; absent means none).  The
  rollout is per box; a camera on a box not yet rolled keeps
  LabVIEW-native saving.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Annotated as A

from ophyd_async.core import PathInfo, PathProvider, SignalR, SignalRW
from ophyd_async.core._path_providers import generate_directory_uri
from ophyd_async.epics.adcore import NDFileHDF5IO
from ophyd_async.epics.core import PvSuffix

from geecs_bluesky.data_paths import (
    plugin_save_path,
    pva_addr_tokens,
    read_config_entry,
)


class GeecsHdfIO(NDFileHDF5IO):
    """``NDFileHDF5IO`` plus the GEECS PVs: the refire guard and the writer's status."""

    rewind: A[SignalRW[int], PvSuffix("Rewind")]
    write_status: A[SignalR[str], PvSuffix("WriteStatus")]
    write_message: A[SignalR[str], PvSuffix("WriteMessage")]


class PluginPathProvider(PathProvider):
    """``ScanNNN/<stem>/`` as the plugin and Tiled each see it, one stream per stem.

    Parameters
    ----------
    shared :
        The worker's run-scoped provider (called with the stem, the way it
        is called with a GEECS device name).
    device :
        The GEECS device name.
    variable :
        ``None`` for the device's primary stream — the stem is the device
        name, ``<device>/<device>.h5`` — else the GEECS variable of a
        secondary stream, which lives in the sibling folder
        ``<device>-<variable>/`` (see the module docstring).
    plugin_path :
        Worker path → the path the plugin's host can write (the UNC root of
        the data share); defaults to the ``config.ini`` mapping.
    """

    def __init__(
        self,
        shared: PathProvider,
        device: str,
        *,
        variable: str | None = None,
        plugin_path: Callable[[str], str] = plugin_save_path,
    ) -> None:
        self._shared = shared
        self._stem = device if variable is None else f"{device}-{variable}"
        self._plugin_path = plugin_path

    @property
    def stem(self) -> str:
        """The folder name and file stem this provider hands out (``<device>`` or ``<device>-<variable>``)."""
        return self._stem

    def __call__(self, datakey_name: str | None = None) -> PathInfo:
        """The device directory this run: Windows path for ``FilePath``, URI for Tiled.

        The directory is **created here**, inside the scan folder the
        scanner claimed (``mkdir(exist_ok=True)``, never ``parents``: a
        missing scan folder is an error, the root ``CLAUDE.md`` invariant).
        The plugin never creates it — it refuses to arm on a missing
        ``FilePath`` — and in a fly prepare (a gated batch, a non-essential
        stream) the LabVIEW-native saving logic, whose ``prepare_single``
        used to create it as a side effect of the dual-write, is not part
        of the context.
        """
        local = self._shared(self._stem)
        local_dir = Path(local.directory_path)
        if not local_dir.parent.is_dir():
            raise FileNotFoundError(
                f"{local_dir.parent} does not exist: the scan folder is claimed "
                "by the scanner, never created by a detector's path provider"
            )
        local_dir.mkdir(exist_ok=True)
        return PathInfo(
            directory_path=PureWindowsPath(self._plugin_path(str(local_dir))),
            filename=self._stem,
            directory_uri=generate_directory_uri(local_dir),
        )


def file_plugin_hosts(config_path: Path | None = None) -> set[str] | None:
    """Camera-server IPs whose gateway serves the file plugin.

    ``config.ini [pva] file_plugin_addr_list`` (space- or comma-separated);
    ``None`` when absent — the caller then treats no host as plugin-backed.
    Deliberately **not** ``[pva] addr_list``: that key is the deployed PVA
    fleet, and a box in it that is not yet re-bootstrapped serves no plugin.
    """
    raw = read_config_entry("pva", "file_plugin_addr_list", config_path)
    if raw is None:
        return None
    return set(pva_addr_tokens(raw))
