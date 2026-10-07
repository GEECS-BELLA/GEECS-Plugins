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
- :class:`GeecsHdfDataLogic` — the stock ``ADHDFDataLogic`` whose
  provider (:class:`GeecsStreamResourceDataProvider`) reads the stream's
  geometry at the first describe rather than at ``prepare``: the plugin
  settles a stream's shape on the session's first fresh frame
  (GEECS-Plugins#1023), after the arm.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path, PureWindowsPath
from typing import Annotated as A

from bluesky.protocols import StreamAsset
from event_model import ComposeStreamResource, DataKey
from ophyd_async.core import (
    PathInfo,
    PathProvider,
    SignalR,
    SignalRW,
    StreamableDataProvider,
    StreamResourceDataProvider,
)
from ophyd_async.core._path_providers import generate_directory_uri
from ophyd_async.epics.adcore import ADHDFDataLogic, NDArrayDescription, NDFileHDF5IO

# The stock geometry reader is not exported (the stock logic calls it at
# prepare); the lazy provider below calls it at the first describe instead.
from ophyd_async.epics.adcore._data_logic import get_ndarray_resource_info
from ophyd_async.epics.core import PvSuffix

from geecs_core.configs_repo import read_config_entry

from geecs_bluesky.data_paths import (
    plugin_save_path,
    pva_addr_tokens,
)


class GeecsHdfIO(NDFileHDF5IO):
    """``NDFileHDF5IO`` plus the GEECS PVs: the refire guard and the writer's status."""

    rewind: A[SignalRW[int], PvSuffix("Rewind")]
    write_status: A[SignalR[str], PvSuffix("WriteStatus")]
    write_message: A[SignalR[str], PvSuffix("WriteMessage")]


class GeecsStreamResourceDataProvider(StreamResourceDataProvider):
    """The stock HDF provider, its main dataset's geometry read at the first describe, not at ``prepare``.

    The plugin declares a stream's geometry at the arm from the frame it
    holds and **re-declares** it on the session's first fresh frame when
    the device's settings moved since (GeecsPvaGateway 0.15.0,
    GEECS-Plugins#1023: the MagSpec lineouts follow the energy axis).  The
    stock provider froze the main dataset's shape and dtype at ``prepare``
    — before any frame — so the descriptor and the StreamResource would
    have carried the held frame's shape over a stack written at another.
    This one re-reads the main dataset's ``StreamResourceInfo`` from the
    plugin's geometry PVs at every ``make_datakeys`` / ``make_stream_docs``
    until a stream datum is out, then keeps it: the descriptor is composed
    after the first frame (strict: at the first ``save``; gated: the
    cameras' stream is declared after the first batch; a non-essential
    plugin stream: declared at the run's close, right before its collect —
    both in ``plans/gated.py``), the resource document goes out with the
    first datum, and both read the shape the plugin settled on.  The NDAttribute datasets are scalars and
    stay as the stock logic described them.  What this relies on — the
    plugin posts the geometry before ``NumCaptured_RBV`` advances — is the
    plugin's contract, pinned by its own ``test_file_plugin``.

    Parameters
    ----------
    stock :
        The provider the stock ``ADHDFDataLogic.prepare_unbounded``
        returned; its URI, resources and signals are taken over.
    array_description :
        The geometry signals the main dataset is re-read from.
    """

    def __init__(
        self,
        stock: StreamResourceDataProvider,
        array_description: NDArrayDescription,
    ) -> None:
        # The stock provider keeps no mimetype; its first bundle's document
        # carries the one it composed with.
        mimetype = str(stock.bundles[0].stream_resource_doc["mimetype"])
        super().__init__(
            uri=stock.uri,
            resources=stock.resources,
            mimetype=mimetype,
            collections_written_signal=stock.collections_written_signal,
            flush_signal=stock.flush_signal,
        )
        self._array_description = array_description
        self._mimetype = mimetype

    async def refresh_geometry(self) -> bool:
        """Re-read the main dataset's shape and dtype off the plugin, unless a datum is out.

        Returns ``True`` when the description changed.
        """
        if self.last_emitted:
            return False
        main = self.resources[0]
        fresh = await get_ndarray_resource_info(
            self._array_description,
            main.data_key,
            main.parameters,
            frames_per_chunk=main.chunk_shape[0],
        )
        if (fresh.shape, fresh.dtype_numpy) == (main.shape, main.dtype_numpy):
            return False
        self.resources[0] = fresh
        self.bundles[0] = ComposeStreamResource()(
            mimetype=self._mimetype,
            uri=self.uri,
            data_key=fresh.data_key,
            parameters={"chunk_shape": fresh.chunk_shape, **fresh.parameters},
            uid=None,
            validate=True,
        )
        return True

    async def make_datakeys(self, collections_per_event: int) -> dict[str, DataKey]:
        """The stock data keys, the main dataset's geometry refreshed first (until a datum is out)."""
        await self.refresh_geometry()
        return await super().make_datakeys(collections_per_event)

    async def make_stream_docs(
        self, collections_written: int, collections_per_event: int
    ) -> AsyncIterator[StreamAsset]:
        """The stock stream documents, the main dataset's geometry refreshed first (until a datum is out)."""
        await self.refresh_geometry()
        async for doc in super().make_stream_docs(
            collections_written, collections_per_event
        ):
            yield doc


class GeecsHdfDataLogic(ADHDFDataLogic):
    """The stock ``ADHDFDataLogic`` whose provider describes the stream lazily.

    Everything the stock logic does at ``prepare`` — the writer's
    parameters, ``Capture=1``, the NDAttribute datasets — is unchanged; the
    provider it returns is wrapped as a
    :class:`GeecsStreamResourceDataProvider`, so the main dataset's
    geometry is the plugin's at the first describe, not at the arm.
    """

    async def prepare_unbounded(self, datakey_name: str) -> StreamableDataProvider:
        """The stock prepare (parameters, ``Capture=1``, attributes); its provider wrapped lazily."""
        stock = await super().prepare_unbounded(datakey_name)
        assert isinstance(stock, StreamResourceDataProvider)
        return GeecsStreamResourceDataProvider(stock, self.array_description)


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
