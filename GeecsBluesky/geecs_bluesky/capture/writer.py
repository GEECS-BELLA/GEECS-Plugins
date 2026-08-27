"""Frame-stack writers: the container seam of the capture daemon.

``FrameStackWriter`` is the deliberate format-agnostic boundary — the daemon
talks only to this protocol, so the container technology (HDF5 today, e.g.
Zarr tomorrow) can change without touching capture logic. The contract lives
in ``FORMAT.md`` next to this module; every file is self-describing via its
``schema`` attribute.

``h5py`` rides the ``capture`` extra and is imported lazily so the package
import stays light for non-capture consumers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_ID = "geecs-capture/1"


class FrameStackWriter(Protocol):
    """One device's frame-stack file for one scan."""

    def append(
        self, frame: "np.ndarray", acq_timestamp: float, recv_timestamp: float
    ) -> None:
        """Append one frame with its timestamps."""
        ...

    def finalize(self, counters: dict[str, int]) -> int:
        """Stamp reconciliation *counters*, mark finalized, close; return frame count."""
        ...

    def abort(self) -> None:
        """Close without marking finalized (daemon shutdown mid-scan)."""
        ...


class Hdf5StackWriter:
    """The v0 ``FrameStackWriter``: one ``<device>.h5`` per device per scan.

    The target directory must already exist — the engine's save-enable plan
    owns ``scans/ScanNNN/<device>/`` creation and this class refuses to
    create it (cross-package invariant).
    """

    def __init__(
        self,
        target_dir: Path,
        *,
        device: str,
        experiment: str,
        scan_number: int | None,
        source_pv: str,
    ) -> None:
        if not target_dir.is_dir():
            raise FileNotFoundError(
                f"capture target dir missing (never created by the daemon): {target_dir}"
            )
        import h5py

        self._path = target_dir / f"{device}.h5"
        # libver="latest" enables efficient appends; flush-per-append keeps
        # the crash window to the unflushed tail (trailing-flush design).
        self._h5 = h5py.File(self._path, "w", libver="latest")
        self._h5.attrs["schema"] = SCHEMA_ID
        self._h5.attrs["device"] = device
        self._h5.attrs["experiment"] = experiment
        if scan_number is not None:
            self._h5.attrs["scan_number"] = scan_number
        self._h5.attrs["source_pv"] = source_pv
        self._h5.attrs["created"] = time.time()
        self._frames = None
        self._acq = self._h5.create_dataset(
            "acq_timestamp", shape=(0,), maxshape=(None,), dtype="f8"
        )
        self._recv = self._h5.create_dataset(
            "recv_timestamp", shape=(0,), maxshape=(None,), dtype="f8"
        )
        self._n = 0

    @property
    def path(self) -> Path:
        """The file being written."""
        return self._path

    def append(
        self, frame: "np.ndarray", acq_timestamp: float, recv_timestamp: float
    ) -> None:
        """Append one frame; dataset geometry is fixed by the first frame."""
        if self._frames is None:
            self._frames = self._h5.create_dataset(
                "frames",
                shape=(0, *frame.shape),
                maxshape=(None, *frame.shape),
                chunks=(1, *frame.shape),
                dtype=frame.dtype,
            )
        if frame.shape != self._frames.shape[1:]:
            # A mid-scan ROI/binning change breaks the stack contract —
            # count upstream, never write a mismatched frame.
            raise ValueError(
                f"frame shape {frame.shape} != stack shape {self._frames.shape[1:]}"
            )
        n = self._n
        self._frames.resize(n + 1, axis=0)
        self._frames[n] = frame
        self._acq.resize(n + 1, axis=0)
        self._acq[n] = acq_timestamp
        self._recv.resize(n + 1, axis=0)
        self._recv[n] = recv_timestamp
        self._n = n + 1
        self._h5.flush()

    def finalize(self, counters: dict[str, int]) -> int:
        """Stamp *counters* + ``finalized=True`` and close; return frame count."""
        for key, value in counters.items():
            self._h5.attrs[key] = value
        self._h5.attrs["finalized"] = True
        self._h5.close()
        return self._n

    def abort(self) -> None:
        """Close without the finalized stamp (data written so far stays valid)."""
        try:
            self._h5.close()
        except Exception:  # noqa: BLE001 - best-effort shutdown
            logger.warning("Hdf5StackWriter abort close failed for %s", self._path)
