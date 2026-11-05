"""A fake Tiled run for ``TiledScanCatalog``: streams of table and 1-D array parts.

Shared by the portal's shot-table tests.  Every part read is counted; a
part whose value is an exception raises it when read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.tiled_catalog import TiledScanCatalog


class Part:
    def __init__(self, run, value):
        self.run, self.value = run, value

    def read(self):
        self.run.reads.append(self)
        if isinstance(self.value, Exception):
            raise self.value
        assert isinstance(self.value, pd.DataFrame) or np.ndim(self.value) <= 1
        return self.value


class Stream:
    def __init__(self, run, parts: dict):
        self.base = {name: Part(run, value) for name, value in parts.items()}
        self.metadata: dict = {}

    def get_contents(self) -> dict:
        def item(part):
            family = "table" if isinstance(part.value, pd.DataFrame) else "array"
            shape = list(np.shape(part.value)) if family == "array" else []
            return {
                "attributes": {
                    "structure_family": family,
                    "structure": {"shape": shape},
                }
            }

        return {name: item(part) for name, part in self.base.items()}


class Run(dict):
    """Stream name → node, with start/stop metadata and the parts read."""

    def __init__(self, start: dict, streams: dict, stop: dict | None = None):
        self.reads: list[Part] = []
        super().__init__({k: Stream(self, v) for k, v in streams.items()})
        self.metadata = {
            "start": start,
            "stop": {"exit_status": "success"} if stop is None else stop,
        }


def camera(device: str, lv_stamps, **scalars) -> dict:
    """A plugin camera's parts: the stack (never read), its stamps and scalars."""
    parts = {
        device: np.zeros((len(lv_stamps), 2, 2)),
        f"{device}-hdf-image-frame_acq_timestamp": np.array(lv_stamps)
        - LABVIEW_EPOCH_OFFSET,
    }
    parts.update(
        {f"{device}-hdf-image-{k}": np.asarray(v, float) for k, v in scalars.items()}
    )
    return parts


def catalog(runs: dict) -> TiledScanCatalog:
    """A ``TiledScanCatalog`` over *runs* (uid → :class:`Run`)."""
    tiled = TiledScanCatalog(uri="http://fake:8000")
    tiled._client = runs
    return tiled
