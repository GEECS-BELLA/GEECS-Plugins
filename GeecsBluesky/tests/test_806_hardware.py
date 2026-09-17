"""#806 hardware acceptance: a plugin-backed camera end to end on HTU.

Hardware-marked **and** gated on ``GEECS_HW=1`` (it arms the trigger and
fires shots).  Run in process on the worker host, from a checkout whose
``config.ini`` lists the camera's server in ``[pva] file_plugin_addr_list``
and carries ``[Paths] geecs_pva_plugin_data_base_path`` (design §8 step 3)::

    GEECS_HW=1 GEECS_SCANNER_CONFIG_DIR=.../GEECS-Plugins-Configs/scanner_configs/experiments \\
    poetry run python -u -m pytest tests/test_806_hardware.py -m hardware -s

One strict ``count`` on the camera through the worker's own wiring; then,
from disk and from the documents:

- the run's stream resource names ``ScanNNN/<device>/<device>.h5`` with the
  NDFileHDF5 dataset, one stream datum per row, indices contiguous;
- the stack exists, is finalized, holds exactly one frame per row, and
  every frame's ``acq_timestamp`` is its row's (the stack check's own
  verdict is in ``scan.log``);
- **parity with the native PNGs** (dual-write stays on until #738): one
  PNG per row, named by the same stamps — the diff tool's join, inline;
- the per-shot cadence, printed (the count wait adds one PVA monitor
  update after the write over SMB — expected to hold 1 Hz);
- a Tiled read of the image array, **reported, not asserted**: the Tiled
  server needs ``HDF5_USE_FILE_LOCKING=FALSE`` in its unit to open a
  stack over SMB, a deployment edit outside this test.

Environment: ``GEECS_HW_EXPERIMENT`` (``Undulator``), ``GEECS_HW_CAMERA_DEVICE``
(``UC_Amp4_IR_input``), ``GEECS_HW_TRIGGER_PROFILE`` (``HTU-NoGas``),
``GEECS_HW_SHOTS`` (5), ``GEECS_HW_DOCS_OUT`` (optional JSON dump).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tests.ca_mock_helpers import DocCollector, wait_for_native_files

pytestmark = pytest.mark.hardware
pytest.importorskip("aioca")
pytest.importorskip("p4p")
if os.environ.get("GEECS_HW") != "1":
    pytest.skip(
        "fires real shots: set GEECS_HW=1 to run on the lab network",
        allow_module_level=True,
    )

EXPERIMENT = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
CAMERA = os.environ.get("GEECS_HW_CAMERA_DEVICE", "UC_Amp4_IR_input")
PROFILE = os.environ.get("GEECS_HW_TRIGGER_PROFILE", "HTU-NoGas")
SHOTS = int(os.environ.get("GEECS_HW_SHOTS", "5"))


def _tiled_read(start: dict, key: str) -> str:
    """Best-effort: read the run's image array back from Tiled; a report line."""
    try:
        from tiled.client import from_uri

        from geecs_bluesky.data_paths import read_config_entry

        uri = read_config_entry("tiled", "uri")
        api_key = read_config_entry("tiled", "api_key")
        if not uri:
            return "tiled: no [tiled] uri in config.ini"
        client = from_uri(uri, api_key=api_key)
        run = client[start["uid"]]
        node = run["primary"]
        # Tiled 0.2 lays external data under the stream's data node.
        for path in (["data", key], ["external", key], [key]):
            try:
                array = node
                for part in path:
                    array = array[part]
                shape = array.shape
                frame = array[0]
                return f"tiled: {'/'.join(path)} shape {shape}, frame 0 {frame.dtype} read OK"
            except Exception as exc:  # noqa: BLE001 - try the next layout
                last = exc
        return f"tiled: image array not readable ({type(last).__name__}: {last})"
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return f"tiled: read failed ({type(exc).__name__}: {exc})"


@pytest.mark.hardware
def test_plugin_camera_count_on_hardware() -> None:
    """A strict count on a plugin-backed camera leaves an exact stack beside its PNGs."""
    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.namespace import GeecsNamespace
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider
    from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
    from geecs_bluesky.run_engine import make_run_engine
    from geecs_data_utils.io.scan_stack import (
        LABVIEW_EPOCH_OFFSET,
        find_stack_file,
        open_stack,
        read_stack_timestamps,
    )
    from geecs_data_utils.native_files import timestamp_key

    t_build = time.monotonic()
    provider = GeecsScanPathProvider()
    namespace = GeecsNamespace.from_experiment(EXPERIMENT, path_provider=provider)
    camera = namespace[CAMERA]
    assert camera.plugin_backed, (
        f"{CAMERA} is not plugin-backed: is its camera server in "
        "config.ini [pva] file_plugin_addr_list, and does the DB list an image variable?"
    )
    assert camera.native_save, f"{CAMERA}: dual-write needs its native saving controls"
    profiles = TriggerProfiles.from_resolver(
        ConfigsRepoResolver(EXPERIMENT), experiment=EXPERIMENT
    )
    RE = make_run_engine(
        experiment=EXPERIMENT,
        tiled=True,
        claim=True,
        path_provider=provider,
        telemetry=namespace.telemetry(),
    )
    plans = bind_plans(profiles)
    print(
        f"\nbuilt in {time.monotonic() - t_build:.1f} s: {len(namespace)} devices; "
        f"{CAMERA} plugin PVs at {camera.hdf.capture.source}"
    )

    docs = DocCollector()
    RE.subscribe(docs)
    t0 = time.monotonic()
    RE(
        plans["count"](
            [camera],
            SHOTS,
            trigger_profile=PROFILE,
            md={"description": "806 acceptance: plugin camera count"},
        )
    )
    wall = time.monotonic() - t0
    out = os.environ.get("GEECS_HW_DOCS_OUT")
    if out:
        Path(out).write_text(json.dumps(docs.docs, default=str, indent=1))

    (start,) = docs.docs["start"]
    (stop,) = docs.docs["stop"]
    assert stop["exit_status"] == "success", stop
    folder = Path(start["scan_folder"])
    primary = docs.primary_events()
    assert len(primary) == SHOTS, (
        f"{len(primary)} rows for {SHOTS} shots (partial rows?)"
    )
    key = camera.name
    stamps = [e["data"][f"{key}-acq_timestamp"] for e in primary]
    assert len(set(stamps)) == SHOTS, stamps

    # --- the documents reference the stack
    resources = [r for r in docs.docs["stream_resource"] if r["data_key"] == key]
    assert len(resources) == 1, [r["data_key"] for r in docs.docs["stream_resource"]]
    (resource,) = resources
    assert resource["mimetype"] == "application/x-hdf5"
    assert resource["parameters"]["dataset"] == "/entry/data/data"
    assert resource["uri"].endswith(f"{CAMERA}/{CAMERA}.h5"), resource["uri"]
    datums = [
        d for d in docs.docs["stream_datum"] if d["stream_resource"] == resource["uid"]
    ]
    assert [(d["indices"]["start"], d["indices"]["stop"]) for d in datums] == [
        (i, i + 1) for i in range(SHOTS)
    ]

    # --- the stack on disk is exactly the rows
    stack = find_stack_file(folder / CAMERA)
    assert stack is not None, (
        f"no stack in {folder / CAMERA}: {sorted(p.name for p in (folder / CAMERA).iterdir())}"
    )
    with open_stack(stack) as f:
        assert bool(f.attrs.get("finalized", False)), dict(f.attrs)
        frames = f["/entry/data/data"]
        assert frames.shape[0] == SHOTS, frames.shape
        assert frames.chunks[0] == 1
        counters = {
            k: int(v)
            for k, v in f.attrs.items()
            if k.startswith("frames_")
            or k.endswith("_dropped")
            or k.endswith("_skipped")
            or k.endswith("_errors")
            or k == "rewound"
        }
        shape, dtype = frames.shape, frames.dtype
    stack_stamps = read_stack_timestamps(stack)  # Unix s
    row_keys = [timestamp_key(s) for s in stamps]  # rows are LabVIEW-epoch doubles
    stack_keys = [timestamp_key(s + LABVIEW_EPOCH_OFFSET) for s in stack_stamps]
    assert stack_keys == row_keys, (stack_keys, row_keys)
    log = (folder / "scan.log").read_text()
    assert "stack check:" in log and "match the rows' stamps" in log, log[-600:]

    # --- parity with the native PNGs (dual-write): the diff tool's join, inline
    pngs = wait_for_native_files(folder / CAMERA, SHOTS)
    names = {p.name for p in pngs}
    for stamp in stamps:
        assert any(f"{stamp:.3f}" in n for n in names), (stamp, sorted(names)[:3])

    cadence = [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]
    print(
        f"{folder.name}: {SHOTS} rows, stack {shape} {dtype}, {len(pngs)} PNGs, "
        f"cadence {cadence}, wall {wall:.1f} s; plugin counters {counters}"
    )
    print(_tiled_read(start, key))
