# GeecsBluesky

Bridges GEECS to the [Bluesky](https://blueskyproject.io/) experiment
orchestration ecosystem via [ophyd-async](https://ophyd-async.readthedocs.io/).

Devices are **CA-backed**: they consume the PVs served by
[`GeecsCAGateway`](../GeecsCAGateway) (the GEECS access layer) as a standard
EPICS IOC — stock `epics_signal_r/rw` under the hood, no bespoke transport.
This package owns the *scan* side:

- `GeecsSession` — headless scans with the full GEECS run discipline
  (scan numbering, ScanInfo, save paths, event schema v1, Tiled, s-file export)
- `BlueskyScanner` — the `ScanManager`-compatible GUI bridge (a thin adapter
  over the session)
- `devices/ca/` — `CaGenericDetector`, `CaTimestampedReadable`,
  `CaSnapshotReadable`, `CaMotor`, `CaSettable` + the shared shot-id /
  contributor / native-saving mixins
- `ShotController` — arm/disarm/quiesce/single-shot plan stubs driving the
  shot-control device through the gateway `:SP` PVs
- plans: reference-paced free-run and strict plan-owned single-shot, one
  shared orchestration recipe (`plans/orchestration.py`)

## Current status

Both acquisition modes (selected by `GEECS_BLUESKY_ACQUISITION_MODE`) are
hardware-verified over the gateway, including native image saving, external
asset documents, Tiled persistence, and DG645 shot control:

- **`free_run_time_sync`** — external trigger free-runs; a reference device
  paces event rows and other devices contribute timestamp-matched data
  (tolerant of late/missing devices).
- **`strict_shot_control`** — every device required per shot; true plan-owned
  single-shot. Requires a reachable shot-control device and a non-empty
  `ARMED` state.

Both modes write the same versioned event schema (see `EVENT_SCHEMA.md`).
For `save_images` devices, each event records the detector `acq_timestamp`
and the save directory; file names remain hardware-native and are joined by
`acq_timestamp`.

Still open (features, not architecture): setup/closeout actions,
background/optimization modes, legacy TDMS output.

## Requirements

- Python 3.11
- A running GeecsCAGateway serving your experiment's PVs. Point clients at
  it with `[epics] ca_addr_list = <gateway-host>` in
  `~/.config/geecs_python_api/config.ini` (applied automatically at package
  import; `EPICS_CA_AUTO_ADDR_LIST` defaults to `NO` when applied) — or by
  exporting `EPICS_CA_ADDR_LIST`, which always wins over the config value
- The `ca` extra (`aioca`; bundles libca — no system EPICS needed)

## Installation

```bash
cd GeecsBluesky
poetry install --extras "ca tiled"
```

The `geecs-ca-gateway` path dependency provides the GEECS access-layer
library (`GeecsDb` metadata, `pv_naming`, wire-level exceptions) and the
`FakeGeecsServer` test double. DB credentials resolve through the standard
`~/.config/geecs_python_api/config.ini` → `Configurations.INI` chain.

## Quick start (headless session)

```python
from geecs_bluesky.session import GeecsSession

s = GeecsSession("Undulator")                       # RE + Tiled subscription
cam = s.detector("UC_Amp2_IR_input", ["centroidx"], save_images=True)
top = s.contributor("UC_TopView", ["centroidx"])
jet = s.motor("U_ESP_JetXYZ", "Position.Axis 1")
s.shot_control("HTU-LaserOFF")                      # from the configs repo

s.scan(detectors=[cam, top], motor=jet, start=4.0, end=5.0, step=0.5,
       shots_per_step=3)                            # free-run step scan
s.noscan(detectors=[cam], shots=10, mode="strict")  # plan-owned single-shot
```

Every session scan claims a real scan number, writes `ScanInfoScanNNN.ini`,
drives native image saving through the gateway, persists documents to Tiled,
and exports the legacy `ScanDataScanNNN.txt` / `sNN.txt` files — identical to
a GUI scan.

Ad-hoc acquisition without touching the data tree: `save_data=False`.

## Reading data back

Scalars round-trip from Tiled; native files (images, traces) load through the
asset contract by date/scan/device — see `geecs_bluesky.assets`
(`load_asset_from_tiled`) and the `tiled_camera_analysis_sidecar` notebook.

## Running the tests

```bash
poetry run pytest            # hermetic suite (ophyd-async mock backends)
```

Plain `pytest` needs no lab network and no gateway: shots are simulated with
`set_mock_value` and an RE-loop pacer (`tests/ca_mock_helpers.py`). The
hardware test is explicit (integration-marked, real scans against lab
devices):

```bash
poetry run pytest tests/test_scan_request_hardware.py -m integration -s
```

Save set, trigger profile, and every other name are parameterizable via
`GEECS_HW_*` env vars — see the module docstring.
