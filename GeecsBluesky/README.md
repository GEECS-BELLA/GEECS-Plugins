# GeecsBluesky

Bridges GEECS to the [Bluesky](https://blueskyproject.io/) experiment
orchestration ecosystem via [ophyd-async](https://ophyd-async.readthedocs.io/).

Devices are **CA-backed**: they consume the PVs served by
[`GeecsCAGateway`](../GeecsCAGateway) (the GEECS access layer) as a standard
EPICS IOC — stock `epics_signal_r/rw` under the hood, no bespoke transport.
This package is being rebuilt as a **native Bluesky application**
(GEECS-Plugins#807; plan of record
`Planning/native_bluesky/03_clean_room_rebuild.md`): the scan path is the
stock `bluesky.plans` verbs over ophyd-async devices, and the only GEECS
line in it is the fire between trigger and wait.  It owns:

- `namespace.py` — `GeecsNamespace`: every device of the experiment as a
  long-lived noun built from the GEECS DB; an acquirer is a `GeecsDetector`
  (`devices/detector.py`, a stock `StandardDetector` whose shot is its
  `acq_timestamp` advancing; LabVIEW-native saving as its data logic), a
  scalar-only device a `CaSnapshotReadable`, every settable a Movable child
  (`U_S1H.current`)
- `devices/shot_control.py` — `ShotControl`, the trigger box as a `Movable`
  over the trigger profile's states and `Pausable`
- `plans/strict.py` — `geecs_take_reading`: `bps.trigger_and_read` with the
  `SINGLESHOT` fire between the triggers and the wait, plus the bounded
  refire; `geecs_per_shot` / `geecs_per_step` bind it into `bp.count` and
  every N-d scan plan
- `plans/claim_scan.py` — the day-scoped scan-number claim (the one place
  a `scans/ScanNNN/` folder comes into existence)
- `run_engine.py` — `make_run_engine`: one RunEngine with
  `connect_on_demand` (`preprocessors.py`) installed outermost and the
  Tiled / s-file callbacks subscribed
- `qserver/` — the **queueserver worker**: a bluesky-queueserver RE Manager
  whose startup profile exports the namespace's devices and the stock plans
  (`plan_names.GEECS_PLAN_NAMES`); `qs_client/` — the manager client every
  GEECS client uses
- `optimization/` — the Xopt/evaluator core (importable, its tests green;
  re-glued to the native scan path in a later phase)

## Where the rebuild stands

Phase 0 (one camera as a `GeecsDetector`, strict shots under stock
`bp.count` / `bp.list_scan`) is hardware-accepted
(`Planning/native_bluesky/04_phase0_measurements.md`).  Phase 1 PR 1
deleted the `ScanRequest` funnel, the free-run mode, `GeecsSession` and
the funnel-only devices; PR 2 added the plan layer — the stock plans
registered strict under their own names, every run claiming a scan
number, the ScanInfo / s-file / `scan.log` callbacks, the baseline
telemetry stream, presets as the saved queue item.  PR 3 accepted it on
HTU (Scans 104–108 of 26_0910, in process and through a second RE
Manager — `Planning/native_bluesky/05_phase1_acceptance.md`).

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

The `geecs-core` path dependency provides the GEECS access library
(`GeecsDb` metadata, `pv_naming`, wire-level exceptions). The CA gateway
itself is consumed only as a service (its PVs). DB credentials resolve
through the standard `~/.config/geecs_python_api/config.ini` →
`Configurations.INI` chain.

## Quick start (headless)

```python
import bluesky.plan_stubs as bps
import bluesky.plans as bp

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.namespace import GeecsNamespace
from geecs_bluesky.plans.strict import geecs_per_step
from geecs_bluesky.run_engine import make_run_engine

RE = make_run_engine(tiled=True)                      # RE + connect_on_demand + Tiled
ns = GeecsNamespace.from_experiment("Undulator")      # every DB device, lazily connected
box = ShotControl.from_profile(
    ConfigsRepoResolver("Undulator").resolve_trigger_profile("HTU-NoGas"),
    experiment="Undulator", name="shot_control",
)
RE(bps.mv(box, "ARMED"))
RE(bp.list_scan([ns["UC_Amp4_IR_input"]], ns["U_S1H"].current,
                [-1, -0.5, 0, 0.5, 1], per_step=geecs_per_step(box)))
RE(bps.mv(box, "STANDBY"))
```

`tests/test_phase0_hardware.py` is the runnable version of this, with the
scan-number claim and native saving into the claimed folder.

## Reading data back

Scalars round-trip from Tiled (`TILED_SETUP.md`); native files (images,
traces) are named with the row's `acq_timestamp` and join by it.

## Running the tests

```bash
poetry run python -u -m pytest tests   # hermetic suite (ophyd-async mock backends)
```

Plain `pytest` needs no lab network and no gateway: shots are simulated with
`set_mock_value` on `acq_timestamp` (`tests/ca_mock_helpers.py`).  The
hardware acceptance is explicit (hardware-marked, arms the machine trigger):

```bash
GEECS_HW_SCAN_VARIABLE=U_S1H:Current GEECS_HW_SCAN_START=-1 GEECS_HW_SCAN_END=1 \
GEECS_HW_SCAN_STEP=0.5 GEECS_HW_SAVE=1 \
poetry run python -u -m pytest tests/test_phase0_hardware.py -m hardware -s
```
