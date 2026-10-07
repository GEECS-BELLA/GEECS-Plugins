# GeecsBluesky

Bridges GEECS to the [Bluesky](https://blueskyproject.io/) experiment
orchestration ecosystem via [ophyd-async](https://ophyd-async.readthedocs.io/).

Devices are **CA-backed**: they consume the PVs served by
[`GeecsCAGateway`](../GeecsCAGateway) (the GEECS access layer) as a standard
EPICS IOC — stock `epics_signal_r/rw` under the hood, no bespoke transport.
The package is a **native Bluesky application**: a
stock `RunEngine` running a small set of registered plans over ophyd-async
devices, and the only GEECS line in the acquisition path is the fire
between trigger and wait.  It owns:

- `namespace.py` — `GeecsNamespace`: every device of the experiment as a
  long-lived noun built from the GEECS DB; an acquirer is a `GeecsDetector`
  (`devices/detector.py`, a stock `StandardDetector` whose shot is its
  `acq_timestamp` advancing; LabVIEW-native saving or the PVA gateway's
  file plugin as its data logic), a scalar-only device a
  `CaSnapshotReadable`, every settable a Movable child (`U_S1H.current`)
- `devices/shot_control.py` — `ShotControl`, the trigger box as a `Movable`
  over the trigger profile's states and `Pausable`
- `plans/` — the registered plans (`plan_names.GEECS_PLAN_NAMES`): `count`
  (stock `bp.count` bound strict), `sweep` (a `geecs_schemas.Sweep`
  trajectory through stock `scan_nd`), `optimize` (native Xopt ask/tell),
  plus the utilities `mv`, `run_action` and the two shot-offset
  calibration plans.  `plans/strict.py` is the strict `take_reading` (the
  `SINGLESHOT` fire between the triggers and the wait, plus the bounded
  refire); `plans/gated.py` the gated batch (the box free-runs while the
  plugin-backed cameras count frames); `plans/registry.py` binds them
- `plans/claim_scan.py` — the day-scoped scan-number claim (the one place
  a `scans/ScanNNN/` folder comes into existence)
- `run_engine.py` — `make_run_engine`: one RunEngine with
  `connect_on_demand` (`preprocessors.py`) installed outermost, the
  ScanInfo / s-file / `scan.log` / stack-check callbacks (`callbacks/`)
  and the Tiled spool subscribed (`tiled/spool.py`; the
  `geecs-tiled-writer` service in `tiled/writer.py` registers the spooled
  runs off the engine thread)
- `qserver/` — the **queueserver worker**: a bluesky-queueserver RE Manager
  whose startup profile exports the namespace's devices and the registered
  plans; `qs_client/` — the manager client every GEECS client uses
  (`submit_plan`, `submit_preset`, the pre-submit preflight)
- `optimization/` — the native Xopt ask/tell driver, live PVA frame joins
  and the measurement compiler behind the `optimize` plan (the `optimize`
  extra)

`CLAUDE.md` is the design of record; `EVENT_SCHEMA.md` lists the GEECS
keys a run's documents carry.

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
from geecs_bluesky.plans.strict import geecs_per_shot
from geecs_bluesky.run_engine import make_run_engine

RE = make_run_engine(tiled=True)                      # RE + connect_on_demand + the Tiled spool
ns = GeecsNamespace.from_experiment("Undulator")      # every DB device, lazily connected
box = ShotControl.from_profile(
    ConfigsRepoResolver("Undulator").resolve_trigger_profile("HTU-NoGas"),
    experiment="Undulator", name="shot_control",
)
RE(bps.mv(box, "ARMED"))
RE(bp.count([ns["UC_Amp4_IR_input"]], 5, per_shot=geecs_per_shot(box)))
RE(bps.mv(box, "STANDBY"))
```

This is the strict hook on a bare stock plan, without the scan claim.  A
GEECS scan — the claim, ScanInfo, the s-file, `scan.log`, the native files
in the claimed folder — is what the worker's bound plans add
(`make_run_engine(claim=True)` + `plans.registry.bind_plans`);
`tests/test_phase0_hardware.py` and `tests/test_phase1_hardware.py` are the
runnable versions of both.

## Reading data back

Scalars round-trip from Tiled (`TILED_SETUP.md`; a run is registered there
by `geecs-tiled-writer` at its close); native files (images, traces) are
named with the row's `acq_timestamp` and join by it.

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
