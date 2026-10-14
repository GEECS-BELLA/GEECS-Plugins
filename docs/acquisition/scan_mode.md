# GEECS scan mode

A scan run under a trigger profile with `use_scan_mode: true` drives the
GEECS devices the way Master Control does. The worker tells them the scan
is starting, announces every shot, and releases them when the run closes.
In between, each device waits for an announced shot, acquires once, saves a
file named by that shot's number and applies its own database start and end
values. Nothing changes on the LabVIEW side.

Use it when the devices should behave as they do under Master Control:
paused between shots, one file per shot, start/end values applied. Without
the flag the devices free-run and the worker joins their data to rows by
timestamp, as for every other Bluesky scan.

## Turning it on

The flag lives on the trigger profile, so a scan picks it up by naming the
profile. A facility whose `experiment_defaults.trigger_profile` names a
flagged profile runs every scan in scan mode unless the scan names another
profile.

```yaml
name: htu_scan_mode
use_scan_mode: true
states:
  STANDBY:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: External rising edges}
  ARMED:
    - {device: U_DG645_ShotControl, variable: Trigger.Source, value: Single shot external rising edges}
  SINGLESHOT:
    - {device: U_DG645_ShotControl, variable: Trigger.ExecuteSingleShot, value: "on"}
```

A profile with the flag and **no writes at all** is the box-less case: no
shot-control device is written, and the announced shot is the only trigger.
It suits free-running devices that should take one shot each when told.
Without the flag an empty profile is refused.

Scan mode is strict acquisition only. `count`, `sweep` and `optimize` run
under it; a gated run and the shot-offset calibration plans refuse a
scan-mode profile.

## What happens during a run

1. **Entry.** After the box is armed and before a scan number is claimed,
   the worker multicasts `ScanStart` on the experiment's group. Every
   enabled device acknowledges and opens a connection back to the worker.
   The worker waits for every device the run records from and the
   profile's shot-control device, retries
   `ScanStart` once for the silent ones, and otherwise refuses the run,
   naming them. A refused entry claims no scan number.
2. **Each shot.** After the detectors are prepared, the worker pushes the
   shot's save path (`…\ScanNNN\<device>\ScanNNN_<device>_SSS.<ext>`) to
   every connected device, then fires the box as usual.
3. **Exit.** When the run closes, normally or on abort or failure, the
   worker sends `ScanEnd` before the box goes to STANDBY. The devices apply
   their end values, acknowledge and close their connections.

## Shot numbers and files

The shot number is run-wide and starts at 1. Every fire takes the next
number, refires included, and a number is never sent twice. A device that
misses a shot still leaves a row: the partial row keeps its number and the
retake takes the next one, so the row index and the number usually agree.
They part when an announced shot ends in no row. For example, an
immediate pause rewinds the shot in progress and the replay is announced
under the next number.

Each row records the number it was announced as:

| Where | Column |
|---|---|
| Tiled `primary` stream | `announced_shot` |
| s-file (`ScanDataScanNNN.txt`) | `Announced Shot` |

`Shotnumber` stays the row index. Readers join a device's shot-numbered
files to rows **by the announced shot**, and a row the device missed gets
no file even if one carries its number. ScanAnalysis's shot mapper, the
`.himg` stack converter, `ScanData`'s expected-path columns and the Data
Portal's image viewer all do so whenever the run carries the column. Code
that reads such files itself takes the numbers from
`geecs_data_utils.tiled_schema.announced_shots(rows)`, which accepts both
spellings.

!!! example "A missed shot, from the pulsed-wire test stand"
    A `count` of 3: the picoscope missed shot 1 and the refire took 2. The
    run has four rows, announced 1 to 4; row 1 is partial (no picoscope
    stamp), and the device folder holds
    `Scan999_pulsewire-picoscope_002.tdms` through `_004`. Rows 2 to 4 get
    their files; row 1 gets none.

The post-run stack check does not match a non-essential device's files to
its stream under scan mode, because the files are named by shot number and
not by timestamp.

## Movers

A device's DB `scanmode_lock` is its opt-out of the paused per-shot loop,
and a mover's is never `enabled`. A mover acknowledges `ScanStart` like
every device but keeps its readbacks flowing, so a sweep moves it between
shots exactly as outside scan mode. A mover with the lock `enabled` is a
database mistake: fix the row. The worker neither checks nor warns.

## Preflight refusals

The worker refuses a scan-mode run before it starts when:

- the worker and the CA gateway are on different hosts. Once in scan mode a
  device accepts commands only from the address announced in `ScanStart`,
  which is the gateway's.
- a profile write and the device's own DB start/end row set the same
  variable to different values.
- the experiment has no scan-mode endpoint (multicast group and port) in
  the database.
- the device-side save root (`device_server_save_path` in `config.ini`) is
  not a drive or UNC path. `ScanStart` reaches every saving device of the
  experiment, so a POSIX path would create folders on each device's own
  drive.

## When a run never ended

A run ends scan mode in its finalizer, which every path runs except
`RE.halt()`. After a halt the devices stay in scan mode, paused. Queue the
`release_scan_mode` plan to end it; the next scan-mode run also ends it
first. If the worker process died, the devices end the scan themselves
when their connection drops, and the worker clears its record at the next
environment open.

## Where to look in the code

- The protocol (multicast, the path channel, acknowledgements):
  GEECS-Core `transport/scan_mode.py`.
- The worker side: GeecsBluesky `devices/scan_mode.py` (`ScanModeControl`)
  and `plans/scan_mode.py` (the bracket, preflight and
  `release_scan_mode`); its `CLAUDE.md` section "Scan mode".
- The column contract: GeecsBluesky `EVENT_SCHEMA.md`.
