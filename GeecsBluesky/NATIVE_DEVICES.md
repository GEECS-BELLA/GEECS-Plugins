# Native EPICS devices in the scan path — design note

Status: **proposal, revised after red team** (2026-10-21). When it is
built, the rules below move into `CLAUDE.md` and this note is deleted.

## Why

A device can now be served by a native EPICS IOC instead of a LabVIEW
device app (`GEECS-BELLA/geecs-epics`: a Basler on `htu-linux-camserv2`
behind ADAravis, writing HDF5 through its own `NDFileHDF5` plugin). This
note is how such a device takes part in a GEECS scan.

Settled (`geecs-epics/docs/decisions.md`): **one owner per device** — the
GEECS DB or geecs-epics; a host migrates whole; IOC configuration is
EPICS-native. Integration code, the device registry included, lives on
the GEECS side (decision 6).

**The lesson of the red team:** a native camera cannot be the stock
`AravisDetector` with a few attributes added. The stock detector's
per-shot acquire model, its paths, its stack layout and its timeouts all
differ from what the scan path assumes. It must be a **GEECS camera whose
transport is areaDetector**: the same contract as `GeecsDetector`, over
stock AD records.

## Rules in `CLAUDE.md` this changes

- *"The DB as the roster's source of truth"* → the DB **plus** a native
  registry in the configs repo, disjoint by construction (§2).
- *"touches devices only through the gateway's CA PVs"* → also a native
  IOC's own CA PVs (stock areaDetector records); still never a device
  command outside EPICS.
- The #868 caveat (a refused `signal.set()` reads as success) applies to
  the native plugin's puts exactly as to the gateway's.

## Scope of the first build

In: one **native camera** as an **essential detector** in **strict**
`count` and `sweep`, frames in `ScanNNN/<device>/<device>.h5`, joined to
the rows by stamp, registered in Tiled, found by analysis.

Refused at bind with a named error (not absent — each falls through a
GEECS-only branch otherwise): gated runs, non-essential streams, the
calibration plans, `optimize` (its live frames match on gateway PVs),
scan-mode profiles (the IOC never acknowledges `ScanStart`; box-less
profiles fire no edge), a preset that asks for `X.scalars` of a native
camera. Out entirely: other device kinds, native settables in the
scanner's lists, the archive set.

## 1. The registry — in the configs repo

`<experiment>/native_devices.yaml` in the configs repo (private,
per-experiment, already pulled on the worker; no deploy key):

```yaml
schema_version: 1
devices:
  UC_ExampleCam: {kind: aravis_camera}
```

- **Names and kind only.** The PV prefix is derived by decision 5
  (`<experiment>:<device>:`, GEECS normalization), so it cannot drift from
  `ioc.yaml`; the record suffixes are fixed per kind (`cam:`, `hdf1:`).
  No hosts, no paths: facility values keep their one home.
- Schema `geecs_schemas.epics.NativeDevices` (v1); `kind` a discriminated
  union. Read through `ConfigsRepoResolver` like every configs document.
- geecs-epics CI checks every registry name has a `services/*/ioc.yaml`
  serving the derived prefix (it reads the configs repo; not the reverse).
- The test IOC (`ADTEST:CAM1`) is renamed to the convention before it is
  registered.

## 2. One owner, enforced — new code

- Migration = disable the device in the experiment **and** add it to the
  registry, in one change, then restart the gateways (they serve until
  restarted; `--include-disabled` must not be set) and re-derive the
  archive set. A checklist in geecs-epics `docs/` covers this and the
  configs that name the device's old variables (presets, catalog,
  actions, analysis).
- `GeecsNamespace` gains an explicit check: a registry name (after
  `identifier_name`) that is also an enabled roster device → refuse to
  open, naming both owners. The existing check compares the roster only
  with itself.

## 3. The device — `devices/native.py::NativeCamera`

A `StandardDetector` over stock AD IO (`AravisDriverIO`, `NDFileHDF5IO`)
with GEECS logics:

- **Acquire model** (as GEECS cameras run): driver in `Continuous`,
  `TriggerMode On`, `TriggerSource Line1`; **armed once per run** at the
  first prepare, the HDF plugin capturing in Stream mode for the run;
  `unstage` restores the driver's pre-run state (live view keeps working
  between scans). No per-shot `Acquire` writes — the stock model's
  per-shot AcquisitionStart is both too slow for 1 Hz and, in Continuous,
  a hang (`ADAcquireLogic.wait_for_idle` awaits a put that never ends).
  Strict calls `prepare` **every shot** (`strict.py`), so the trigger and
  acquire logics make every prepare after the run's first a no-op: no
  `Acquire` put, the event count ignored.
- **A shot** = the plugin's `NumCaptured` advancing. `trigger()` takes the
  baseline synchronously (the `GeecsAcquireLogic` rule); the wait's
  timeout raises `GeecsTriggerTimeoutError` and sets `missed_shot`.
- **No orphan frames in v1.** The stack must hold exactly the frames the
  datums reference, in order: Bluesky's `HDF5Consolidator` sizes the array
  as the sum of datum widths and reads from frame 0, so a skipped frame
  silently shifts every later frame in Tiled and the portal by one shot.
  `GeecsDetector` avoids this by rewinding its plugin; stock `NDFileHDF5`
  cannot rewind. So, for v1:
  - a native camera that **misses a shot fails the step** (no refire —
    `discard_uncollected` raises, naming the camera);
  - a run with a native camera **and** a gateway plugin camera is refused
    at bind (a refire of the other camera would need a native rewind);
  - a stray frame (an edge outside a shot) is the same fault — hence the
    cabling rule in Acceptance.
  Refires come back when the Tiled writer registers native stacks through
  a consolidator that honours the seq_num→index map (a second
  substitution in `tiled/writer.py`, tested against real Tiled).
- **Stamp, two epochs**: the IOC writes a per-frame `acq_timestamp`
  attribute in **Unix** seconds (the frame's EPICS timestamp) through its
  NDAttributes XML into the stack — the dataset `scan_stack.is_stack_file`
  requires — and a root `finalized` attribute when the file closes (or the
  stack check learns the stock writer's close signal). `NativeCamera`'s
  event column carries the same stamp in the **LabVIEW** epoch (+
  `LABVIEW_EPOCH_OFFSET`, the `test_epoch_contract.py` constant) with
  `_column_headers`, as every row stamp does; one epoch for both and every
  join reads `NaN` without an error. Requires the IOC host's clock
  disciplined (chrony now, PTP later) — the stamp is the join key.
- **Path**: a POSIX variant of `PluginPathProvider` — `PurePosixPath`, the
  worker's path mapped through `config.ini [Paths]
  geecs_native_ioc_data_base_path` (unset = the worker's own root: the IOC
  hosts mount the share at the same path, bind-mounted unchanged into the
  container). `create_dir_depth=0`: the IOC creates nothing; the provider
  makes `<device>/` inside the claimed folder, as today.
- **Liveness**: the asyn port's connection (`cam:AsynIO.CNCT`) through an
  adapter signal rendering the verdict's `Disconnected` — including when
  the read itself fails (IOC down), since `read_disconnected` is fail-open
  and would otherwise call a dead IOC live. `devices/ca/liveness.py` stays
  the one rule. (devIocStats' heartbeat only says the IOC is up, and a
  counter never reads `Disconnected`; a connected but stalled camera is
  caught by the strict timeout.)
- **Facts strict reads**: `name = safe_name(device)`,
  `_geecs_device_name`, `plugin_backed = True`, `native_save = False`,
  `missed_shot`, `plugin_reasons()` (the plugin's `WriteMessage`),
  `discard_uncollected()` (above), `count_zeroed`, `_geecs_namespace_member`.

## 4. Failure behaviour

| Fault | What happens |
|---|---|
| IOC down / camera disconnected | liveness gate refuses before the claim, naming it |
| Share not mounted on the IOC host | `FilePathExists_RBV = 0`; prepare fails naming the path |
| No edge for a shot | `GeecsTriggerTimeoutError` → the step fails naming the camera and the plugin's message (no refire in v1) |
| Stray frame (edge outside a shot) | stack and datums disagree → the stack check reports it; fixed by cabling, not tolerated |
| Native + gateway plugin camera in one run | refused at bind |
| Registry device enabled in the DB | namespace refuses to open |
| Registry unreadable | environment open fails (configured but broken ≠ none) |

## 5. Where the changes land

- **GEECS-Schemas**: `geecs_schemas.epics.NativeDevices` + docgen.
- **GeecsBluesky**: `devices/native.py`, the POSIX path provider, the
  namespace merge + ownership check, the bind-time refusals, the stack
  check's finalized signal if needed, `CLAUDE.md`.
- **configs repo**: `<experiment>/native_devices.yaml`.
- **geecs-epics**: the NDAttributes XML (stamp + finalized), the renamed
  test IOC, the migration checklist, the CI name check.
- **Deploy**: IOC hosts mount the share at the worker's path, run chrony.

## Acceptance

**Cabling rule**: `Line1` hangs off a DG645 output **gated by the trigger
box's states**, as a GEECS camera's input does — the camera captures every
edge it sees all run long, so a free-running channel puts a stray frame
in the stack on every edge.

With the camera cabled that way: a strict 10-shot `count` of the
registered camera through the queueserver gives 10 frames in
`ScanNNN/<device>/<device>.h5` that `find_stack_file` finds, an s-file of
10 rows joined by stamp, a clean stack check, the run in Tiled with frame
*k* = shot *k*, the frames in the portal; the live view runs before and
after; a deliberately blocked edge fails the step naming the camera.

## Open questions

1. **Refires for native cameras** — v1 has none (above). Worth the
   consolidator work, or is a failed step on a missed native shot
   acceptable? Decide after running with it.
2. **Finalized signal** — root attribute written by the IOC's XML layout
   vs the stack check reading the plugin's capture state; decided when
   the XML is written.
