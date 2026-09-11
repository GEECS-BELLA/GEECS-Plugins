# #806 — the areaDetector-shaped file plugin, designed against the installed APIs

Status: **design, 2026-09-11** (phase 1, second leg; `03_clean_room_rebuild.md`
§4.A, §8).  Every API claim below was read in the installed source
(ophyd-async 0.19.3, bluesky 1.15.0, tiled 0.2.9, event-model 1.23.1, p4p
4.2.2, h5py 3.16), not the docs.  Where this document disagrees with #806
or with `03_clean_room_rebuild.md`, the disagreement is stated and the
older text is amended in the same PR (the staleness rule, §1 of 03).

## 1. What stays exactly as #806 decided

- The writer runs on the camera server, inside the GeecsPvaGateway process,
  as a second consumer of the frame the gateway already receives — branching
  off **before** the latest-wins slot (`server.py::_CameraWorker._on_frame`).
- On the worker each plugin-backed camera is the existing `GeecsDetector`
  with **stock `ADHDFDataLogic` + `NDFileHDF5IO`** as its data logic.  Only
  the data logic changes; trigger logic, acquire logic and the scalars
  logic are untouched (§7 of 03: "#806 swaps exactly one of the three").
- Layout is the NDFileHDF5 convention (`/entry/data/data`,
  `/entry/instrument/NDAttributes/<name>`), so ophyd-async's stream
  resource description and Tiled's stock HDF5 adapter read it unchanged.
- The capture daemon and everything that existed only to stand outside the
  run are deleted in the same PR (§7 below).

## 2. What this design changes, and why

### 2.1 A missed frame keeps the row, adds a row, and rewinds only the late frame

03 §4.B says a refired shot's late frame "stays on disk, unreferenced by
any document, which is Bluesky's normal model", and that the recovery is
the whole-event redo.  With a file plugin the first is false, and the
second throws away a shot that is 99 % complete.  The ecosystem's
contract, read in three places:

- `StreamResourceDataProvider.make_stream_docs` emits one datum covering
  **every** frame written since the last datum (`last_emitted` →
  `NumCaptured_RBV`), and carries the TODO "fail if we get dropped frames".
- The RunEngine bundler raises on a per-event datum wider than one index
  (`bundlers.py:611`); a zero-width datum (no frame for this event) is
  accepted and the next datum gets the next event number.
- The consolidator keeps a `seq_nums → indices` map with `has_skips`
  (`consolidators.py:224`): a device with no frame for event *k* is a
  documented shape.  `TiledWriter` batches only consecutive datums and
  writes non-consecutive ones separately (`tiled_writer.py:719-727`).

**Decision (Sam, 2026-09-11), as built.**  A missed frame on device B
does not void the row.  The row is saved with **every scalar** the shot
produced, B's columns marked (`NaN` in its scalar columns and its
stamp), and the plan takes **one more shot** for the step, bounded by
the existing refire count.  Strict therefore means "every step gets its
full quota of complete rows, and partial rows are kept and marked", not
"every row complete".  A device confirmed `DISCONNECTED` still aborts
the run (retaking cannot help).

**The partial row carries no frames** — found when the plan was built,
not in the reading above: the RunEngine bundler also requires, within
one event, a datum for **every** external key of the descriptor, all of
one width, or none at all (`bundlers.py:938-943`, `:841-851`).  So a
row where A has no frame cannot reference B's frame either.  Before the
partial row is read, the plan rewinds **every** plugin-backed device of
the shot to the last frame a document referenced (§ below): B's
uncollected frame goes with A's late one.  What is lost per partial row
is the delivered devices' frames of that shot; their scalars stay.
(The zero-width datum the consolidator's `has_skips` map describes is
therefore a *fly-stream* shape — no per-event bundling — and stays
phase 2's, for the non-essential stream.)

What the detector and plan do to make the partial row honest:

- **The scalars of a missed shot are stale** (the CA monitor cache holds
  the previous shot; LabVIEW's own timeout event carries unchanged values
  and the gateway's change suppression drops it, M1).  The acquire logic
  remembers a missed shot until the next `baseline()`, and the scalars
  provider reads `NaN` for every column of that device in that row, the
  stamp included.  The detector knows, because the timeout was raised
  by its own trigger status.
- **Per-device trigger groups** in `fire_and_await_shot`, so every
  device's outcome is known, not just the first failure; the statuses run
  concurrently under one 3 s deadline, so the wall time is unchanged.
  The RunEngine *throws* a failed status into the plan at the next
  message and a second failure replaces the first, so the exceptions
  seen are not a census of the misses; the devices' own `missed_shot`
  flags are, and each group is re-waited until consumed so no stashed
  failure escapes to a later message.
- **The count baseline is asynchronous** (a PVA get inside the trigger
  coroutine), unlike the synchronous stamp baseline; it is covered by
  the fire put's own ~100 ms plus the next-edge wait, two orders of
  magnitude of margin.

**The late frame, and why `Rewind` stays.**  A frame from B's missed
shot that arrives after row *k* closed is written as frame *c*; the
retake's frame becomes *c+1*; row *k+1*'s datum for B spans two frames
and the bundler aborts the run — at every arrival time except after the
run's last collect.  So the plugin gets one non-areaDetector verb,
`Rewind` (int): truncate the datasets to *N*, post `NumCaptured_RBV = N`,
and drop any later frame whose stamp is older than now (the late frame's
stamp is its edge time, ≥ 3 s in the past; the retake's is newer).
Before the partial row is read, the plan calls
`GeecsDetector.discard_uncollected()` on every plugin-backed device of
the shot, through the stock `bps.wait_for` stub; the method sets `Rewind`
to each streamable provider's `last_emitted` (public) and waits for the
readback.  It cannot reach a frame any row references, because the
retake fires only after it.  The alternative — keep the frame and skip its index — needs a
provider of our own and leaves Tiled's positional view of that camera
unreliable for the run, while the stamps stay correct; rejected for the
stock property (Sam: "not a unique issue we encounter at BELLA").

**Synchronicity is asserted, not assumed.**  Every frame carries
`acq_timestamp` in the file and every row carries it in the event, so at
the run's stop a callback checks, per plugin-backed camera, that the
frames written equal the rows with a stamp for it and that each stamp
matches its row, and writes a mismatch to `scan.log`.

### 2.2 The count wait now precedes the stamp wait, with its own timeout

`StandardDetector._wait_for_index` observes `NumCaptured_RBV` until it
reaches baseline + 1 with `TriggerInfo.exposure_timeout` (default
livetime + deadtime + 10 s), **then** calls the acquire logic's
`wait_for_idle` (our 3 s stamp wait).  Two consequences:

- A dropped frame on a plugin-backed camera now surfaces as the count
  timeout, raised as `TimeoutError`, not as `GeecsTriggerTimeoutError`; the
  refire gate re-raises anything that is not the GEECS timeout (Codex
  review of #811).  `GeecsDetector.trigger` therefore translates a count
  timeout into `GeecsTriggerTimeoutError(device)` — the detector owns its
  name, the plan's gate stays as it is.
- `STRICT_TRIGGER_INFO` carries `exposure_timeout = shot_timeout` (3 s),
  so a wedged NAS or a dead plugin fails the shot in 3 s, not 13.

### 2.3 `Capture=1` completes when the plugin has seen a frame

`ADHDFDataLogic.prepare_unbounded` reads the frame geometry
(`ArraySizeX/Y/Z_RBV`, `DataType_RBV`, `ColorMode_RBV`) right after
`set_and_wait_for_value(capture, True)` to describe the stream resource.
A real areaDetector driver knows its geometry before acquiring; LabVIEW
tells us only by pushing a frame.  So the plugin acknowledges `Capture=1`
(posts `Capture_RBV=1`, completing the put) only once its subscription is
live **and one frame has been decoded** — the idle re-push LabVIEW sends
at 1 Hz is enough, and it is stale-filtered, never written.  Cost ≤ 1 s
per run per camera.  A camera that pushes nothing fails `prepare` on the
10 s put timeout naming the PV: the "dead plugin fails at stage/prepare"
property of #806, made concrete.

### 2.4 Two paths per run folder: the plugin's and Tiled's

`PathInfo` carries `directory_path` and an independent `directory_uri`
(`_path_providers.py:53`).  The detector's provider returns the
**service-visible Windows path** as `directory_path` (what `FilePath`
receives) and the **worker's `file://` URI** as `directory_uri` (what the
stream resource carries and Tiled resolves on Linux).  No translation on
the plugin side.

The plugin runs as LocalSystem (`DEPLOYMENT.md` session-0 rule 1): it
cannot see `Z:`.  Read on the worker 2026-09-11: the share is
`//192.168.6.12/hdna2` mounted at `/mnt/hdna2/data/`, and
`geecs_device_server_data_base_path = Z:/data/`.  The plugin needs the
UNC form, so the worker's `config.ini` gains one key,
`[Paths] geecs_pva_plugin_data_base_path` (the UNC root the service can
write, e.g. `\\192.168.6.12\hdna2\data`), used by the plugin path
provider and falling back to `geecs_device_server_data_base_path` when
absent.  Facility value, one home (root `CLAUDE.md`).  **Unverified until
step 2 of the rollout:** that the camera servers' machine accounts can
*write* to `data` on that share (they are known to *read* `software`,
where the fleet clone lives).

### 2.5 The path provider is called with the ophyd name; the directory is the GEECS name

`ADHDFDataLogic` calls `path_provider(self.name + datakey_suffix)` — the
lowercase ophyd name (`uc_amp4_ir_input`).  The directory analysis
readers build is the GEECS device name (`Scan005/UC_Amp4_IR_input/`,
`ScanPaths.build_device_file_map`).  A ~20-line per-detector provider
(`PluginPathProvider(shared, device_name, unc_root)`) wraps the shared
`GeecsScanPathProvider`: it ignores the datakey, asks the shared provider
for `<run>/<GEECS device>/`, and returns the two paths of §2.4 with
`filename = <GEECS device>` so the stock template `%s%s.h5` yields
`UC_Amp4_IR_input/UC_Amp4_IR_input.h5` — the daemon's file name, so the
read side keeps its lookup.

## 3. PV contract (what the plugin serves per image variable)

Prefix: `pv_name(experiment, device, variable) + ":hdf1:"` — the naming
contract mints the lowercase GEECS part; the suffixes are the
areaDetector names verbatim (mixed case), because they are the stock
`NDFileHDF5IO` contract and renaming them would mean an IO subclass of
our own.  Example: `undulator:uc_amp4_ir_input:image:hdf1:Capture_RBV`.
Per image variable, not per camera (a camera with two image variables gets
two plugins and two data logics, the second with `datakey_suffix =
"-<variable>"`), answering #806's Q3.

The set is exactly what `NDFileHDF5IO`'s inheritance chain connects (the
connector connects every annotated signal, so every one must exist):

| Group (`_io.py`) | PVs | Type | Plugin behaviour |
|---|---|---|---|
| `NDArrayBaseIO` | `PortName_RBV`, `UniqueId_RBV`, `ADCoreVersion_RBV`, `DriverVersion_RBV` | s / i / s / s | constants (`UniqueId_RBV` = frames received) |
| | `NDAttributesFile` (+ no RBV) | s, RW | **served as XML** declaring the attribute datasets (§4); a put is accepted and ignored — the attributes are the plugin's |
| | `ArraySizeX/Y/Z_RBV`, `ArraySize0/1/2_RBV`, `ArrayCounter` + `_RBV` | i | geometry of the last decoded frame (Z = 0); counter = frames written this session |
| | `ColorMode_RBV`, `DataType_RBV` | NTEnum | choices `["Mono"]` and the ten `ADBaseDataType` names (both `SupersetEnum` → served choices must be a subset of the class's) |
| `NDPluginBaseIO` | `NDArrayPort` + `_RBV`, `NDArrayAddress` + `_RBV`, `QueueSize` + `_RBV` | s / i / i | stored, readback echoed; `NDArrayPort` = the image variable |
| | `EnableCallbacks` + `_RBV` | NTEnum `Enable`/`Disable` | stored; when `Disable`, frames are counted as skipped |
| `NDFileIO` | `FilePath` + `_RBV`, `FilePathExists_RBV` | s / ? | `FilePathExists_RBV` = `isdir` of the put value, evaluated on the put |
| | `FileName` + `_RBV`, `FileTemplate` + `_RBV`, `FullFileName_RBV` | s | `FullFileName_RBV` = the file once opened |
| | `FileNumber`, `AutoIncrement`, `CreateDirectory` | i / ? / i | stored; **`CreateDirectory` is accepted and ignored — the plugin never creates directories** (the scanner claims the run folder, the detector's `prepare` creates the device leaf inside it, exactly as `LvNativeFileDataLogic` does; answers #806's Q2 "engine-side") |
| | `FileWriteMode` + `_RBV` | NTEnum `Single`/`Capture`/`Stream` | only `Stream` is implemented; another value fails the next `Capture=1` with `WriteStatus` |
| | `NumCapture` + `_RBV`, `NumCaptured_RBV` | i | `0` = unbounded (the stock logic always sets 0); `NumCaptured_RBV` posts after each frame is **on disk** |
| | `Capture` + `_RBV` | ? | the session (§4); `Capture_RBV=1` only after the first frame is seen (§2.3); `Capture=0` closes the file, posts `_RBV=0` |
| | `ArraySize0`, `ArraySize1` (no RBV) | i | H, W of the open stack |
| `NDFileHDF5IO` | `Compression` + `_RBV` | NTEnum (the eight `ADCompression` names) | `None` → raw; `zlib` → gzip-1 + shuffle (built-in filters); others fail the next `Capture=1` |
| | `NumFramesChunks` + `_RBV`, `ChunkSizeAuto` + `_RBV`, `NumExtraDims` + `_RBV`, `PositionMode` + `_RBV`, `LazyOpen` + `_RBV`, `XMLFileName` + `_RBV` | i / ? / i / ? / ? / s | `NumFramesChunks` defaults to **1** (whole-frame chunks; the stock logic reads it and uses it as the resource's chunk shape); the rest stored and echoed |
| | `SWMRMode` + `_RBV`, `FlushNow` | ? / ? | `SWMRMode` accepted, **never** mapped onto HDF5 SWMR writes (§5); `FlushNow` flushes the file |
| **GEECS** | `Rewind` | i | §2.1 — put *N*: truncate to *N*, `NumCaptured_RBV = N`, stale watermark = now |
| **GEECS** | `WriteStatus`, `WriteMessage` | i / s | 0 / "" while healthy; the last writer error otherwise — read by the run's `stop` check and the fleet screen |

Bools are `NTScalar('?')` (ophyd-async's PVA backend maps `bool` to it
directly); ints `NTScalar('i')`; strings `NTScalar('s')`; enums `NTEnum`
with the exact choice strings above (`StrictEnum` classes require set
equality, `SupersetEnum` a subset).  About 55 PVs per image variable,
built from one table.

`GeecsHdfIO(NDFileHDF5IO)` on the worker adds the three GEECS signals
(`rewind`, `write_status`, `write_message`) as `PvSuffix` annotations —
three lines; the stock class is otherwise used as is.

## 4. Plugin behaviour (`geecs_pva_gateway/file_plugin.py`)

One `HdfFilePlugin` per (camera worker, image variable).

**Intake** (event loop, in `_on_frame`, before the latest-wins stash):
`plugin.offer(blob, stamp, recv_time)` → a bounded `queue.Queue`
(64 raw blobs).  Full queue = the writer is wedged: the frame is counted
as a queue drop, `WriteStatus` goes non-zero, and the count stops
advancing, so the shot times out in 3 s and the run fails naming the
camera.  Nothing ever backlogs silently.

**Session** (`Capture=1` → `Capture=0`), on the writer thread:

1. Retain the variable's subscription through the worker's existing
   refcount (`retain(var)`), the same gate a PVA client holds — the
   plugin is a client of the gateway in everything but transport.
2. Wait for the first decoded frame (§2.3); post geometry and dtype
   RBVs; post `Capture_RBV=1`.
3. `stale_before = time.time()` at that moment.  A frame is written iff
   `stamp > stale_before - 0.1 s` (skew margin; ARMED precedes `prepare`
   by hundreds of milliseconds, so no real shot can be older) **and**
   its stamp has not been seen this session (dedupe: LabVIEW re-pushes
   the last frame at 1 Hz when idle, measured 2026-08-27).  Both counted.
4. First accepted frame opens the file (`LazyOpen` semantics; a session
   that accepts nothing leaves no file, and the stock provider then emits
   no stream resource, so no document references a missing file).
   Geometry is fixed by that frame; a later frame of another shape is a
   shape error (counted, not written, `WriteStatus` set).
5. Per frame: decode (`decode_imaq_image_string`, on this thread — the
   gateway's executor decode serves PVA clients; sharing it would couple
   the two delivery contracts), append, flush, post `NumCaptured_RBV`.
6. `Rewind=N`: resize both datasets to *N*, post `NumCaptured_RBV=N`,
   `stale_before = now`.  Discarded stamps stay in the seen set so their
   re-pushes dedupe.
7. `Capture=0`: write the reconciliation counters as root attributes
   (`frames_received == written + duplicates + stale + shape_errors +
   queue_drops + rewound`), `finalized = True`, close, release the
   subscription, post `Capture_RBV=0`.

**File** (`h5py`, `libver="latest"`, `locking=False`):
`/entry/data/data` `(N, H, W)` chunks `(1, H, W)`, maxshape unbounded,
compression per the PV; `/entry/instrument/NDAttributes/acq_timestamp`
and `.../recv_timestamp` `(N,)` float64, chunks `(16384,)` — the chunk
shape ophyd-async declares for attribute datasets, so the consolidator's
structure matches the file without its "fixing chunk shape mismatch"
warning.  Root attributes: `device`, `experiment`, `source_pv`,
`created`, the counters, `finalized`.  `NDAttributesFile` serves
`<Attributes><Attribute name="acq_timestamp" type="PARAM"
datatype="DOUBLE" .../>…</Attributes>`, which `get_ndattribute_dtype_source`
parses into the two attribute stream resources (`<f8`, chunk `(16384,)`).

**Provenance across restarts:** none needed.  The plugin holds no
per-scan state between sessions; the worker tells it where to write.

## 5. HDF5 over SMB — the rules

1. **Never HDF5 SWMR across SMB.**  `SWMRMode` is accepted so the stock
   logic's `swmr_mode.set(True)` succeeds, and ignored.  The writer
   flushes after every append (a crash loses the unflushed tail, never
   the scan).
2. **Never read a file while it is being written.**  Analysis reads after
   `Capture=0`; live use is the PVA stream (#744).  A contract, not
   best-effort (03 §12).
3. **Writer:** `locking=False` on open (the HDF5 lock over SMB is the
   known failure mode).  **Readers on Linux:** the consolidator passes
   `swmr=True` by default (`consolidators.py:361`), and Tiled opens with
   `swmr=True, libver="latest"` — verified 2026-09-11 (h5py 3.16, HDF5
   2.0): a closed `libver="latest"` file opens that way; truncation by
   `resize` works.  File locking on the Tiled host is disabled by
   `HDF5_USE_FILE_LOCKING=FALSE` in its unit environment — a site.env /
   unit edit, which per §10.5 of 03 waits for the end-of-branch
   deployment touch; the acceptance run sets it in the shell.
4. **Chunking** (#806 Q1): one frame per chunk.  The analysis pattern is
   per-shot random access (`read_shot_for_acq_timestamp`) and whole-stack
   reads; whole-frame chunks serve both, and the resource's chunk shape
   is what the stock logic reports.  **Compression:** ships `None`, so
   the parity phase compares like with like against the PNGs; the
   daemon measured shuffle + gzip-1 at 2.0 MB/frame vs 7.9 raw vs 2.5
   PNG at 3.9 ms/frame, so `zlib` is the expected setting once the
   parity diff is clean — flipped by a put, not a release.

## 6. Worker side (GeecsBluesky)

- **Namespace rule.** A `looks_triggerable` device is plugin-backed iff
  the DB lists an image-typed variable for it (`effective_vartype ==
  "image"`, the gateway's own camera test) **and** its endpoint host is
  in `config.ini [pva] file_plugin_addr_list` (absent → `[pva] addr_list`,
  the deployed gateways).  The list exists because the rollout is per box
  (h5py is a re-bootstrap, `DEPLOYMENT.md` "External deps are frozen at
  bootstrap"); a camera on a box not yet rolled keeps `LvNativeFileDataLogic`.
  `DeviceRoster` gains `endpoints` (device → ip, from the same
  `get_experiment_devices` call the PVA gateway's config uses).
- **Detector.** `GeecsDetector(..., hdf_plugins=[(variable, prefix)])`:
  per plugin one `GeecsHdfIO(f"pva://{prefix}")` child and one stock
  `ADHDFDataLogic(array_description=NDArrayDescription([io.array_size_z,
  io.array_size_y, io.array_size_x], io.data_type, io.color_mode),
  path_provider=PluginPathProvider(...), driver=io, writer=io)` — the
  writer IO is also the "driver" the logic reads geometry and the
  attribute XML from, since `NDFileHDF5IO` inherits every signal
  `NDArrayDescription` needs.  `native_save` stays for the LabVIEW-native
  devices and for the not-yet-rolled cameras; **a plugin-backed camera
  still clears a stale `save=on` at stage** (the crash case found live
  26_0828) but never turns it on.
- **`discard_uncollected()`** and the refire hook (§2.1); the count
  timeout translation and `exposure_timeout` (§2.2).  The plan calls the
  method through the stock `bps.wait_for` stub — no custom message.
- **Telemetry** (`GeecsNamespace.telemetry()`): the plugin IO's signals
  are never baseline columns (they are the detector's, not the
  experiment's scalars) — the existing "detector's scalar signals" walk
  excludes children that are `Device`s, verified against
  `_scalar_signals()`.
- **Documents.** The stream resource is stock (`application/x-hdf5`,
  `dataset=/entry/data/data`, `chunk_shape=(1, H, W)`); the descriptor's
  data key is `<name>` (no suffix) with `external="STREAM:"`.  `prepare_
  descriptor_for_tiled` keeps stripping only `geecs://` keys, so the
  plugin's key reaches Tiled as a stream.  `tiled_schema.COMPANION_
  SUFFIXES` / `EVENT_SCHEMA.md` gain the stream key beside the legacy
  `-nonscalar_save_path` column.
- **Read side** (`geecs_data_utils.io.scan_stack`): dataset paths become
  the areaDetector ones; `is_stack_file` tests for `/entry/data/data`
  instead of the `geecs-capture` schema attribute.  Callers (ImageAnalysis
  `ShotRef` loading, ScanAnalysis stack mapping, the portal's resources)
  go through that module's functions and are untouched.

## 7. Deletions (same PR)

`geecs_bluesky/capture/` (daemon, `__main__`, heartbeat, discovery,
subscriber, writer, `FORMAT.md`), `tests/capture/`, the `capture` extra,
`pyzmq`, the two `geecs-capture-*` scripts, the daemon's systemd unit
and its `DEPLOYMENT.md`/`fleet_map.md`/`site_profile.md` rows.  `p4p`
moves under the `ca` extra (the worker talks PVA to the plugin; `ca` is
"the gateway clients", which is what the extra always meant), so the
deploy install line does not change.  `capture/diff.py` becomes
`geecs_pva_gateway/diff.py` (`geecs-pva-gateway diff`, the parity tool
of rollout step 4; it needs only `geecs_data_utils`, already a
dependency).  The `NonScalarSaveSupport` `geecs://` asset docs and the
Tiled descriptor patch stay until PNG retirement (#738).

## 8. Rollout and acceptance (hardware, after the PR merges into the feature branch)

1. **Share clone → feature branch.**  The fleet pin is the share clone's
   commit; the plugin ships inert (no PVs unless `h5py` imports, no
   detector uses it unless the host is listed).
2. **Re-bootstrap one camera server** (adds `h5py` to the venv; a
   console-session step per `DEPLOYMENT.md`); confirm the plugin PVs
   appear (`pvget …:hdf1:Capture_RBV`) and `FilePathExists_RBV` reads
   true for a UNC run folder (the write-permission question of §2.4).
3. **Worker:** add the host to `[pva] file_plugin_addr_list` and the UNC
   root to `[Paths]`; restart `geecs-qserver`; the acceptance test
   (`tests/test_phase1_hardware.py`, extended: `GEECS_HW=1`, HTU-NoGas,
   amp4in — `UC_Amp4_IR_input` is on the rolled box or the first rolled
   box is chosen for it) asserts the stack, its frame count = rows, the
   stamps = the rows' `acq_timestamp`, the stream resource/datum
   documents, and a Tiled read of the run's image array.
4. **Parity:** several scans with PNG dual-write on; `geecs-pva-gateway
   diff` against the PNGs.
5. Roll the fleet box by box; retire the daemon's unit.  Then (phase 2 or
   later) `LvNativeFileDataLogic` narrows to the non-image devices and the
   `file_plugin_addr_list` key goes.

## 9. Residual risks, named

- Machine-account write access to the data share (§2.4) — decides whether
  the service needs the shared domain account (`DEPLOYMENT.md` fallback).
- CA/PVA monitor coalescing of `NumCaptured_RBV` under load: the count
  wait observes a monotonically increasing integer and compares `>=`, so
  a coalesced update cannot lose a shot, only delay noticing it.
- Camera-server clock lag > 0.1 s would stale-drop real first frames —
  attributable through `stale_skipped`, as the daemon's FORMAT.md noted.
- `Compression=zlib` on a 4 MB frame costs ~4 ms of the writer thread per
  frame; fine at 1 Hz × a few cameras per host, to be re-measured before
  flipping it on.

## 10. Test plan (offline, both packages)

- Gateway: the PV table (every `NDFileHDF5IO` suffix served, types and
  choices as ophyd-async expects — asserted by connecting a real
  `GeecsHdfIO` over `pva://` against an `isolate=True` server, which pins
  the contract on both sides in one test); a session over the fake camera
  (gated subscription, first-frame ack, dedupe, stale, shape error,
  rewind, counters, no directory creation, no file when nothing accepted);
  the file's layout read back through `h5py` and through
  `geecs_data_utils.io.scan_stack`.
- Worker: `GeecsDetector` with the plugin logic on mock signals (stage →
  prepare → trigger → collect → unstage document shapes); the refire hook
  calling `discard_uncollected`; the count-timeout translation; the
  namespace rule with and without the host list; the path provider's two
  paths; the read-side path change.
