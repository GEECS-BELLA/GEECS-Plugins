# Central PVA image capture — audit & scope

*Drafted 2026-08-27 from a four-surface codebase audit (PVA gateway, GeecsBluesky
engine, GEECS-Schemas + GEECS-Core DB, read-side consumers). Status: scoping
document — nothing here is built. Decisions marked (Sam) were made in the
originating session; everything else is proposal.*

## The feature

During a Bluesky scan, a capture daemon on the Linux services box subscribes to
the Point Grey cameras' NTNDArray PVs (served by the per-camera-server
GeecsPvaGateway fleet), buffers frames in RAM, and writes **one HDF5 file per
device per scan** into `scans/ScanNNN/<device>/`. Point Grey cameras are ~90%
of scan data; devices with proprietary formats (HASO `.himg/.has`, scope
traces) keep the legacy per-shot file save via per-device configuration.

Motivating measurements (BELLAAPPSERVER → netapp `hdna2`, 2026-08-27):

- A real scan (26_0826 Scan006) is **21,407 files / 6.7 GB** — file *count* is
  the pathology (~0.32 MB average).
- Sequential small-file read: 14.8 MB/s (~22 ms/file latency). Single big
  file: 43.9 MB/s. 16-way parallel robocopy: 38.4 MB/s → the share has a
  **hard ~40 MB/s path cap, access-pattern independent** (question open with
  IT; client NIC is 10G and carries the SMB session).
- Per-device HDF5 collapses ~21k files → ~1k (the proprietary residue), and
  RAM-resident capture removes the NAS from the fresh-scan analysis path
  entirely.

**Transition doctrine (Sam):** phase capture in *alongside* the untouched LV
file save — PNGs keep writing, HDF5 duplicates them, and the diff between the
two is continuous validation. PNG deprecation is a later, per-device decision
gated on accumulated zero-mismatch evidence. Phase 1 therefore needs **no
save-set schema change and no engine behavior change.**

## The two gates (unknowns that block design lock)

**G1 — LabVIEW push semantics.** The gateway's upstream feed is the device's
GEECS TCP subscription (`Wait>>image,acq_timestamp,systimestamp`; framed,
lossless transport — `geecs_core/transport/tcp_subscriber.py:277-348`). Unknown
whether the LV device pushes once per acquisition or on a state timer. If the
source never emits a frame per shot, no downstream work makes capture lossless.
The wire **carries a shot counter** the gateway currently discards
(`include_shot=True` exists since geecs-ca-gateway 0.18.0,
`tcp_subscriber.py:406-410`; the PVA gateway subscribes without it,
`geecs_pva_gateway/server.py:166-170`). A standalone subscriber script logging
shot-counter gaps during a real scan settles G1 without touching the gateway.

**G2 — p4p client-side monitor queue.** Even a lossless gateway can lose
frames in p4p's per-subscriber MonitorFIFO (depth negotiated by the *client's*
pvRequest; in-repo evidence of squashing at
`GeecsPvaGateway/tests/test_server.py:198-199`). Needs an empirical test: deep
client queue (`record[queueSize=N]`) against a camera at rate. If deep client
queues are honored, the cheapest design (single PV, lossless server-side,
viewers self-squash via shallow queues) works; if not, a second queued-mode PV
per variable (e.g. `…:image:all`) is required.

Both gates are answered by **Phase 0** below — one lab afternoon.

## Phase 0 results (2026-08-27, live — G1/G2 PASS at 1 Hz)

Run: Scan001 26_0827 (strict noscan, 10 shots, save set Amp4In, trigger
HTU-NoGas, `UC_Amp4_IR_input`), probes on the interim Linux worker box
(lab network, where the daemon will live). Reconciliation:

| Leg | In-window frames | acq_timestamp match vs PNGs |
|---|---|---|
| LV PNGs (ground truth) | 10 | — |
| G1 device TCP feed | 10 distinct | exact, all 10 |
| G2 PVA deep (`queueSize=100`) | 10 distinct | exact, to the ms |
| G2 PVA shallow (default) | 10 distinct | identical to deep |

**Verdicts.** G1: the LV feed emits every triggered frame (timestamp
proof; the wire shot counter also advanced 0→10 across the run — 11
values including the pre-scan 0 — but is Master-Control-only per the
owner: key on timestamps). G2: no client-side squash at 1 Hz —
even the default queue delivered everything; the unmodified gateway is
empirically lossless at HTU's operating rate. The Phase-1 bounded-queue
work is therefore **margin engineering for bursts/5 Hz, not a
prerequisite**; counters/observability remain the substantive Phase-1
content.

**Secondary findings.**
1. **Idle re-push**: after a scan the device resumes the 1 Hz state push
   re-sending the last frame with an *unchanged* acq_timestamp; the
   gateway re-posts it (3 duplicate posts observed post-scan). The
   capture daemon MUST dedupe on acq_timestamp (already the join-key
   doctrine).
2. **Remote TCP image payloads are empty even during acquisition**
   (0 bytes on every update, scan included), while the gateway's
   host-local subscription receives full blobs — images ship only to
   local subscribers. Direct remote-TCP capture is confirmed non-viable;
   PVA is the only capture path (as designed).
3. The device pushes state at exactly 1 Hz continuously, scans or not —
   scan-gated capture must window by timestamp/doc-stream boundaries,
   not by "frames are flowing".

**Statistical + load run (same day): Scan002 26_0827, 200-shot strict
noscan, with deep PVA monitors on ALL 11 cameras hosted by the same
server (192.168.6.100, the Amp-chain camera server).** Results: 200 PNGs ground
truth; G1 200 distinct acq_timestamps; scanned camera's PVA stream
200/200 exact; **every one of the other 10 cameras also delivered
200/200 distinct frames in-window** (all free-run at 1 Hz continuously)
— ~2,200 frames across 11 concurrent streams, zero drops, max
inter-delivery gap 1.52 s. Camera-server load measured over ssh during
the run: **~9% total CPU, gateway python process 172 MB working set,
9 GB / 16.7 GB RAM free** — sustained full-fleet subscription is a
non-event for the host. Aggregate egress ~8 MB/s (11 × 600×600×2B ×
1 Hz), as predicted.

Raw probe records: `/tmp/probe-run/results/` + `/tmp/probe-run2/` on
the worker box (session scratchpad copies kept). Still not measured:
MonitorFIFO behavior at >1 Hz burst (only relevant for the 5 Hz
future); camera-server load while *saving* 11 cameras' PNGs
simultaneously (the dual-write condition — fold into Phase-2
acceptance with a purpose-built all-cameras save set).

## Where loss and blindness live today (audit facts)

- Gateway-side loss has two modes: the primary drop point is
  `_CameraWorker._on_frame`/`_publish` (`server.py:199-228`) — a
  latest-wins **dict slot** per variable, depth 1, overwrite silent — and
  separately a frame whose decode or post raises is logged and skipped
  (`server.py:214-228`), i.e. decode/post failure is a loss mode of its
  own, not just telemetry blindness (both get counters in Phase 1). Documented as contract ("Completeness lives in the GEECS
  file path, not this stream" — `GeecsPvaGateway/CLAUDE.md:66-71`). This
  feature revises that written contract.
- A capture client **cannot detect a gap today**: NTNDArray `uniqueId` is
  hardcoded 0 (p4p `NTNDArray.wrap` — line varies by p4p version), no
  NTAttributes populated, timestamp
  repeats if the device repeats it.
- **Zero telemetry**: no frames-received/dropped/decode-failure counters
  anywhere; only three instance PVs (version/heartbeat/restart —
  `server.py:287-298`). `:restart` is unauthenticated.
- Gating: first subscriber activates the upstream TCP subscription (~1–2 s
  round trip); first monitor update is a **stale cached frame or a (1,1)
  zeros placeholder** (`server.py:92`). The capture daemon must pre-arm
  subscriptions before the scan and discard the first update.
- Incidental bug found (filed separately): `_frame_timestamp` checks
  plausibility **before** subtracting the LabVIEW epoch offset
  (`server.py:47-53`) — small positive LV values become negative Unix
  timestamps instead of falling back.

## Eligibility: reconciling two predicates

Two non-identical definitions of "camera served over PVA" exist:

1. **What the gateway serves**: devices with ≥1 variable whose
   `effective_vartype == "image"` AND whose DB endpoint IP is local to a
   fleet host (`geecs_pva_gateway/config.py:58-127`).
2. **The devicetype string**: `"Point Grey Camera"`
   (`geecs_bluesky/assets/specs.py:11`), used by the asset registry.

The capture-eligibility predicate should be the **conjunction**: devicetype in
the capture registry AND actually PVA-served (fleet-host check). Missing
plumbing, both trivial:

- Batch DB query `get_experiment_device_types(experiment)` in
  `geecs_core/db/geecs_db.py` (mirror `get_experiment_devices` at `:528-556`;
  today devicetype is single-device-per-connection only, `:320-343`; test
  pattern at `GEECS-Core/tests/test_geecs_db.py:98-129` already uses the
  Point Grey fixture value).
- A failure-tolerant provider in `geecs_bluesky/db_runtime.py` (mirror
  `GeecsDbScalarPolicy` `:64` — DB blip degrades, never aborts).

Capture-box discovery: no manifest PV exists; the daemon re-derives
device→PV mappings from the same DB queries the gateway uses
(`config.py:58-127`) and needs `EPICS_PVA_ADDR_LIST` carrying the fleet IPs
(subnet-spanning; `DEPLOYMENT.md:160-196`).

## Phases

### Phase 0 — probes (1 lab afternoon + a prep session)

1. **LV semantics** (G1): standalone `GeecsTcpSubscriber` script with
   `include_shot=True` on one camera during a real 200-shot strict scan; log
   shot-counter sequence. No gateway changes.
2. **MonitorFIFO** (G2): deep-queue pvRequest client against the current
   gateway at rate; measure delivered/dropped.
3. **End-to-end count**: PVA client captures during the same scan; diff frame
   count + shot IDs against the LV-written PNGs.
4. Watch camera-server CPU/RAM during sustained subscription (the
   gated/latest-wins design partly protects those boxes; sustained capture
   load is new).

Output: probe report → design-lock addendum to this doc.

### Phase 1 — lossless-capable gateway (GeecsPvaGateway, small-medium)

- Bounded `asyncio.Queue` per variable + consumer task replacing the
  `_latest` slot (`server.py:199-228`) — **bounded, with a PV-exposed drop
  counter; silent unbounded growth on a camera server is worse than the
  current drop.**
- `include_shot=True` in the upstream subscribe; populate NTNDArray
  `uniqueId` (or a shot NTAttribute) so gaps are detectable downstream.
  This changes externally observable NTNDArray content — PV_CONTRACT/CLAUDE
  truth-up required.
- Telemetry counter PVs per variable (received/posted/dropped/decode-failed)
  in the instance-PV block (`server.py:287-298`).
- One-PV-lossless vs second `:image:all` PV: decided by G2.
- Fleet rollout is cheap (pull-on-restart `deploy/launch.bat`); note the
  unauthenticated `:restart` PV — a mid-scan restart kills capture silently;
  capture side must detect (heartbeat/connection loss).

### Phase 2 — capture daemon (new component, medium-large)

**STATUS: v0 BUILT, REVIEWED (adversarial 14 findings + codex 2 P1s, all
dispositioned), and LIVE-ACCEPTED 2026-08-27 (PR #694, merged).**
GeecsBluesky 0.64.0 `geecs_bluesky/capture/` + GEECS-Core 0.4.0 batch
devicetype query. Acceptance: Scan003 26_0827 dual-write — all 10 LV PNG
acq_timestamps present in the HDF5 (zero missing), counter identity
closed exactly, one attributable pre-save-window extra (free-running
camera), lazy-writer creation validated live (create_fail=2 pre-mkdir
frames, no shot lost). Key build-time discovery (review HIGH): the
engine creates device dirs AFTER the start doc → writers are lazy on
first accepted frame. Remaining Phase-2 items: RAM-buffer analysis
handoff, systemd unit, the all-cameras save-set stress test.

- **(Sam) Lives in GeecsBluesky** — the daemon code as an importable
  subpackage behind an optional extra at `geecs_bluesky/capture/` (the
  `qs_client` pattern), accepting that its release cadence rides with the
  engine's; launch/deploy assets top-level beside `qserver/` (that dir
  pattern is non-package tooling — the Phase-0 probes at
  `GeecsBluesky/capture/probes/` already live that way); systemd unit on
  the services box (template: `qserver/deploy/`).
- **(Sam) Scan-gated subscriptions** — subscribe on the scan boundary
  (pre-arm when a scan enters the queue where possible), discard the stale
  first frame, release after stop; preserves the fleet's idle-when-unwatched
  design. RAM buffer with **trailing flush**, not hold-then-write (crash
  window = flush lag, not the scan).
- Scan-boundary awareness via the existing 0MQ document stream
  (`bluesky-0MQ-proxy` out-port, `qserver/launch_re_manager.sh`;
  consumer precedent: `sfile_callback.py`). Start doc supplies scan folder +
  `nonscalar_save_paths`; stop doc triggers flush finalization.
- Writes `scans/ScanNNN/<device>/<device>.h5`. **Invariant: the daemon never
  creates `scans/ScanNNN/`** — only the engine's claim does
  (`plans/run_wrapper.py:43`); device dirs are created by the engine's
  save-enable plan (`run_wrapper.py:121`), which keeps running in the
  dual-write phase.
- Serves its own PVs: per-camera frames-captured counters (consumed by
  Phase 5), daemon heartbeat.
- **(Sam) HDF5 stores decoded arrays**: `(N, H, W)` uint16/uint8 dataset
  chunked per shot, aligned shot-number + timestamp datasets, device metadata
  as attrs, light compression (shuffle+gzip-1, empirically chosen 2026-08-27) —
  consumers read arrays directly, no IMAQ
  decode at read time.
- **(Sam) No container handcuffs**: HDF5 is the v0 *implementation* behind
  the `FrameStackWriter` protocol (`geecs_bluesky/capture/writer.py`), not
  the contract. The contract = `capture/FORMAT.md` + the `schema` attribute
  stamped in every file (`geecs-capture/1`); a future container (e.g. Zarr)
  is a new writer implementation + a reader branch, no daemon changes.
  GEECS-Schemas stays out of it (it owns *config* vocabulary; data-file
  contracts follow the `EVENT_SCHEMA.md` precedent — a versioned doc beside
  the code, schema key in the data). Revisit a `capture_mode` format field
  only if Phase 3 shows a real need.
- **Dual-write validation tool**: per-scan diff (HDF5 vs PNGs — count, shot
  IDs, optionally pixel checksum). Runs continuously; its accumulated record
  is the deprecation evidence.

### Phase 3 — engine + schema integration (medium) — REDESIGNED 2026-08-27

**(Sam) The devicetype toggle replaces `capture_mode`.** Insight: the LV
per-shot save infrastructure is permanent regardless (HASO, scopes —
proprietary formats keep `save=on` forever), so image-saving ownership
never actually needs per-save-set-entry vocabulary. The only real
decision is one devicetype-scoped switch: *do Point Grey cameras write
native PNGs, or does the capture daemon own their images?* The save
set's `images: true` keeps meaning "this device's images are wanted";
the toggle changes who writes them for capture-eligible devicetypes.
This deletes the former plan's `SaveSetEntry.capture_mode` reserved
field, its `compose_save_sets` merge semantics, and Phase 6's
per-device deprecation choreography.

- **Schema**: `native_image_save` (name TBD) — experiment-wide default in
  `ExperimentDefaults`, tri-state per-scan override on `ScanRequest`
  (`None` = inherit; the `background_telemetry` pattern,
  `scan_request.py:546-558`). Default ON — nothing changes until an
  experiment flips it on dual-write evidence; the per-scan override is
  the operational escape hatch. Per-camera opt-out deliberately NOT
  built (all capture-eligible cameras switch together); a per-entry
  override can be added additively if a single camera ever needs it.
- **Eligibility provider**: failure-tolerant db_runtime provider over
  `GeecsDb.get_experiment_device_types` (built, GEECS-Core 0.4.0)
  intersected with the capture registry (`CAPTURE_DEVICE_TYPES`).
- **Engine seams** (the three couplings that make this "a toggle plus
  three seams"):
  1. With native save off, PG cameras leave `nonscalar_save_paths` (it
     lists LV-*saving* detectors) — the engine must publish the capture
     list explicitly: a `capture_save_paths` start-doc md key; the
     daemon prefers it, falling back to `nonscalar_save_paths` ∩
     registry (backward compatible).
  2. Device-dir creation: `save_enable_plan`'s `makedirs`
     (`run_wrapper.py:121`) only runs for saving detectors — the engine
     must still create dirs for captured-unsaved cameras (engine-side,
     invariant intact; the daemon still never mkdirs).
  3. Asset documents: `NonScalarSaveSupport`'s save-path column +
     PNG-pointing Resource/Datum docs are suppressed for
     captured-unsaved devices (or switched to the HDF5 spec below).
- Capture-availability preflight check (free-form check name in the
  existing `PreflightOutcome` vocabulary — `scan_request.py:417-421`) —
  refuse/warn when the toggle is off but the daemon looks absent.
- Tiled: new `GEECS_HDF5_STACK` asset spec + registry entry + per-shot
  Datum (`datum_kwargs={"frame": i}`) or run-scoped StreamResource;
  **relax the descriptor patch** at `tiled_integration.py:54-67` for the
  new spec (today it strips external registration or the run aborts on
  stop).

### Phase 4 — read side (medium)

**STATUS: BUILT, REVIEWED (adversarial 8 findings + codex clean, all
dispositioned), MERGED 2026-08-27 (PR #696).** geecs-data-utils 0.14.0
(`io/scan_stack.py` + `ShotRef`), ImageAnalysis 1.12.0 (`load_image`
ShotRef branch), ScanAnalysis 1.17.0 (opt-in `data_format: device_hdf5`
with unconditional per-shot-file fallback — default byte-for-byte
unchanged). End-to-end verified against the real Scan003 daemon stack.
Compression landed same day (PR #695: shuffle+gzip-1 — stacks now
~20% smaller than the PNG set). Known opt-in hazard (documented in the
field docs): analyzers deriving per-shot output names from
`file_path.stem` (MagSpec, LineStitcher) must not opt in — a runtime
guard is a follow-up candidate. Deferred to the stack-only world:
background pre-pass enumeration, `infer_device_ext`/expected-asset
guards, per-bin bulk reads.

- `geecs_data_utils/io/scan_stack.py`: `read_shot(path, shot)`,
  `list_shots(path)`, open-handle cache; naming/index contract documented in
  `native_files.py` (the declared source of truth) alongside the two existing
  conventions.
- **Highest-leverage edit**: third strategy in
  `SingleDeviceScanAnalyzer._build_data_file_map`
  (`single_device_scan_analyzer.py:296-366`) — probe for the stack file
  first; existing timestamp/shot-number strategies untouched (HASO/scope
  devices never see it). `ShotRef` design: a `Path` subclass carrying the
  shot index keeps the worker submit sites (`:788-793`, `:830-833`) and
  `analyze_image_file` signatures unchanged.
- `ImageAnalyzer.load_image`/`analyze_image_file` (`base.py:89-149`) route
  `ShotRef` → stack reader; `Standard1DAnalyzer.analyze_image_file`
  (`standard_1d_analyzer.py:90-145`) needs the same treatment (bypasses
  `load_image`).
- Background pre-pass enumeration branch
  (`processing/array2d/background.py:370` glob).
- Guards: `ScanPaths.infer_device_ext` returning `h5` makes
  `_append_expected_asset_columns` fabricate bogus paths
  (`scan_data.py:914-1006`) — teach it the stack case or skip stack devices.
- Per-device gradual switchover; PNGs remain the fallback until Phase 6.

### Phase 5 — capture in the strict contract (small-medium)

- Per-shot check between `bps.wait` and `create`
  (`plans/single_shot.py:146-182`) reading the daemon's frames-captured
  counter; must distinguish "no frame, refire helps" from "writer stalled,
  refire won't" (pattern: `_confirm_device_down` `:56` +
  `plans/liveness.py:29`). Never insert between `create` and `save`.
- `capture_policy` knob threaded through the exact `failed_move_policy`
  chain (`plans/step_scan.py:201` → `orchestration.py:83` →
  `session.py:654` → `scan_request_plan.py:625`).

### Phase 6 — PNG deprecation (small — one toggle flip per experiment)

REDESIGNED with Phase 3 (2026-08-27, Sam): deprecation is no longer
per-device choreography — it is flipping the Phase-3 `native_image_save`
experiment default to off, after the evidence gate.

- Gate: N weeks of dual-write with zero diff mismatches (the Phase 2 diff
  CLI's record).
- **HARD precondition (review of PR #697, finding 2): an engine-checkable
  daemon-liveness signal** before the experiment default ever flips —
  today the engine suppresses native saving blind to daemon process
  death, a disconnected doc stream (missed start doc = whole scan
  uncaptured, zero evidence), or PVA-down-while-CA-healthy. Candidate
  shapes: daemon heartbeat PV consumed by a capture-availability
  preflight check, or a daemon-written liveness marker. Per-scan
  toggle-off use before then is human-supervised: the operator confirms
  the daemon's session-open line names every capture device and
  reconciliation shows frames_written == shots.
- Flip: the `ExperimentDefaults` toggle → PG cameras stop writing PNGs
  (`save_images=False` at `_build_request_detectors`,
  `scan_request_runner.py:1204-1206` — the downstream save-toggle
  machinery skips them automatically, `session.py:1592`); the per-scan
  tri-state override remains the day-to-day escape hatch.
- Probe sunset: `capture/probes/` stays in-tree through the arc (it is the
  independent re-verification instrument for Phases 1–2 and the 5 Hz
  question); once the daemon's telemetry + the dual-write diff tool fully
  absorb that role post-deprecation, prune the probes (the Planning-prune
  ritual) — deliberate, not speculative.

## Effort estimate (honest)

Build sessions at the demonstrated pace (build + adversarial review per PR):
Phase 1 ≈ 1–2, Phase 2 ≈ 2–4 (the daemon is the biggest new artifact),
Phase 3 ≈ 1–2, Phase 4 ≈ 2–3, Phase 5 ≈ 1, Phase 6 ≈ trivial per device.
**Total ≈ 8–12 build sessions.** The scarce resource is **lab time**: Phase 0
(one afternoon), a Phase 2 live validation scan day, a Phase 5 live contract
test, plus the passive dual-write weeks that gate Phase 6. Phases 1–2 are
independent of 3–5 after Phase 0, and Phase 4 can start any time after the
HDF5 schema is fixed (Phase 2 design).

## Standing risks

- G1/G2 outcomes can redirect the whole plan (→ local-staging packer becomes
  the fallback architecture — see session record in project memory).
- NTNDArray content change is a declared external contract; needs truth-up +
  coordinated fleet restart.
- Capture fan-out doubles a camera server's egress when a viewer is also
  attached (modest at 1 Hz; the distributed design exists because of
  bandwidth — the daemon re-concentrates it at the Linux end, where 10G
  absorbs it).
- Queue memory bounds on camera servers are mandatory (multi-MB blobs;
  shared default executor for decode).
- The engine's CLAUDE.md lists a "central subscription daemon (event-builder
  service)" as *deliberately deferred* (`GeecsBluesky/CLAUDE.md:776-781`) —
  this project is that deferral coming due; update the deferral note when
  Phase 2 lands. Keep `shot_id` the universal join key (`:783-785`).
