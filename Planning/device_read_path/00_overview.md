# Device read-path hardening — audit findings & phased plan

**Date:** 2026-07-13.  **Trigger:** live scans (Undulator Scans 1–3) at a
1 Hz trigger ran at exactly 0.5 Hz — camera `acq_timestamp` deltas were
exactly 2.000 s (free-run rows landing on every second trigger tick), and
strict single-shot cadence measured 1.6–2.0 s/shot.

**Target envelope:** 5 Hz (the system limit — laser rep rate; the LV GEECS
device acquisition loop can be raised via a DB setting but ~5 Hz is the
practical ceiling) with ~50 save-set devices plus background telemetry.
Verification happens at 1 Hz (the available trigger rate); the design must
make 5 Hz follow trivially.  Rates of 100 Hz+ are explicitly out of scope —
that is a different framework, not a tuning exercise.

## Root cause (measured, 2026-07-13)

`background_telemetry` defaults on for ScanRequest submissions (the console
path; the legacy exec_config path never ran telemetry), adding ~87 soft
devices / 376 columns to every event row.  The RunEngine reads devices
**sequentially** (one `read` Msg at a time), no plan ever **staged** a
device, and an unstaged ophyd-async `read()` is one network CA get per
signal — so each row paid ~87 serialized round trips (~7 ms each over VPN
≈ 0.7 s), plus the blocking SINGLESHOT fire in strict mode.  Tracked as
issue #540.

## Audit verdicts (four deep-dives, 2026-07-13)

1. **Foundation is sound.** `CaAcqTimestampReadable` is not a hack: it uses
   ophyd-async's own subscription API, and its queue-plus-synchronous-baseline
   `trigger()` solves a real shipped race (`wait_for_value`-style
   subscribe-after-fire windows would reintroduce it).  The systemic gap was
   only that plans never staged.
2. **Staging semantics verified from installed source** (ophyd-async
   0.19.3): `stage()` starts one caching CA monitor per signal; reads then
   serve from memory (0 backend calls, proven with a counting-proxy
   experiment); coexists with the persistent `acq_timestamp` subscription
   (shared, listener-refcounted cache); works identically under mock
   backends; **not refcounted** (stage exactly once); reads stay serialized
   across devices (parallelism exists only within one device's read).
3. **Framework ceiling measured** (mock-backend benchmark through the real
   plans): `ms/event ≈ 0.5 + 0.30·devices + 0.013·columns`.  At the
   50-device target ≈ 16 ms/event including TiledWriter's synchronous work —
   ~8 % of the 5 Hz budget (200 ms/row).  Today's 87-device/785-column
   telemetry shape ≈ 42 ms/event — fine at 5 Hz.  TiledWriter batches
   (10 000-event buffer): HTTP at run open/close, never between shots.
   Device *count* is the dominant framework lever (0.30 ms per read Msg).
4. **Shot coherence of cached reads verified link-by-link** (gateway posting
   order → caproto FIFO circuit → aioca FIFO hop → synchronous cache
   update): at `trigger()` completion, staged data caches hold the
   triggering frame or newer — never the previous frame.  Conditions and
   sharp edges: `GeecsBluesky/CLAUDE.md` "Read path: staging & shot
   coherence".
5. **Strict mode is structurally ≲1–2 Hz** (the SINGLESHOT fire is a
   blocking GEECS set awaited per shot, plus bounded refire budget).
   5 Hz is free-run territory; free-run does zero puts per shot.

## Phased plan

### Phase 1 — staging + rate-derived bounds (LANDED, 0.32.0)

- `build_step_scan_plan` stages every per-row read device (outermost
  `bpp.stage_wrapper`; unstage runs on every exit path including abort).
- Contributor grace wait capped at half a trigger period
  (`FreeRunContributorSupport._effective_grace_wait_s`) — was a fixed 0.3 s
  (3 periods at 10 Hz, 1.5 at 5 Hz, serialized per lagging contributor).
- t0-sync window capped at `0.4 / rep_rate_hz`; effective value recorded in
  the start doc.
- Telemetry per-signal read budget 2 s (was ophyd's 10 s default; also
  covers the staged first-read-never-delivers hazard).
- Shot queue bound 32 → 128 (worst case 15 at 5 Hz).
- Per-phase DEBUG timing: strict fire/frame-wait durations, free-run row
  duration.
- Pinned by `tests/test_read_path_staging.py`; full suite green.

**Hardware verification (1 Hz):** free-run NOSCAN row cadence must be
1.00 s (was exactly 2.00 s); strict cadence drops by the former read cost.
5 Hz is *designed for, untested* — the plan's obligations at 5 Hz are all
rate-derived, none hardcoded.

### Phase 2 — telemetry aggregation (margin, not critical path at 5 Hz)

Fold the ~87 per-device `CaTelemetryReadable`s into a handful of composite
readables whose `read()` gathers across devices (each device already
gathers internally — one more level of the same pattern).  Saves
~25 ms/row of Msg dispatch, gives one stage point and one place to bound
total telemetry wall time.  No schema impact (same columns).

### Phase 3 — t0-sync seed verification

The Phase 1 window cap makes 5 Hz *arithmetically* sound (window ∈
(50, 100) ms between the clock-skew floor and half a period), but with
little margin.  Add a cheap post-seed cross-check: verify the first armed
rows land at `shot_offset == 0` across sync devices before trusting the
scan (or re-seed from a plan-owned single shot).  Beyond 5 Hz the design's
two requirements (window > skew, window < period/2) collide — redesign
territory, out of scope.

### Phase 4 — timestamp-only telemetry shot-attribution (design note first)

Confirmed feasible with no MySQL flag: every gateway device serves
`acq_timestamp` (pinned at 0.0 forever unless triggered) + `systimestamp`,
so triggered-ness is *observable* (advancing in lockstep with the trigger
vs static).  Per-sample attribution = include each telemetry device's own
`acq_timestamp` in its variable set + seed a `ShotIdTracker` per device +
label rows like `FreeRunContributorSupport` minus the grace wait
(telemetry must never gate a shot).  One trap: deadband-suppressed data
columns carry last-*change* timestamps — attribution must key on the
timestamp PV, never data-column metadata.  Open semantic decisions for the
design note: labeling of async devices' rows, when the observation window
runs (needs trigger authority → pre-claim stage), which columns are added.

## Deferred / out of scope

- Telemetry as Bluesky monitor streams (`sd.monitors`) — the fully
  conventional shape (zero shot-path cost, device-native cadence), deferred
  because it is an event-schema change; revisit whenever the schema
  iterates anyway (also the natural moment to wire the schema-version check,
  issue #528).
- `TiledWriter` per-event normalizer cost (~9 ms at 800 columns) — only
  matters past 5 Hz.
- Strict-mode rate: bounded by the blocking DG645 fire; not a software
  problem.
- 100 Hz+ — ground-up framework reconsideration, per the owner.
