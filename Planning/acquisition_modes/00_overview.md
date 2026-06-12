# Acquisition Modes — Overview

Split Bluesky acquisition into two explicit modes with different acquisition
contracts but **one shared event schema**:

| Mode | Contract |
|---|---|
| `strict_shot_control` | The plan owns each shot. Shot controller (DG645) is put in single-shot mode; the plan fires each trigger, then awaits every sync device's `acq_timestamp` advance. Every row is complete by construction. A required device failing to respond is a hard failure. |
| `free_run_time_sync` | External trigger free-runs at the machine rep rate. One **reference device** (pacemaker) gates event creation: its `acq_timestamp` advance ⇒ emit a row. All other sync devices contribute their latest cached data, labeled with their own derived shot ID. Missing/slow devices never block row emission. |

## Locked design decisions

These were settled in design discussion (June 2026). Do not relitigate without
new information.

1. **Trigger confirmation = `acq_timestamp` advance.** This is the standard
   pattern (EPICS AD counter advance, ophyd-async `observe_value`), adapted to
   GEECS. Timestamps beat counters here: they survive device restarts, and
   GEECS timeout frames (TCP push without timestamp advance) are inherently
   ignored. No hardware pulse ID exists at BELLA; at 1 Hz with ~50 ms NTP sync
   the derived shot ID has 3 orders of magnitude of discrimination margin.

2. **Cross-device matching is by shot-ID equality, never by cross-device
   timestamp deltas.** Each device's shot ID is derived from *its own*
   `acq_timestamp` against *its own* t0, so clock skew between control
   machines cancels. The original `timestamp_tolerance_s` matching concept is
   dropped. The only windows that remain: the t0-sync acceptance window
   (~200 ms) and the optional grace wait (~one TCP push period).

3. **Shot ID is computed incrementally**, not as `(ts − t0) × rep_rate`
   absolute. Each device's ID advances by `round(Δt × rep_rate)` since its own
   previous event. Absolute computation accumulates rep-rate error
   (0.05% × 30 min ≈ a full shot); incremental resets the error every shot.
   See `02_shot_id_and_t0_sync.md`.

4. **Shot ID is "physical trigger-opportunity number", not a row counter.**
   Jumps > 1 across stage-move dead time are expected and harmless — matching
   is equality across devices, never consecutiveness. Named `shot_id` in the
   schema to avoid the `shotnumber` connotation. It is matching machinery and
   diagnostics, **not** a file-join key — files join to events by device
   `acq_timestamp` (existing decision, unchanged).

5. **Coordinated t0 capture is a plan stage** at the start of a free-run run:
   with the trigger disarmed, read every sync device's last `acq_timestamp`;
   if all are within the acceptance window (~200 ms, machines NTP-synced) they
   came from the same physical trigger and become per-device t0s. Replaces
   today's lazy per-device first-read capture (which can land on different
   physical shots, making IDs incomparable).

6. **One event schema for both modes** (`geecs_event_schema: 1` in run
   metadata). Per-device companion columns (`shot_id`, `shot_offset`, `valid`)
   are emitted in *both* modes — trivially `offset=0, valid=True` in strict
   mode — so consumers never branch on acquisition mode to read data. See
   `01_event_schema_contract.md`.

7. **Free-run rows are emitted immediately at reference acceptance** with
   per-device shot offsets, rather than grace-waiting for every device to
   reach offset 0. A long-exposure camera (e.g. 900 ms at 1 Hz) lands at
   offset −1 with no data loss; realignment downstream is a per-device shift
   because every cell self-labels. A short grace wait (~one TCP push period)
   is optional polish to raise the offset-0 fraction. An end-of-scan **flush
   read** captures lagging devices' final shot.

8. **Reference device = first synchronous device in the save list, picked
   automatically.** No YAML field; the scanner classifies the first sync
   device as the Triggerable pacemaker and the rest as timestamped
   contributors, recording the choice in run metadata (`reference_device`).
   Nothing in the design depends on the reference being special — it is
   only the row gate, needed because Bluesky events are immutable once
   saved (the legacy "first device to report creates the row, others
   back-fill" scheme requires mutable rows). Known consequence: a shot the
   reference misses produces no row (recoverable via offsets/flush). If a
   flaky reference ever bites, the upgrade path is an any-of gate (row when
   *any* sync device's shot ID advances) — same schema, deferred until
   needed. The DG645 does not emit TCP events per output trigger; adding
   that to the LabVIEW driver is a nice-to-have, not blocking.

9. **Per-shot plan stubs are the unit of reuse.** The contract is enforced by
   device classes + stubs, not by BlueskyScanner or the GUI. Custom notebook
   workflows compose the same stubs and inherit the schema, scan numbering,
   and save-path discipline automatically (via the run wrapper,
   `05_run_wrapper_and_plumbing.md`).

10. **Bluesky `scan_id` is set to the claimed GEECS day-scoped scan number**
    via per-run `md` (overrides the RE counter). `scan_id` has no uniqueness
    contract in Bluesky — daily reset to 1 is fine; `uid` is the real key.
    Analysis must never look up runs by `scan_id` alone (qualify with day, or
    use `scan_number` + start-doc `time`).

11. **No backwards compatibility required.** Sole user. Field renames and
    behavior changes land directly; no legacy third behavior is preserved.
    Schema-version bumps do **not** require a new Tiled catalog — every run is
    self-describing (own descriptors + metadata); versions coexist in one
    catalog. Additive changes don't bump the version; only breaking ones do.

## Commit sequence

1. `docs(geecs-bluesky): plan acquisition modes split` — this directory
2. `feat(geecs-bluesky): incremental shot-id tracking and coordinated t0 sync`
3. `feat(geecs-bluesky): event schema contract v1` — contract doc + uniform
   companion columns from GeecsGenericDetector
4. `feat(geecs-bluesky): free-run timestamped readable` — non-triggerable sync
   contributor with offset/valid + grace wait
5. `feat(geecs-bluesky): free-run time-sync step scan` — t0-sync stage,
   pacemaker loop, tail flush
6. `feat(geecs-bluesky): strict single-shot plan stub` — SINGLE_SHOT shot
   control state, fire-between-trigger-and-wait stub
7. `feat(geecs-bluesky): geecs run wrapper` — extract scan-number claim +
   save-path + metadata injection into reusable preprocessor
8. `feat: plumb acquisition_mode config` — scan options / scanner dispatch
   (may touch geecs-data-utils; acceptable)
9. Tests accompany each commit (FakeGeecsServer; see mode test matrices in
   `03`/`04`).

## Deferred (deliberately)

- **Tiled → clean-DataFrame / s-file exporter** — only after the free-run
  event shape has survived real use.
- **GUI exposure of acquisition mode** — YAML/env toggle first.
- **Persistent t0 registry / central subscription daemon** ("event-builder
  service") — per-scan t0 sync is self-healing against rep-rate changes and
  device restarts; revisit when per-scan build/teardown actually hurts.
- **DG645 per-trigger TCP events** (LabVIEW driver change) — would make the
  shot controller the ideal pacemaker.
- **Multi-stream flyer/collect acquisition** (Bluesky-native offline event
  building) — keep `shot_id` as the universal join key so a future migration
  is mechanical.
