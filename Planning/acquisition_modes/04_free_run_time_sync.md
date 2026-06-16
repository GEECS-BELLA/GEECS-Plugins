# Mode 2 — free_run_time_sync

External trigger free-runs at the machine rep rate; the plan never blocks on
any device except the **reference** (pacemaker).

## Plan shape

Mostly today's `geecs_step_scan`: the SCAN/STANDBY arm/disarm bracketing,
step loop, ScanContext, and finalize wrapper all carry over. What changes:

1. **t0-sync stage** at run start, before the first arm (see `02`).
2. **Device classification** does the real work — `bps.trigger_and_read` only
   triggers Triggerable devices, so with exactly one Triggerable in the list
   the free-run contract falls out of construction:
   - reference → `GeecsGenericDetector` (Triggerable; its awaited
     `acq_timestamp` advance *is* "a shot happened, emit a row")
   - other sync devices → `GeecsTimestampedReadable` (new; **no** `trigger()`)
   - async devices → `GeecsSnapshotReadable` (unchanged)
3. **Tail flush**: after the last shot, one extra read of all contributor
   devices emitted as a final event, so offset −1 devices' last shot isn't
   lost.
4. Metadata: `acquisition_mode="free_run_time_sync"`, `reference_device`,
   `device_t0s`, `t0_sync_window_s`.

## `GeecsTimestampedReadable`

Non-triggerable readable for optional sync contributors. Like
`GeecsSnapshotReadable` plus the sync-device companion columns. At `read()`:

1. Take own cached `acq_timestamp`; `shot_id = tracker.update(ts)`.
2. `shot_offset = shot_id − row_shot_id` where the row's shot ID is the
   reference's (the readable holds a reference to the pacemaker device /
   its tracker; reads happen only after the reference trigger completed, so
   the reference value is fresh).
3. Optional **grace wait**: if `shot_offset < 0`, wait up to ~one TCP push
   period (~0.2–0.3 s) for own cache to advance, then re-derive. Bounded;
   never blocks row pacing meaningfully at 1 Hz.
4. Emit data variables (real values) + `acq_timestamp`, `shot_id`,
   `shot_offset`, `valid = (shot_offset == 0)`.

No NaN-ing of *late* data: an offset −1 cell carries shot N−1's real values,
truthfully labeled — downstream realignment is a per-device shift keyed on
`shot_id`. NaN appears only when a device has nothing derivable (unseeded /
never connected), per the stable-keys rule.

## Why arrival latency forces this design (recorded rationale)

`acq_timestamp` is back-dated to acquisition start, so shot IDs agree across
devices regardless of exposure. But the TCP frame carrying it arrives
exposure + readout + push later. A 900 ms exposure camera's frame for shot N
arrives as shot N+1 fires — a row emitted at reference-acceptance time
*cannot* contain that camera's shot-N data. Hence: emit immediately, label
truthfully with offsets, realign downstream. Grace-waiting every device to
offset 0 would drag every row by the slowest exposure and skip shots
(the reference trigger drains stale queue frames on each call).

Known cosmetic effect: a device whose arrival delay straddles the row-emission
boundary flickers between offset 0/−1 row to row. Correctness is unaffected
(cells self-label); only the valid-fraction statistic looks noisy.

## Tests (FakeGeecsServer)

- Reference emits 4 shots → 4 rows regardless of other devices.
- Contributor keeps pace → offset 0, valid, real values on all rows.
- Contributor misses shot 2 (no fire on that device) → row 2 emitted with
  that device's `shot_id` stale, `valid=False`; rows 1/3/4 valid.
- Contributor with delayed TCP push (arrival after reference acceptance,
  within grace) → offset recovers to 0 via grace wait.
- Contributor with arrival delay ≫ grace (long exposure) → offset −1 rows,
  real values, correct `shot_id` labels; tail flush captures the final shot.
- Snapshot device values appear on every row.
- Stage-move dead time → all devices' `shot_id` jump together; equality
  matching unaffected.
