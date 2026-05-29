# Piece 1 — DataLogger streaming write

**Branch model:** independent feature branch off master
**Abort risk:** low
**Depends on:** nothing (can ship first)

---

## Current behavior

DataLogger is the single consumer of TCP events from all devices flagged
for the scan. The existing flow:

1. **Trigger.** Shot control fires a central trigger at the user-set
   `rep_rate` (currently capped at 1 Hz, range supported by this design
   is ≤ 1 Hz).
2. **Synchronous device events.** Each sync device's TCP event carries
   an `acq_timestamp` pinned to its observed trigger arrival. (A ~100
   ms spread of trigger arrival across devices is expected and known.)
3. **Per-device t0.** At scan setup, each device's `t0` is recorded.
   Elapsed time within the scan, for that device, is
   `elapsed_time = acq_timestamp - device.t0`.
4. **Shot resolution.** `shot = int(elapsed_time * rep_rate)`. At 1 Hz
   this is just the integer of `elapsed_time`.
5. **Bucket assembly.** DataLogger maintains an in-memory dict keyed by
   `elapsed_time`. The first device for a given `elapsed_time` creates
   the bucket; subsequent sync devices fill into it.
6. **Async devices.** When a new `elapsed_time` bucket is created,
   async devices have their state snapshot-polled and written into
   the bucket — so async-device values appear once per shot rather
   than at the device's faster native update rate.
7. **Flush.** At scan-end (or on a flush interval) the in-memory
   shot-keyed table is written to `ScanData_scan.txt` (the s-file).

This works and is correct. The motivation for change is **not** that
this is broken.

---

## Motivation (decouple, don't fix)

We want to **decouple shot-table assembly from the hot path** so that:

1. **Live readers can tail an append-only log.** Today a live-plot
   consumer would have to either reach into DataLogger's in-memory
   state or wait for scan-end. With an append-only TSV, any reader
   can tail the file without coordinating with the writer.
2. **Post-scan code does the reshape with whole-scan context.** A
   ScanAnalysis task (piece 4) reshapes the long-format TSV into the
   shot-keyed table with the full scan in hand — easier joins with
   per-device files (piece 3 helps here), TDMS columns, and any
   late-arriving data.
3. **DataLogger's hot path stays simple.** No shot-completion
   bookkeeping, no growing in-memory table, no decision to make about
   when a row is "done."

The shot-keyed s-file is still produced — just by piece 4 instead of
DataLogger. During the transition DataLogger keeps emitting it in
parallel so nothing downstream needs to change yet.

---

## Proposed solution

Replace the in-memory shot table with a **streaming long-format TSV
append**. Every TCP event becomes one row per scalar variable.

### Schema

```
elapsed_time	shot	bin	device	variable	value
1.0234	1	0	UC_CameraA	exposure	0.0123
1.0234	1	0	UC_CameraA	gain	2.0
1.0418	1	0	UC_BPM3	x_um	-12.4
1.0418	1	0	UC_BPM3	y_um	5.1
2.0156	2	0	UC_CameraA	exposure	0.0123
12.0234	12	1	UC_CameraA	exposure	0.0145
```

Properties:

- **One row per (event, scalar variable).**
- **Append-only**, line-buffered. Concurrent readers can tail.
- **`shot`** is derived from `int(elapsed_time * rep_rate)`. No
  shot-completion check; the bucketing is implicit in the column.
- **`bin`** is the current scan-step index, supplied by ScanManager
  and stamped onto every row that lands in a given bucket.
- **Scalars only.** Arrays/images stay in their per-device side files.
- **Async devices** are still snapshot-polled when DataLogger sees the
  first event for a new `elapsed_time` bucket; their snapshot values
  are emitted as rows alongside the sync-device rows for that bucket.

### Sketch of the new on_tcp_event

```python
ACQ_TIMESTAMP_KEY = "acq_timestamp"

def on_tcp_event(self, device_name: str, message: dict) -> None:
    acq_ts = message.get(ACQ_TIMESTAMP_KEY)
    if acq_ts is None:
        return  # only sync devices land here directly

    t0 = self.device_t0[device_name]
    elapsed_time = acq_ts - t0
    shot = int(elapsed_time * self.rep_rate)
    bin_ = self.scan_manager.current_bin

    # If this is the first event for this elapsed_time, poll async devices.
    if elapsed_time not in self._seen_buckets:
        self._seen_buckets.add(elapsed_time)
        async_rows = self._poll_async_devices(elapsed_time, shot, bin_)
    else:
        async_rows = []

    rows = list(async_rows)
    for var, value in message.items():
        if var == ACQ_TIMESTAMP_KEY or not _is_scalar(value):
            continue
        rows.append(
            f"{elapsed_time}\t{shot}\t{bin_}\t{device_name}\t{var}\t{value}\n"
        )
    if rows:
        with self._lock:
            self._fh.writelines(rows)
```

`self._seen_buckets` is a set of `elapsed_time` values already polled;
it stays small (one entry per shot, bounded by scan length). It can be
trimmed periodically if scan length is a concern.

### Transition strategy

DataLogger **keeps emitting the existing shot-keyed s-file** during
the transition. Both files get written; tools that read the s-file
continue to work; piece 4 starts reading the long-format TSV when
ready. Once piece 4 is solid and consumers have migrated, retire the
in-engine s-file emission in a follow-up PR.

---

## Files touched

- `GEECS-Scanner-GUI/geecs_scanner/engine/data_logger.py` — add
  `_open_long_tsv` and the new `on_tcp_event` body; keep the existing
  shot-keyed path intact during transition.
- `GEECS-Scanner-GUI/geecs_scanner/engine/scan_manager.py` — open the
  long-format TSV at scan start, close at scan end; expose
  `current_bin` if not already.
- `GEECS-Scanner-GUI/tests/engine/test_data_logger_streaming.py` — new
  tests: per-device t0 → elapsed_time → shot correctness, bin
  stamping, async snapshot on first event for a bucket, scalar-only
  filtering, missing acq_timestamp handling, schema header.
- `GEECS-Scanner-GUI/CHANGELOG.md` — minor bump entry.

---

## Open questions

1. **Write architecture.** Three options for the writer side:
   - **(a) Synchronous append.** `on_tcp_event` grabs the lock, writes,
     releases. Simplest. At ~250–500 rows/sec × ~50 bytes/row ≈ 25 KB/s
     local disk, the write is microseconds. Lock contention is a
     non-issue because DataLogger is already the single consumer of all
     device events.
   - **(b) Queue + dedicated writer thread.** `on_tcp_event` pushes
     rows to a thread-safe queue; a writer thread drains and appends.
     Doesn't block `on_tcp_event` under disk hiccups (network drive
     stalls, fsync pauses).
   - **(c) Per-device files.** Each device gets its own long-format
     TSV. No write contention. Downstream reshape concatenates. More
     files, more state.

   Recommend defaulting to (a); promote to (b) only if production
   shows hot-path stalls. (c) is overkill for the throughput we have.

2. **Schema versioning.** Header line on the TSV?
   `# scan_log_v1\telapsed_time\tshot\tbin\tdevice\tvariable\tvalue`.
   Cheap insurance. Recommend yes.

3. **Bucket trimming.** `self._seen_buckets` grows linearly with scan
   length. At 1 Hz an hour-long scan is 3600 entries — negligible. But
   if we ever care, a sliding-window trim on `elapsed_time - 60s` is
   trivial. Defer unless it surfaces.

4. **What if `acq_timestamp < device.t0` (negative elapsed_time)?**
   Treat as setup-noise event, don't write. Log a warning.

5. **Float precision in `elapsed_time` and `value`.** Long-format means
   stringification at write time. Use a fixed format like `f"{x:.4f}"`
   to keep the file deterministic. The typed reshape happens in piece 4.

---

## Sequencing

- Lands independently of pieces 2, 3, 4.
- Order is convenience, not correctness — pieces 1, 2, 3 can land in
  any order.

---

## Out of scope

- Removing the existing shot-keyed s-file (deferred to post-piece-4
  cleanup).
- Changing the TCP event format itself.
- Logging non-scalar payloads — arrays/images stay in per-device side
  files.
- Live plotting infrastructure (separate effort; this enables it,
  doesn't deliver it).
- Raising `rep_rate` above 1 Hz (deliberately deferred until the new
  pipeline is proven at 1 Hz).

---

## Abort risk

Low. The hot-path change is small and bounded; the old shot-keyed
s-file keeps being produced in parallel during transition. If the
long-format TSV turns out to be wrong, we revert this PR and the old
code path is unchanged.

---

## Branching strategy

`feature/datalogger-streaming-tsv` off master. Single PR, aim for
≤500 LOC including tests. Squash-merge.
