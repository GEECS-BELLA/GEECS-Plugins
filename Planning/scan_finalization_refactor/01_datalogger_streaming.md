# Piece 1 — DataLogger streaming write

**Branch model:** independent feature branch off master
**Abort risk:** low
**Depends on:** nothing (can ship first)

---

## Problem

`DataLogger.on_tcp_event` currently:

1. Receives a TCP event from a device (a dict of variable→value).
2. Tries to associate the event with a shot number based on the engine's
   current notion of "where are we in the scan."
3. Buffers values into a partially-filled shot row.
4. Decides — somehow — when the shot row is "complete," at which point
   it gets flushed to the in-memory shot table and eventually written
   to the s-file.

This model has several problems:

- **No central trigger.** Devices report independently over TCP; the
  engine does not see a single "shot N began" event. Shot membership is
  inferred from timing and from which variables have arrived, which is
  brittle.
- **Lost events.** If a device emits an extra event late, or two events
  for the same variable within a shot window, the existing logic
  silently overwrites or drops.
- **Synchronous IO under contention.** Writing the shot-keyed table
  back to the s-file happens at scan-end or on a flush interval;
  meanwhile the in-memory representation grows unboundedly for long
  scans.
- **Untestable behavior.** Several internal paths in `data_logger.py`
  and `device_manager.py` are not behaviorally tested (documented debt
  in root `CLAUDE.md`). The shot-completion heuristic is one of those
  paths.

---

## Proposed solution

Replace the shot-keyed in-memory model with a **streaming long-format
TSV append**. Every TCP event becomes one or more rows:

```
elapsed_time	shot	device	variable	value
1.0234	1	UC_CameraA	exposure	0.0123
1.0234	1	UC_CameraA	gain	2.0
1.0418	1	UC_BPM3	x_um	-12.4
1.0418	1	UC_BPM3	y_um	5.1
2.0156	2	UC_CameraA	exposure	0.0123
```

Key properties:

- **One row per (event, variable).** No grouping at write time.
- **Append-only.** Line-buffered writes; concurrent readers can tail the
  file for live plotting.
- **No shot-completion logic at runtime.** `shot` is derived trivially
  from the integer part of `elapsed_time` (1 Hz quantization) — or, in
  later iterations, by an explicit quantization function.
- **Scalar values only.** Arrays/images stay in their per-device files;
  this TSV is for scalars.

### Sketch of the new on_tcp_event

```python
ACQ_TIMESTAMP_KEY = "acq_timestamp"

def on_tcp_event(self, device_name: str, message: dict) -> None:
    elapsed_time = message.get(ACQ_TIMESTAMP_KEY)
    if elapsed_time is None:
        return
    shot = int(elapsed_time)  # 1 Hz quantization
    rows = []
    for var, value in message.items():
        if var == ACQ_TIMESTAMP_KEY or not _is_scalar(value):
            continue
        rows.append(f"{elapsed_time}\t{shot}\t{device_name}\t{var}\t{value}\n")
    if rows:
        with self._lock:
            self._fh.writelines(rows)
```

That's the whole hot path. No shot-completion check. No buffer of
partially-filled rows. The reshape to shot-keyed table is piece 4's job.

### Transition strategy

To keep the door open for an abort, **also keep emitting the existing
shot-keyed s-file** during the transition. Both files get written; tools
that read the s-file continue to work; piece 4 starts reading from the
long-format TSV when ready.

After piece 4 is solid and downstream consumers have moved, retire the
old code path in a follow-up PR.

---

## Files touched

- `GEECS-Scanner-GUI/geecs_scanner/engine/data_logger.py` — replace
  `on_tcp_event` body, add `_open_long_tsv`, retire shot-completion
  helpers.
- `GEECS-Scanner-GUI/geecs_scanner/engine/scan_manager.py` — open the
  long-format TSV at scan start, close at scan end.
- `GEECS-Scanner-GUI/tests/engine/test_data_logger_streaming.py` — new
  tests for: append correctness, line-buffering under simulated
  concurrent events, scalar/non-scalar filtering, missing timestamp
  handling.
- `GEECS-Scanner-GUI/CHANGELOG.md` — minor bump entry.

---

## Open questions

1. **Schema version.** Add a header line like
   `# scan_log_v1\telapsed_time\tshot\tdevice\tvariable\tvalue`?
   Or rely on the file extension and column order? Header is cheap
   insurance — recommend yes.
2. **Float vs string for `value`.** Long-format means everything is
   stringified at write time. Piece 4 does the typed reshape. Fine.
3. **What is `elapsed_time` zero?** Currently inconsistent. Recommend
   "first event received by DataLogger after scan-start," and document
   in the docstring.
4. **Concurrent read durability.** TSV append on Linux/Mac with
   line-buffered writes supports concurrent tail-read; on Windows
   network drives it's iffy. Confirm with a test before committing to
   live-plot use cases. (Live plotting is not blocking for piece 1.)
5. **What if a device emits zero scalars in an event?** Don't write a
   row. Currently the engine logs a warning; keep that.

---

## Sequencing

- Lands independently of pieces 2, 3, 4.
- Recommend landing **after** piece 3 if both are ready, only because
  the timestamp-based fallback in geecs-data-utils makes it easier to
  validate that the streaming TSV is producing valid shot membership.
  But strictly independent — order is convenience, not correctness.

---

## Out of scope

- Removing the existing shot-keyed s-file (deferred to post-piece-4
  cleanup).
- Changing the TCP event format itself.
- Logging non-scalar payloads in this file — arrays/images stay in
  per-device files.
- Live plotting infra (separate effort; this enables it, doesn't
  deliver it).

---

## Abort risk

Low. The change is bounded to DataLogger's hot path; the old s-file
keeps being produced during transition. If the long-format TSV turns
out to be wrong, we revert the DataLogger PR and the old code path is
untouched.

The main thing that could force an abort is if 1 Hz quantization turns
out to lose events (multiple events from the same device within one
second). Mitigation: validate against existing scan data before
committing to it; the quantization function is one line and trivially
replaceable.

---

## Branching strategy

`feature/datalogger-streaming-tsv` off master. Single PR. Aim for ≤500
LOC including tests. Squash-merge.
