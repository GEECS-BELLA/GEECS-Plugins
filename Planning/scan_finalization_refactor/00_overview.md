# Scan Finalization Refactor — Overview

**Status:** planning
**Branch model:** mix of independent feature branches + one long-running branch
**Started:** 2026-05-15

This directory captures the umbrella plan for a multi-piece refactor of how
GEECS scans are logged, written to disk, and finalized into analyzable form.
Each numbered doc covers one piece.

---

## Thesis: "runtime minimum, post-scan async"

The scan engine currently does a lot of synchronous work during the scan
itself — building a complete shot-keyed row out of incoming TCP events,
deciding when a row is "done," moving files into the per-scan directory,
and (in some configurations) waiting on slow per-device save operations.

This forces a fragile invariant: the engine has to know, at runtime, which
device events belong to which shot, and it has to make blocking decisions
that can stall the next shot trigger.

The refactor inverts that posture:

1. **At runtime:** write everything you observe, with a timestamp, in the
   simplest possible append-only form. Don't decide what's "complete."
   Don't make the scan wait for slow IO.
2. **After the scan:** a separate process — running in the
   ScanAnalysis task queue — has the luxury of seeing the whole scan at
   once, quantizing events into shots, reshaping into the final
   shot-keyed table, and producing the artifacts downstream code expects.

The four pieces below each implement part of this thesis. They can be
shipped independently in roughly the order presented, but only piece 4
fully commits to the new model.

---

## The four pieces

### Piece 1 — DataLogger streaming write (`01_datalogger_streaming.md`)
Rewrite `DataLogger.on_tcp_event` so each incoming TCP event is appended
as a single long-format row to a TSV (`elapsed_time`, `shot`, `device`,
`variable`, `value`) instead of being held in memory and reassembled into
a shot-keyed dict. Strip the "wait until row complete" logic.

### Piece 2 — Per-device save mode (`02_per_device_save_mode.md`)
Let each device opt into a `save_mode_override` (e.g. `direct`, `local`,
`skip`) on a per-scan basis. Currently the engine uses a single global
mode that forces a one-size-fits-all tradeoff between speed and
per-device file isolation. Configurable per-device unblocks faster
acquisition for camera-light scans.

### Piece 3 — Timestamp filenames + fallback (`03_timestamp_naming_and_matching.md`)
Add a timestamp-based naming option for per-device output files, and a
matching fallback in `geecs-data-utils` ScanData/ScanPaths so the
shot-number ↔ file association can be rebuilt by timestamp if the shot
number isn't in the filename. Strictly additive to ScanData; safe to land
ahead of pieces 1 and 4.

### Piece 4 — Async finalization in ScanAnalysis (`04_scan_finalization_in_scananalysis.md`)
The "after the scan" half of the thesis. A ScanAnalysis task that runs
after a scan finishes and produces the canonical shot-keyed s-file from
the long-format TSV (piece 1), the per-device file index, and the scan's
TDMS — making the synchronous engine path no longer responsible for
producing the analysis-ready artifact.

---

## Sequencing & branching

| Piece | Branch model | Why |
|---|---|---|
| 1 | Independent feature branch off master | Self-contained change to DataLogger. Reverting is `git revert`. |
| 2 | Independent feature branch off master | Adds a config option; default behaviour unchanged. |
| 3 | Independent feature branch off master | Additive to ScanData/ScanPaths. No callers forced to opt in. |
| 4 | Long-running `refactor/scan-finalization` branch | The integration point. Higher abort risk — wants the safety net of a branch we can leave open or discard. |

**Why piece 4 lives on a long branch.** Piece 4 is where the runtime
posture actually changes — DataLogger stops producing the s-file
directly, and downstream consumers (ScanAnalysis tasks, the GUI, log
upload) start reading from finalization output. That's the spot where
we'd most plausibly discover the model is wrong and want to abort. A
long-running branch keeps that work visible without committing master
to it until we're sure.

Pieces 1, 2, and 3 stand on their own value even if piece 4 is never
merged:
- Piece 1 simplifies DataLogger and removes shot-completion guesswork.
- Piece 2 unblocks faster scans today.
- Piece 3 makes ScanData more forgiving of filename schemes we already
  encounter in the wild.

---

## Abort risk per piece

- **Piece 1:** low. Long-format TSV is a superset of the current
  shot-keyed s-file; we can keep emitting the old s-file in parallel
  during transition.
- **Piece 2:** low. Pure addition with a default that preserves current
  behaviour.
- **Piece 3:** low. Strictly additive matching path; old filenames keep
  working.
- **Piece 4:** medium. Real change in producer/consumer contract.
  Mitigated by the long-running branch and by piece 1's parallel s-file
  emission while we validate.

---

## Out of scope

- GEECS-PythonAPI restructuring (separate effort, do not touch).
- Replacing TDMS as a storage format.
- Changing the GUI's scan submission flow.
- Bluesky integration — orthogonal track; this refactor should make
  Bluesky easier later but does not depend on it.

---

## Open questions

- Do we want piece 4 to also emit a Parquet copy of the s-file? Leaning
  yes, but defer to piece 4's own doc.
- Does the long-format TSV need a schema version field? Probably yes;
  decided in piece 1's doc.
- What happens to the existing `data_logger.py` tests when we change the
  shot-completion path? See piece 1.

---

## How to read the rest of this dir

Each numbered doc follows the same shape:

1. **Problem** — what's broken or limiting today
2. **Proposed solution** — the change in concrete terms
3. **Files touched** — best current estimate
4. **Open questions** — things to decide before / during the PR
5. **Sequencing** — what has to land first
6. **Out of scope** — what we're explicitly not doing
7. **Abort risk** — what could force us to back out
8. **Branching strategy** — independent feature branch vs long-running
