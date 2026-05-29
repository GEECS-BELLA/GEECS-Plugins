# Piece 4 — Async scan finalization in ScanAnalysis

**Branch model:** long-running branch `refactor/scan-finalization`
**Abort risk:** medium (the integration point — most plausibly forces
us to back out)
**Depends on:** piece 1 (long-format TSV), benefits from 2 and 3

---

## Problem

After a scan finishes, the per-scan folder contains:

- A shot-keyed s-file (`ScanData_scan.txt`) produced by DataLogger
  during the scan.
- A scan-level TDMS file.
- Per-device output files (images, etc.).
- `scan_info.ini`.

The s-file is produced synchronously by the scan engine. That's a
runtime burden the engine doesn't need: it has to know, at write time,
what columns will exist, what shot a value belongs to, and how to
reconcile late-arriving events. Piece 1 retires the in-memory shot-keyed
representation in favor of a long-format TSV stream.

But the rest of the system still expects a shot-keyed s-file. Today
that file is produced by the engine; piece 1's long-format TSV is a
**supplement**, not a replacement.

---

## Proposed solution

A new ScanAnalysis task — `FinalizeScan` — that runs after a scan
finishes and is responsible for producing the analysis-ready
shot-keyed s-file from the inputs the engine produced:

1. **Long-format TSV** (piece 1) — scalar event stream.
2. **Per-device file index** — list of files in the per-scan folder
   and their associations to shots (using piece 3's matching where
   needed).
3. **TDMS file** — scan-level channel data.
4. **`scan_info.ini`** — scan metadata.

The task produces:

1. **`ScanData_scan.txt`** — the canonical shot-keyed s-file (same
   format the engine produces today, so downstream is unchanged).
2. **`ScanData_scan.parquet`** — optional Parquet copy for fast
   downstream loading.
3. **`scan_finalization.log`** — what was matched, what was missing,
   any anomalies.
4. **A status entry in `analysis_status/`** so the ScanAnalysis queue
   knows finalization is done before any other tasks run.

### Sketch of the task

```python
class FinalizeScan(ScanAnalysisTask):
    """Reshape the long-format TSV into a shot-keyed s-file."""

    requires = []          # First task in the queue.
    produces = ["ScanData_scan.txt", "ScanData_scan.parquet"]

    def run(self, scan_path: ScanPaths) -> None:
        long_df = self._load_long_format(scan_path)
        shot_table = self._pivot_to_shot_table(long_df)
        device_files = self._build_device_file_index(scan_path)
        shot_table = self._merge_file_paths(shot_table, device_files)
        tdms = self._maybe_load_tdms(scan_path)
        if tdms is not None:
            shot_table = self._merge_tdms_columns(shot_table, tdms)
        self._write_s_file(shot_table, scan_path)
        self._write_parquet(shot_table, scan_path)
        self._write_log(scan_path)
```

### Producer/consumer contract change

This is the real change in posture:

- **Before:** engine produces `ScanData_scan.txt` synchronously.
  Downstream tools read it immediately on scan completion.
- **After:** engine produces only the long-format TSV + per-device
  files. `FinalizeScan` produces the s-file asynchronously, as a
  ScanAnalysis task. Downstream tools read it once finalization marks
  the scan as ready.

Downstream consumers (ScanAnalysis analyzers, log upload, the GUI's
post-scan summary) need a way to wait for finalization. Two options:
- **Polling:** they look for the s-file or a status marker in
  `analysis_status/`.
- **Event:** ScanAnalysis emits a "scan finalized" signal that
  consumers subscribe to.

Recommend starting with polling — the ScanAnalysis task queue already
has a status-marker convention; reuse it.

---

## Why this lives on a long-running branch

Pieces 1–3 are individually shippable and individually valuable. Piece
4 is where the contract between producer (scan engine) and consumer
(downstream analysis) actually changes. The set of things that have
to all line up for piece 4 to be net-positive:

- DataLogger reliably produces a long-format TSV that contains every
  scalar event (piece 1, validated in production).
- `FinalizeScan` correctly reshapes that TSV into the same s-file
  shape ScanAnalysis/GUI expect.
- Per-device file matching covers the cases today's matching covers
  (piece 3 enables the unhappy paths).
- The GUI's post-scan flow tolerates the s-file appearing
  asynchronously rather than at scan-end.
- Operator workflows (log upload, ad-hoc inspection scripts) tolerate
  the same.

Any of those failing is reason to delay or abort. A long-running
branch keeps the work visible and reviewable without committing
master to a half-finished transition.

The branch model: `refactor/scan-finalization` is created off master
after pieces 1–3 are in. It rebases periodically. When it's ready, it
merges via a single PR. If it never becomes ready, it gets deleted —
master is unchanged.

---

## Files touched

- `ScanAnalysis/scan_analysis/tasks/finalize_scan.py` — new module.
- `ScanAnalysis/scan_analysis/task_queue.py` (or equivalent) — ensure
  `FinalizeScan` runs first.
- `ScanAnalysis/tests/tasks/test_finalize_scan.py` — new tests using
  fixture scans (a small bundled TSV + per-device files).
- `GEECS-Data-Utils/geecs_data_utils/scan_data.py` — possibly a
  `wait_for_finalization` helper; possibly nothing if polling is left
  to consumers.
- `GEECS-Scanner-GUI/geecs_scanner/app/...` — post-scan summary needs
  to handle "scan done, finalization pending" state.
- `ScanAnalysis/CHANGELOG.md` — minor bump entry on merge.

---

## Open questions

1. **What's the exact pivot from long-format to shot-keyed?**
   - Conflict resolution when a (shot, device, variable) triple
     appears more than once (extra event in the window): last write
     wins, mean, or warn? Recommend "last write" + log line.
   - Missing values: leave as NaN (current behavior in s-file) or
     forward-fill?
   - Column ordering: device-grouped or insertion-ordered? Existing
     s-file is insertion-ordered; preserve.
2. **Should `FinalizeScan` also re-produce the long-format TSV in a
   cleaned form?** E.g. quantized shot column, sorted by
   `elapsed_time`. Maybe — but defer; first cut keeps the raw TSV
   untouched.
3. **What if TDMS is missing or partial?** Today the engine writes
   TDMS during the scan. Finalization should tolerate absence (early
   abort), warn, and continue.
4. **Parquet emission — yes/no?** Recommended yes. Cheap; downstream
   speedup is real.
5. **GUI behavior on the "finalization pending" state.** Spinner? Or
   does the GUI just keep using the engine's parallel s-file until we
   retire it? Decide before flipping the consumer.
6. **Retire the engine's parallel s-file when?** Only after this is
   solid in production for a meaningful number of scans. Don't bake
   that retirement into this PR.

---

## Sequencing

1. Piece 1 must be in master and validated.
2. Piece 3 should be in master (the fallback matcher gets used in
   finalization).
3. Piece 2 is independent but nice-to-have first.
4. Branch `refactor/scan-finalization` off current master.
5. Develop `FinalizeScan`. Run shadow comparisons: produce the new
   s-file alongside the engine's, diff them across a corpus of past
   scans.
6. When diffs are zero (or all explainable), flip the consumer.
7. Retire the engine's parallel s-file in a separate follow-up PR
   *after* `refactor/scan-finalization` is merged.

---

## Out of scope

- Changing what scans produce on disk other than the s-file (TDMS,
  per-device files unchanged).
- Real-time finalization during the scan. The whole point is
  post-scan async.
- Replacing the TSV with a different streaming format. The TSV is
  fine; convert to Parquet only as a derived output.
- Adding new analyzers — that's normal ScanAnalysis work, not part
  of finalization.

---

## Abort risk

Medium. The places it can fail:

- Long-format TSV doesn't have enough fidelity to rebuild the s-file
  exactly. Mitigation: shadow comparison before flipping consumer.
- Downstream tools have undocumented assumptions about *when* the
  s-file appears. Mitigation: the polling helper + status marker, and
  a transition period where the engine's parallel s-file is still
  produced.
- Operational workflows that read scans by hand expect the s-file
  the moment the scan ends. Mitigation: same as above; document the
  change in the package CHANGELOGs and the operator docs.

If any of these don't pan out, the branch is deleted, master is
unchanged, and pieces 1–3's value is fully retained on master.

---

## Branching strategy

`refactor/scan-finalization` off master. Long-running. Rebase
periodically. When the shadow comparison is clean across enough real
scans, merge via a single squashed PR.

If the integration story falls apart, the branch is deleted. Pieces
1–3 stay shipped on master regardless.
