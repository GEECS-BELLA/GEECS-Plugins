# Capture container format — `geecs-capture/1`

The capture daemon writes **one frame-stack file per device per scan** into
the engine-created `scans/ScanNNN/<device>/` directory. The *contract* is
this document plus the provenance attributes stamped into every file — the
container technology is an implementation detail behind the
`FrameStackWriter` protocol (`writer.py`), deliberately swappable (e.g. a
future Zarr writer) without touching the daemon. Readers must dispatch on
the `schema` attribute, never on the file extension alone.

## v0 container: HDF5 (`<device>.h5`)

| Path | Content |
|---|---|
| `/frames` | `(N, H, W)` image stack, native dtype, chunked `(1, H, W)`, appended per frame; shuffle + gzip-1 (built-in filters — readable by any HDF5 library; the file self-describes its filters, so readers need no knowledge of this choice and the schema stays `geecs-capture/1`) |
| `/acq_timestamp` | `(N,)` float64 — the device acquisition timestamp (Unix s; PVA timestamp = LabVIEW time − 2082844800), the universal join key |
| `/recv_timestamp` | `(N,)` float64 — daemon receive time (Unix s), diagnostic only |

Root attributes: `schema` (`"geecs-capture/1"`), `device`, `experiment`,
`scan_number`, `source_pv`, `created` (Unix s), and — written at finalize —
the per-device reconciliation counters `frames_received`, `frames_written`,
`duplicates_dropped`, `stale_skipped`, `shape_errors`, `queue_drops`,
`late_frames`, `writer_create_failures`, `append_failures`,
`disconnect_events`, plus
`finalized` (bool; absent means the daemon died mid-scan, a wedged writer
prevented safe finalization, or the file belongs to a session closed by a
mismatched/missing stop document — the tail of `/frames` is still valid up
to `N`).

**The counter identity** (every received frame lands in exactly one bucket):

```
frames_received == frames_written + duplicates_dropped + stale_skipped
                 + shape_errors + queue_drops + late_frames
                 + writer_create_failures + append_failures
```

`append_failures` counts frames the writer accepted but could not append
(an HDF5/NAS write error) — the file's `/frames` tail stays valid; the
frame is lost from the stack but never from the books.

`late_frames` counts frames delivered after session close (p4p can hand a
few in-flight events to the callback after unsubscribe) or left unwritten
behind a wedged writer. `disconnect_events` counts **real** connection
losses only — the initial not-yet-connected event p4p delivers on every
healthy subscription is absorbed, so a clean scan reads 0.

Writers are created **lazily on the first accepted frame**: a device
that accepts no frames produces **no file at all** — readers must treat
an absent `<device>.h5` as "not captured", not as an error. (Since
0.66.0, `geecs_run_wrapper` creates every capture-listed device dir
**pre-start-doc**, dual-write and toggle-off alike; lazy creation is kept
as defense in depth against a dir that still fails to appear.) `/frames`
is created with the first successful append — a file whose very first
append failed carries `/acq_timestamp` but no `/frames`; readers must
treat that as valid-but-empty. `scan_number` is stamped only when the
start doc carried an integer scan number.

## Semantics

- **Dedupe by `acq_timestamp`**: the device re-pushes its last frame at
  1 Hz with an unchanged timestamp when idle (measured 2026-08-27), and the
  gateway re-posts it. Dedupe is on *accepted* (not written) timestamps:
  an append-failed frame's timestamp stays claimed, so its idle re-push
  counts as a duplicate rather than retrying (a partial append may have
  landed rows); a writer-*creation* failure releases the timestamp so
  re-pushes retry. Note the PVA gateway's receive-time fallback
  (0.4.4: an implausible device timestamp is replaced with gateway
  `time.time()`): a camera with a broken timestamp ladder gets a fresh
  timestamp per re-push, so its idle frames are never deduped — the
  signature is a stack with ~1 Hz receive-time stamps and a failing
  s-file join for that camera; attributable, never silent.
- **Stale-window filter**: frames stamped earlier than the run's start-doc
  `time` minus a small margin (2 s) are the gateway's cached pre-scan frame
  — skipped and counted, never written. The gateway's `(1,1)` placeholder
  initial post carries timestamp 0.0 and is always filtered. Two documented
  caveats: (a) if another PVA client (e.g. a live viewer) holds the
  gateway's gate open between scans, the pre-scan cache refreshes at 1 Hz
  and a <2 s-old cached frame can pass the filter — it then appears in the
  stack but not in the LV files, showing as a +1 in the dual-write diff;
  (b) a camera-server clock lagging the worker by >2 s would stale-drop
  real first frames — visible as `stale_skipped` > the expected 0–1.
- **Append-per-frame with flush** (trailing flush): a crash loses at most
  the un-flushed tail, never the scan.
- The daemon **never creates directories**: `geecs_run_wrapper` creates
  every capture-listed `scans/ScanNNN/<device>/` dir pre-start-doc
  (dual-write and toggle-off alike; the save-enable plan covers the
  legacy non-capture path); a missing directory means the device is
  skipped loudly (cross-package invariant — analysis/services never
  create scan folders).
- **Toggle-off actively commands `save="off"`** (GeecsBluesky 0.67.0):
  captured cameras are built `save_control_only` — only the `save` control
  child exists (no `localsavingpath`, no save-path column, no asset docs)
  — and the run wrapper writes `off` eagerly at scan start, so a flag left
  on out-of-band can never keep writing native files to a stale path.
- **Capture ownership is synchronous-role only** (#702): an asynchronous
  (snapshot-role) camera of a capture devicetype is dropped from
  `capture_devices` by the engine with a warning — the snapshot role has
  neither the save-control surface nor an `acq_timestamp` join column —
  so the daemon never targets it and its native saving is left as-is.
