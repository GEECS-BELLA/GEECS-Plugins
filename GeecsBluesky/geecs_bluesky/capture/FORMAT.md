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
| `/frames` | `(N, H, W)` image stack, native dtype, chunked `(1, H, W)`, appended per frame |
| `/acq_timestamp` | `(N,)` float64 — the device acquisition timestamp (Unix s; PVA timestamp = LabVIEW time − 2082844800), the universal join key |
| `/recv_timestamp` | `(N,)` float64 — daemon receive time (Unix s), diagnostic only |

Root attributes: `schema` (`"geecs-capture/1"`), `device`, `experiment`,
`scan_number`, `source_pv`, `created` (Unix s), and — written at finalize —
the per-device reconciliation counters `frames_received`, `frames_written`,
`duplicates_dropped`, `stale_skipped`, `shape_errors`, `queue_drops`,
`late_frames`, `writer_create_failures`, `disconnect_events`, plus
`finalized` (bool; absent means the daemon died mid-scan, a wedged writer
prevented safe finalization, or the file belongs to a session closed by a
mismatched/missing stop document — the tail of `/frames` is still valid up
to `N`).

**The counter identity** (every received frame lands in exactly one bucket):

```
frames_received == frames_written + duplicates_dropped + stale_skipped
                 + shape_errors + queue_drops + late_frames
                 + writer_create_failures
```

`late_frames` counts frames delivered after session close (p4p can hand a
few in-flight events to the callback after unsubscribe) or left unwritten
behind a wedged writer. `disconnect_events` counts **real** connection
losses only — the initial not-yet-connected event p4p delivers on every
healthy subscription is absorbed, so a clean scan reads 0.

Writers are created **lazily on the first accepted frame** (the engine
creates `scans/ScanNNN/<device>/` after the start document, in the
save-enable plan): a device that accepts no frames produces **no file at
all** — readers must treat an absent `<device>.h5` as "not captured", not
as an error. A file always contains `/frames` (created with the first
append).

## Semantics

- **Dedupe by `acq_timestamp`**: the device re-pushes its last frame at
  1 Hz with an unchanged timestamp when idle (measured 2026-08-27), and the
  gateway re-posts it. A frame whose timestamp was already written is
  dropped and counted.
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
- The daemon **never creates directories**: the engine's save-enable plan
  owns `scans/ScanNNN/<device>/` creation; a missing directory means the
  device is skipped loudly (cross-package invariant — analysis/services
  never create scan folders).
