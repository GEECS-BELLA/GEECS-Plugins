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
the reconciliation counters `frames_written`, `duplicates_dropped`,
`stale_skipped`, `writer_queue_drops`, `disconnect_events`, `finalized`
(bool; absent/False means the daemon died mid-scan and the tail of
`/frames` is still valid up to `N`).

## Semantics

- **Dedupe by `acq_timestamp`**: the device re-pushes its last frame at
  1 Hz with an unchanged timestamp when idle (measured 2026-08-27), and the
  gateway re-posts it. A frame whose timestamp was already written is
  dropped and counted.
- **Stale-window filter**: frames stamped earlier than the run's start-doc
  `time` minus a small margin are the gateway's cached pre-scan frame —
  skipped and counted, never written.
- **Append-per-frame with flush** (trailing flush): a crash loses at most
  the un-flushed tail, never the scan.
- The daemon **never creates directories**: the engine's save-enable plan
  owns `scans/ScanNNN/<device>/` creation; a missing directory means the
  device is skipped loudly (cross-package invariant — analysis/services
  never create scan folders).
