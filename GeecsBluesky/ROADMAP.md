# GeecsBluesky — Roadmap

Current status (2026-04-21): BlueskyScanner runs STANDARD and NOSCAN scans
end-to-end against real hardware.  Motor motion, DG645-gated detector
triggering, scan numbering, per-device image saving, and the ScanManager-
compatible bridge API are all working.

---

## Completed

- [x] **TCP-backed signal cache** — coherent per-shot reads; one shared TCP
      subscriber per device populates `_shot_cache`; `get_value()` reads from
      cache with UDP fallback.
- [x] **GeecsTriggerable** — event-driven trigger via `asyncio.Queue`; waits
      for `acq_timestamp` to advance; no polling.
- [x] **GeecsMotor / GeecsSettable** — generic `Movable`; UDP polling for
      position convergence; `GeecsMotor` inherits `GeecsSettable`.
- [x] **GeecsGenericDetector** — dynamically-signalled detector from a
      `variable_list`; `localsavingpath` / `save` signals when
      `save_nonscalar_data=True`.
- [x] **BlueskyScanner bridge** — ScanManager-compatible API; STANDARD and
      NOSCAN modes; `dialog_queue` / `restore_failures` shims.
- [x] **Scan numbering** — `ScanPaths.get_next_scan_tag()` claimed before
      detectors are built so save paths are known at device-build time.
- [x] **Per-device image saving** — `_scan_with_saving` plan wrapper sets
      `localsavingpath` and `save="on"` concurrently (single `bps.mv` fanout)
      before scan; `bpp.finalize_wrapper` sets `save="off"` even on abort.
- [x] **shots_per_step** — explicit key in `config_dictionary`; not derived
      from `wait_time`; hardware timing left to DG645.

---

## 1. Error Handling — Exceptions + Interactive Recovery

### 1a. Exception taxonomy  ← *in progress*

Define a typed hierarchy in `geecs_bluesky/exceptions.py`:

```
GeecsError
├── GeecsConnectionError        # transport-level — device unreachable
├── GeecsCommandError           # device responded but rejected/failed
│   ├── GeecsCommandRejectedError   # no ACK (device offline/busy)
│   └── GeecsCommandFailedError     # ACK but device returned error status
├── GeecsTriggerTimeoutError    # acq_timestamp didn't advance in time
├── GeecsMotorTimeoutError      # position didn't converge in move_timeout
└── GeecsDeviceNotFoundError    # DB lookup failed
```

Raise typed exceptions from transport and device layers; keep devices
ignorant of dialogs or queues.

### 1b. RE-pause-based interactive recovery

**Critical design constraint**: BlueskyScanner runs an asyncio event loop in
a background thread.  The existing scanner's `threading.Event.wait()` pattern
**must not be used inside a plan** — it would freeze the TCP subscriber and
all async device tasks.

Use `bps.pause()` instead:

```python
# Inside a plan wrapper in bluesky_scanner.py:
try:
    yield from bps.mv(motor, pos)
except GeecsCommandError as e:
    dialog_queue.put(DialogRequest(exc=e, ...))
    yield from bps.pause()   # RE freezes; event loop stays alive
    # user fixes device, clicks Continue → RE.resume(); plan retries
    # user clicks Abort       → RE.abort(); finalize_wrapper cleans up
```

Retry policy (silent retries before showing dialog) belongs in the plan
wrapper, not in the device.

### 1c. GUI adaptation — reactive state monitoring

**The GUI should subscribe to RE state changes, not poll `dialog_queue` on a
timer.**  The RE already emits state transitions (idle → running → paused →
…).  When the RE transitions to `"paused"`, the GUI drains `dialog_queue`
(which carries error details) and shows the dialog.  This replaces the
200 ms timer loop in the existing scanner and eliminates dialog latency.

`BlueskyScanner` should set the interface; the GUI adapts to it — not the
other way around.

---

## 2. Tiled Server — Centralized Scalar Storage

**What**: All event data (motor positions, detector scalars, metadata) stored
in a queryable database accessible from any machine.

**Architecture**:
- Run `tiled serve catalog` on the Linux DB machine (192.168.6.14).
- `BlueskyScanner.__init__` subscribes a `TiledWriter` — every event document
  is written automatically.
- Data queryable from Jupyter via
  `tiled.client.from_uri("http://192.168.6.14:8000")`.

**Setup** (on DB machine):
```bash
pip install tiled[server]
tiled catalog create --init-db catalog.db
tiled serve catalog catalog.db --host 0.0.0.0 --port 8000
```

---

## 3. Longer-Term

- **Optimization scans**: wrap `xopt` or `scipy.optimize` in a Bluesky plan;
  the GUI already has an optimization mode.
- **Composite variables**: two motors moving together; implement as a custom
  plan calling `bps.mv(m1, p1, m2, p2)`.
- **Live plotting**: `BestEffortCallback` or custom Bokeh subscriber that
  updates during the scan.
- **Config format**: replace `config_dictionary` (legacy GUI format) with a
  proper dataclass/Pydantic model on the BlueskyScanner side; GUI serializes
  to it.
- **Separate repo**: once stable, publish `geecs-bluesky` to PyPI for use at
  other GEECS-based labs.
