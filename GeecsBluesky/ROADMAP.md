# GeecsBluesky — Next Steps Roadmap

Current status (2026-04-19): The GEECS Scanner GUI executes step scans via
the Bluesky RunEngine using `BlueskyScanner`. Motor motion works end-to-end.
No detectors, no data storage, no file saving yet.

---

## 1. Real Data Acquisition — Triggerable Detectors

**What**: Add camera/diagnostic devices to `BlueskyScanner._detectors` so they
are triggered and read alongside the motor at each shot.

**How**:
- `config_dictionary` passed to `reinitialize()` contains the list of active
  devices from the GUI's device table. Parse device names from it and create
  `GeecsDevice` subclass instances (e.g. `ModeImager`) via `from_db()`.
- Connect them in the RE's loop before the scan starts.
- Pass them as `detectors=[...]` to `geecs_step_scan`.
- Each device's `trigger()` then waits for `acq_timestamp` to advance (the
  DG645 shot signal), automatically gating events to the real shot rate.

**Key challenge**: knowing which `GeecsDevice` subclass to instantiate for a
given device name. Options:
  - Registry/factory: map device names → classes in a YAML config.
  - Generic fallback: `GeecsDevice` with variables listed in `config_dictionary`.

**Result**: events slow to real shot rate; each event contains motor position
+ all subscribed detector scalars (timestamp, counts, etc.) from the same shot.

---

## 2. Tiled Server — Centralized Data Storage

**What**: All scalar data (motor positions, detector readings, metadata) stored
in a queryable database accessible from any machine on the network.

**Architecture**:
- Run `tiled serve catalog` on the Linux DB machine (192.168.6.14).
- Bluesky RunEngine subscribes a `TiledWriter` — every event document is
  written to Tiled automatically during the scan.
- Tiled uses SQLite for metadata (no MongoDB needed).
- Data accessible from any machine via `tiled.client.from_uri("http://192.168.6.14:8000")`.

**Setup steps** (on DB machine):
```bash
pip install tiled[server]
tiled catalog create --init-db catalog.db
tiled serve catalog catalog.db --host 0.0.0.0 --port 8000
```

**RunEngine integration** (in `BlueskyScanner.__init__`):
```python
from tiled.client import from_uri
from bluesky_tiled import TiledWriter
client = from_uri("http://192.168.6.14:8000")
self._RE.subscribe(TiledWriter(client))
```

**Result**: every scan's scalar data is immediately queryable from a Jupyter
notebook, even while the scan is running.

---

## 3. Camera Image Saving

**What**: Camera images saved to a central location with a consistent path
convention that matches `geecs-data-utils`.

**Path convention** (from `ScanPaths` in geecs-data-utils):
```
{base}/{experiment}/Y{year}/{MM}-{Mon}/{YY}_{MM}{DD}/scans/Scan{NNN}/{device_name}/
```
Example: `//netapp/data/Undulator/Y2026/04-Apr/26_0419/scans/Scan001/UC_ModeImager/`

**Approach**:
- Add `localsavingpath` as a writable signal on `GeecsCameraBase` / `ModeImager`.
- In a Bluesky plan stub before the scan, `mv(camera.localsavingpath, target_path)`.
- Increment scan number using `ScanPaths.get_next_scan_tag()`.
- Device computers write images directly to the NetApp share (already mounted).
- Tiled or a lightweight file-watcher indexes the images by scan/shot number.

**Note**: Some devices struggled to write to NetApp at 1 Hz in the past.
Start with direct NetApp writes; fall back to local + rsync if needed.

---

## 4. Scan Numbering

**What**: Each scan gets a unique `ScanXXX` number that ties together scalar
data (Tiled) and image files (NetApp).

**How**:
- Reuse `ScanPaths.get_next_scan_tag()` from geecs-data-utils to claim the
  next scan number atomically before the scan starts.
- Pass `scan_number` as metadata into the RunEngine `start` document:
  `md={"scan_number": scan_number, ...}`.
- Use the same number in `localsavingpath` for all camera devices.
- Tiled stores the scan number as a queryable field.

---

## 5. Shots-Per-Step from the GUI

**What**: `shots_per_step` should come from the GUI's "Shots per Step" or
equivalent field, not hardcoded to 10.

**How**: The GUI passes `wait_time = (shots + 0.5) / rep_rate` in `ScanConfig`.
Extract: `shots_per_step = max(1, round(scan_config.wait_time * rep_rate))`.
The repetition rate could be read from the shot controller config or hardcoded
as a known value (e.g. 1 Hz for DG645 in internal mode).

Alternatively: expose `shots_per_step` directly in `config_dictionary` from
the GUI side.

---

## 6. NOSCAN Mode with Detectors

**What**: Fixed-position data collection (no motor movement), `N` shots.

**How**: Once detectors are wired in (step 1), NOSCAN becomes:
```python
self._RE(bp.count(self._detectors, num=self._shots_per_step))
```
Already stubbed in `BlueskyScanner._run_noscan()` — just needs detectors.

---

## 7. Longer-Term

- **Optimization scans**: wrap `xopt` or `scipy.optimize` in a Bluesky plan;
  the GUI already has an optimization mode.
- **Composite variables**: two motors moving together per a scaling law;
  implement as a custom plan that calls `bps.mv(m1, p1, m2, p2)`.
- **Live plotting**: `BestEffortCallback` or a custom Bokeh/matplotlib
  subscriber that updates during the scan.
- **Separate repo**: once stable, publish `geecs-bluesky` to PyPI for
  community use at other GEECS-based labs.
