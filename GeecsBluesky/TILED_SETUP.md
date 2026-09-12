# Tiled Integration — Current State

## What Is This

Tiled is the persistent scalar/metadata store for all GEECS Bluesky scans.
Every scan (queue-submitted or headless, on `make_run_engine(tiled=True)`) writes start/stop/event documents
to a Tiled catalog on the DB server (`192.168.6.14`).  Data is then queryable
from any Python session on the network without touching the raw data files.

---

## Infrastructure State (last hardware-verified 2026-07-12)

### DB Server — `192.168.6.14`

- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- Tiled: **0.2.14** (upgraded from 0.2.9 on 2026-07-12) installed via
  `pip install --user 'tiled[server]'` into `~/.local`
- Running as systemd service: `sudo systemctl status tiled`
  (`tiled serve config ~/tiled/config.yml`; auth + host/port + trees all
  live in that config file, not in the unit)
- Catalog DB: `~/tiled/catalog.db` (SQLite, metadata index)
- Tabular storage: `~/tiled/tabular.db` (SQLite, event tables)
- File storage: `~/tiled/storage/`
- API key: stable — `single_user_api_key` in `~/tiled/config.yml` on the
  server; stored in `~/.config/geecs_python_api/config.ini` on all client
  machines

### Serving the file plugin's stacks (verified 2026-09-11)

The PVA gateway's file plugin (#806) writes each camera's frames as one
HDF5 stack under the run folder on the data share, and the worker's
`TiledWriter` registers that file by its `file://` URI.  Two server-side
facts, both found by failure on the first run (Scan007 of 26_0911):

1. **`readable_storage` must include the data share** as mounted on the
   Tiled host.  Without it the array read answers 500, `Refusing to serve
   file://…/ScanNNN/<device>/<device>.h5 because it is outside the
   readable storage area for this server`.  `readable_storage` is an
   argument of the catalog tree (Tiled's `CatalogConfig`; a top-level key
   is refused by the config schema), so in `~/tiled/config.yml` it sits
   under the tree's `args:` beside `uri:` / `writable_storage:` — the HTU
   server's form, with its data mount as the example:

   ```yaml
   trees:
     - path: /
       tree: catalog
       args:
         uri: "sqlite:////home/<user>/tiled/catalog.db"
         writable_storage:
           - "/home/<user>/tiled/storage"
           - "sqlite:////home/<user>/tiled/tabular.db"
         readable_storage:
           - "/home/<user>/tiled/storage"
           - "/mnt/hdna2/data"
         init_if_not_exists: true
   ```

2. **`HDF5_USE_FILE_LOCKING=FALSE` in the service's environment.**  The
   stacks are written on Windows over SMB and read on Linux over the same
   share; HDF5's file locking does not survive that path.  On a systemd
   host, `sudo systemctl edit tiled` and add:

   ```ini
   [Service]
   Environment=HDF5_USE_FILE_LOCKING=FALSE
   ```

   then `sudo systemctl restart tiled`.  The check: a plugin-written run's
   array (`run[<device>]` through the same client pattern as
   `tiled_readback.py`) reads back identical to `h5py` on the file.

The Tiled server is pip-installed and unit-less as far as `deploy/` is
concerned (no rendered unit, no `site.env` key), so these two settings
live here, not in the deployment tree.

### Upgrading the server (verified 2026-07-12, 0.2.9 → 0.2.14)

```bash
sudo systemctl stop tiled
cp ~/tiled/catalog.db ~/tiled/catalog.db.bak-YYYYMMDD
cp ~/tiled/tabular.db ~/tiled/tabular.db.bak-YYYYMMDD
python3 -m pip install --user -U 'tiled[server]'
sudo systemctl start tiled
```

**Expect a catalog-schema migration**: with `init_if_not_exists: true` in
the serve config, the service crash-loops after a version jump until the
alembic migration is applied (journal shows `DatabaseUpgradeNeeded` /
`CalledProcessError` from `tiled catalog init`). Fix:

```bash
python3 -m tiled catalog upgrade-database 'sqlite+aiosqlite:////home/<user>/tiled/catalog.db'
sudo systemctl start tiled
```

Post-upgrade verification from any client: `/api/v1/` reports the new
`library_version`; existing runs read back (`run["primary"].read()` — the
pattern `tiled_export.py` / `tiled_readback.py` use — survived 0.2.14's
composite-container change; ad-hoc `run["primary"]["data"]` does **not**,
use `.base` for raw node access).

**The web UI lives at `/ui`, not `/`** (verified live 0.2.14): the pip
wheel ships Tiled's built React catalog browser in `share/tiled/ui/`
(*outside* the Python package dir — easy to miss when searching the
package), and the server serves it at `http://192.168.6.14:8000/ui`. The
root `/` is only a minimal landing page. With
`allow_anonymous_access: false`, open `/ui?api_key=<key>` once — the
server moves the key into a cookie and strips the URL. The UI is a generic
catalog browser (uid-oriented; metadata, tables, array previews, downloads);
the scan-shaped quick-look workflow (day → Scan NNN → plot columns → drift)
is the GEECS scan browser's job (GEECS-Console).

### Client machines

`make_run_engine(tiled=True)` (the worker startup profile included) auto-reads Tiled URI + API key from
`~/.config/geecs_python_api/config.ini` under `[tiled]`:

```ini
[tiled]
uri = http://192.168.6.14:8000
api_key = <stable key>
```

---

## What Works

- The session connects to Tiled on startup and subscribes `TiledWriter` ✓
- Run start/stop metadata written to catalog ✓
- Event documents (motor positions, detector scalars, timestamps) written ✓
- Scan number, scan folder, device list in run start metadata ✓
- Non-scalar device events include save directory and device `acq_timestamp` ✓
- DG645 shot control arm/disarm per step ✓
- Catalog readable from any network-connected Python session ✓
- GUI path complete — `GEECS-Console` submits `ScanRequest`s to the
  queueserver worker: shot control (trigger profiles) and
  setup/per-step/closeout actions all flow
  through it (the legacy `GEECS-Scanner-GUI` path was deleted with G3) ✓
- Scalar s-file exported from Tiled best-effort after each scan ✓
- Hardware acceptance: `tests/test_phase0_hardware.py` (gated on
  `GEECS_HW=1`) runs stock `bp.count` / `bp.list_scan` over a real camera
  against the live gateway — see its module
  docstring for invocation; run it to verify, no standing pass is
  recorded here

## Known Limitations

- **No TDMS output — a decided disposition, not an oversight**
  (2026-07-10): on-shot TDMS was a poorly-implemented Master Control
  preservation and was not in use, so it was dropped outright.  Scalar
  s-files are exported from Tiled best-effort after a scan (needs the Tiled
  client extra and a readable run).  If LabVIEW tooling ever needs TDMS
  again, the natural shape is a post-scan Tiled→TDMS exporter alongside the
  s-file exporter — analysis-side, no scanner integration required.
  Possible future work, not scheduled, not a gate.  The data-pipeline
  end state remains the open strategic question: does `ScanAnalysis`
  grow a Tiled reader, or keep reading exported s-files long-term?
- **No server-side adapters for custom GEECS asset specs** — Tiled/
  databroker readers for them were never written, so external assets
  (per-shot image files etc.) are fillable only client-side from the
  datum-id metadata, not served filled by Tiled itself.  Related undecided
  question: whether Bluesky native-save runs need a finalization/mover step
  for legacy filename compatibility, or whether direct native filenames are
  the canonical Bluesky path.
- **Tiled not yet read by ScanAnalysis** — post-scan analysis continues to
  use the file-based path (bridged by the s-file export); `ScanAnalysis`
  itself does not read Tiled.

---

## Useful Commands

```bash
# Check Tiled service on server
sudo systemctl status tiled
sudo journalctl -u tiled --no-pager | tail -30

# Read catalog from Python (any machine with tiled[client])
from tiled.client import from_uri
c = from_uri("http://192.168.6.14:8000", api_key="<key>")
run = c.values().last()
print(run.metadata["start"])
df = run["primary"].read()

# Run hardware integration test (requires lab network)
cd GeecsBluesky
GEECS_HW=1 poetry run python -u -m pytest tests/test_phase0_hardware.py -m hardware -s
```
