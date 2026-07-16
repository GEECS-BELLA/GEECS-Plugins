# Tiled Integration — Current State

## What Is This

Tiled is the persistent scalar/metadata store for all GEECS Bluesky scans.
Every scan that runs through `BlueskyScanner` writes start/stop/event documents
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

`BlueskyScanner` auto-reads Tiled URI + API key from
`~/.config/geecs_python_api/config.ini` under `[tiled]`:

```ini
[tiled]
uri = http://192.168.6.14:8000
api_key = <stable key>
```

---

## What Works

- `BlueskyScanner` connects to Tiled on startup and subscribes `TiledWriter` ✓
- Run start/stop metadata written to catalog ✓
- Event documents (motor positions, detector scalars, timestamps) written ✓
- Scan number, scan folder, device list in run start metadata ✓
- Non-scalar device events include save directory and device `acq_timestamp` ✓
- DG645 shot control arm/disarm per step ✓
- Catalog readable from any network-connected Python session ✓
- Hardware integration test (`test_bluesky_scanner.py`) passes all 6 checks ✓

## Known Limitations

- **No s-file / TDMS output** — `BlueskyScanner` writes to Tiled only.
  `ScanAnalysis` and other downstream tools still read s-files.  Data pipeline
  transition is an open strategic question (see `ROADMAP.md`).
- **Tiled not yet read by ScanAnalysis** — post-scan analysis continues to use
  the legacy file-based path.
- **GUI path still partial** — `RunControl(use_bluesky=True)` does not yet pass
  shot-control YAML, setup/closeout actions, or the scanner `ScanEvent` callback
  into `BlueskyScanner`.

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
poetry run pytest tests/test_scan_request_hardware.py -m integration -s
```
