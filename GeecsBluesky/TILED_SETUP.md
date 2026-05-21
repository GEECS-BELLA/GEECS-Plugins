# Tiled Integration — Current State

## What Is This

Tiled is the persistent scalar/metadata store for all GEECS Bluesky scans.
Every scan that runs through `BlueskyScanner` writes start/stop/event documents
to a Tiled catalog on the DB server (`192.168.6.14`).  Data is then queryable
from any Python session on the network without touching the raw data files.

---

## Infrastructure State (as of 2026-05-08)

### DB Server — `192.168.6.14`

- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- Tiled: 0.2.9 installed via `pip install 'tiled[server]'` into `~/.local`
- Running as systemd service: `sudo systemctl status tiled`
- Catalog DB: `~/tiled/catalog.db` (SQLite, metadata index)
- Tabular storage: `~/tiled/tabular.db` (SQLite, event tables)
- File storage: `~/tiled/storage/`
- API key: stable — set via `TILED_SINGLE_USER_API_KEY` env var, stored in
  `~/.config/geecs_python_api/config.ini` on all client machines

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
- DG645 shot control arm/disarm per step ✓
- Catalog readable from any network-connected Python session ✓
- Hardware integration test (`test_bluesky_scanner.py`) passes all 6 checks ✓

## Known Limitations

- **No s-file / TDMS output** — `BlueskyScanner` writes to Tiled only.
  `ScanAnalysis` and other downstream tools still read s-files.  Data pipeline
  transition is an open strategic question (see `ROADMAP.md`).
- **Tiled not yet read by ScanAnalysis** — post-scan analysis continues to use
  the legacy file-based path.

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
poetry run python test_bluesky_scanner.py
```
