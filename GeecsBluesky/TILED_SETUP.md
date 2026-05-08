# Tiled Integration — Current State & Next Steps

## What Is This

Tiled is the persistent scalar/metadata store for all GEECS bluesky scans.
Every scan that runs through `BlueskyScanner` writes start/stop/event documents
to a Tiled catalog on the DB server (`192.168.6.14`).  Data is then queryable
from any Python session on the network without touching the raw data files.

---

## Infrastructure State (as of 2026-05-07)

### DB Server — `192.168.6.14`

- OS: Ubuntu 22.04.5 LTS (upgraded from 18.04 this session)
- Python: 3.10.12
- Tiled: 0.2.9 installed via `pip install 'tiled[server]'` into `~/.local`
- Running as systemd service: `sudo systemctl status tiled`
- Catalog DB: `~/tiled/catalog.db` (SQLite, metadata index)
- File storage: `~/tiled/` (arrays/images)
- **Missing:** SQLite tabular storage (`~/tiled/tabular.db`) — causes 500 error
  when TiledWriter tries to write event tables. See Next Steps.

Current `systemd` service (`/etc/systemd/system/tiled.service`) uses CLI flags.
**This needs to be replaced with a config file** (see Next Steps).

API key is auto-generated on each restart and stored in `~/.config/geecs_python_api/config.ini`.
**This needs to be made stable** (see Next Steps).

### Windows DAQ Machine — `HTU-CR-1`

- Branch: `geecs-bluesky-detectors`
- `BlueskyScanner` auto-reads Tiled URI + API key from
  `%USERPROFILE%\.config\geecs_python_api\config.ini` under `[tiled]`
- `tiled[client]` installed in the Scanner GUI poetry env
- TiledWriter subscribes successfully on startup (confirmed in logs)
- **Issue:** scans fail mid-write due to missing tabular storage on server

### Mac (development)

- `[tiled]` section in `~/.config/geecs_python_api/config.ini`
- Read access confirmed: `c = from_uri("http://192.168.6.14:8000", api_key=...)`
- Catalog currently has 2 partial runs (scan 14 and 15) from testing

---

## What Works

- BlueskyScanner connects to Tiled on startup ✓
- Run start/stop metadata written to catalog ✓
- Scan number, scan folder, device list in metadata ✓
- File images saved to network drive per-shot ✓
- Catalog readable from Mac via Python ✓

## What Is Broken

- **500 error on event table write** — `SQLAdapter` needs SQLite/SQL storage
  but server only has `FileStorage`. Fix: add `tabular.db` to server config.
- **API key regenerates on restart** — needs stable key via env var.
- **Shot control (DG645) not integrated** — BlueskyScanner doesn't yet
  arm/disarm the DG645. Tested with UC_TopView in internal trigger mode.

---

## Next Steps (in order)

### 1. Switch server to config-file approach + fix storage (immediate)

Generate a stable API key on the server:
```bash
openssl rand -hex 32
# copy output
echo 'export TILED_SINGLE_USER_API_KEY="<paste key here>"' >> ~/.bashrc
source ~/.bashrc
```

Create `~/tiled/config.yml`:
```yaml
authentication:
  single_user_api_key: ${TILED_SINGLE_USER_API_KEY}
  allow_anonymous_access: false
uvicorn:
  host: 0.0.0.0
  port: 8000
trees:
  - path: /
    tree: catalog
    args:
      uri: "sqlite:////home/abmx/tiled/catalog.db"
      writable_storage:
        - "/home/abmx/tiled/storage"
        - "sqlite:////home/abmx/tiled/tabular.db"
      readable_storage:
        - "/home/abmx/tiled/storage"
      init_if_not_exists: true
```

Update systemd service:
```ini
[Unit]
Description=Tiled Data Server
After=network.target

[Service]
User=abmx
EnvironmentFile=/home/abmx/.bashrc
ExecStart=/home/abmx/.local/bin/tiled serve config /home/abmx/tiled/config.yml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Update `[tiled]` in `config.ini` on all machines with the new stable key.

### 2. DG645 shot control integration

`BlueskyScanner` needs to arm the shot controller before a scan and disarm
after.  The legacy `ScanManager` does this via device set calls.  In the
Bluesky backend this belongs as a plan wrapper around the scan — arm before
`_scan_with_saving`, disarm in the `finalize_wrapper`.

Key questions to resolve:
- Which GEECS variable arms/disarms the DG645?
- Should it be a `GeecsSettable` device or a raw UDP call?

### 3. PR and merge `geecs-bluesky-detectors`

Once steps 1-2 are verified end-to-end with a real scan (DG645 running,
events appearing in Tiled), open a PR to merge into master.

### 4. Longer term

- Add scan metadata to Tiled: scan config YAML, operator name, description
- Live plotting subscriber (BestEffortCallback or Bokeh)
- Read Tiled data in ScanAnalysis post-processing pipeline

---

## Useful Commands

```bash
# Check Tiled service on server
sudo systemctl status tiled
sudo journalctl -u tiled --no-pager | tail -30

# Read catalog from Python (Mac or any machine with tiled[client])
from tiled.client import from_uri
c = from_uri("http://192.168.6.14:8000", api_key="<key>")
run = c.values().last()
print(run.metadata["start"])

# On Windows: start GUI with Bluesky backend
cd /z/software/control-all-loasis/HTU/Active\ Version/GEECS-Plugins/GEECS-Scanner-GUI
poetry run python main.py
```
