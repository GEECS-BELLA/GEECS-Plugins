# Tiled Integration — Current State

## What Is This

Tiled is the persistent scalar/metadata store for all GEECS Bluesky scans.
Every scan (queue-submitted or headless, on `make_run_engine(tiled=True)`)
spools its start/stop/event documents to a per-run file, and the host's
`geecs-tiled-writer` service registers each run in the Tiled catalog on
the DB server (`192.168.6.14`) at its close.  A headless engine on a
machine with no writer spools and nothing drains it — the engine warns at
startup when no fresh writer heartbeat is under its state directory.  Data
is then queryable from any Python session on the network without touching
the raw data files.

---

## Infrastructure State (last hardware-verified 2026-07-12)

### DB Server — `192.168.6.14`

- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- Tiled: **0.2.14** (upgraded from 0.2.9 on 2026-07-12) installed via
  `pip install --user 'tiled[server]'` into `~/.local`.  The install list
  is `GeecsBluesky/tiled_server/requirements.txt` (the 0.2 line + the ADBC
  drivers the adapter override below is tested against) — a range, so
  installing from it may move the version: do that only inside
  § "Upgrading the server" below (catalog migrations), never alone
- Adapter override: `~/tiled/geecs_tiled_sql.py` + `PYTHONPATH` in the
  unit's drop-in (§ "SQLite typed reads" below)
- Running as systemd service: `sudo systemctl status tiled`
  (`tiled serve config ~/tiled/config.yml`; auth + host/port + trees all
  live in that config file, not in the unit)
- Catalog DB: `~/tiled/catalog.db` (SQLite, metadata index — the one
  database Tiled needs)
- Event tables: since GeecsBluesky 0.110.0 **one Parquet file per stream
  in the scan folder** (`ScanNNN/ScanDataScanNNN-<stream>.parquet`),
  written by `geecs-tiled-writer` and registered from `readable_storage`
  like the camera stacks — so the data share must be in
  `readable_storage` (it is, for the stacks).  `~/tiled/tabular.db`
  (SQLite) holds the runs registered before that as appendable SQL tables
  (read through the override below) and stays in `writable_storage` for
  the writer's `--tables appendable` option; the history port to Parquet
  is the arc's next step.
- File storage: `~/tiled/storage/` (Tiled's own, for tables a client asks
  it to store)
- API key: stable — `single_user_api_key` in `~/tiled/config.yml` on the
  server; stored in `~/.config/geecs_python_api/config.ini` on all client
  machines

### Serving the file plugin's stacks (verified 2026-09-11)

The PVA gateway's file plugin (#806) writes each camera's frames as one
HDF5 stack under the run folder on the data share, and the writer
service's `TiledWriter` registers that file by its `file://` URI.  Two server-side
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
   `tiled_catalog.py`) reads back identical to `h5py` on the file.

The Tiled server is pip-installed and unit-less as far as `deploy/` is
concerned (no rendered unit, no `site.env` key), so these two settings
live here, not in the deployment tree.

### SQLite typed reads — the adapter override (#1020; verified 2026-10-01)

**Symptom:** a run page answers 503 (the portal) and Tiled answers 500 on
`GET /api/v1/table/partition/<uid>/primary/internal?partition=0` with

```
OSError: [SQLite] Type mismatch in column N: expected INT64 but got DOUBLE
```

**Cause:** Tiled reads its SQLite tabular storage (`tabular.db`) through
the ADBC SQLite driver, which infers each result column's Arrow type from
the rows of its *first batch* (1024) and ignores the declared column type.
SQLite stores a float NaN as NULL, so a scalar that is NaN for a scan's
first 1024 shots (a diagnostic with no beam) and has a value later is
typed INT64, and the first real value fails the whole read.  Long scans
with intermittent-beam diagnostics are the exposed shape.  No Tiled
release fixes it (0.2.18's read path is identical to 0.2.14's).

**Fix:** `GeecsBluesky/tiled_server/geecs_tiled_sql.py`, a `SQLAdapter`
subclass that reads each dataset in one ADBC batch, so every column is
typed from every row; Tiled's own cast to the declared schema does the
rest.  Install on the Tiled host — one restart:

```bash
cp <checkout>/GeecsBluesky/tiled_server/geecs_tiled_sql.py ~/tiled/
sudo systemctl edit tiled        # add the PYTHONPATH line below
sudo systemctl restart tiled
```

```ini
[Service]
Environment=HDF5_USE_FILE_LOCKING=FALSE
Environment=PYTHONPATH=/home/<user>/tiled
```

and in `~/tiled/config.yml`, under the catalog tree's `args:` beside
`writable_storage`:

```yaml
      adapters_by_mimetype:
        application/x-tiled-sql-table: "geecs_tiled_sql:GeecsSQLAdapter"
```

The `PYTHONPATH` line is required: Tiled resolves the import outside the
window in which it prepends the config directory to `sys.path`, and
without it the service fails at start (`ValueError` from `import_object`,
`ModuleNotFoundError` underneath).  The file and the drop-in survive a
`pip install -U tiled`.

**Check:** the sweep lists every dataset the *stock* reader fails on —
after the override they are informational, before it each one is an
unreadable run:

```bash
ssh <tiled-host> 'cd ~/tiled && nice -n 15 python3 -' < scripts/tiled_sweep_nan_leading.py
```

(~12 s for 220 tables; `--tabular tabular.db.bak-…` runs it against a
backup; exit status = the number of hits).  Then read an affected run the
portal's way (`read_primary_scalars`): the previously failing columns come
back `float64` with their values.

**Storage ceilings measured 2026-10-01** (the reason the tabular store is
still SQLite, and the shape of the follow-up arc):

| Engine | Ceiling | HTU today |
|---|---|---|
| PostgreSQL | a heap tuple must fit one 8 KB page and fixed-width doubles are never moved out of line: ≈1000 non-null doubles per row (900 insert, 1000 fail: `row is too big: size 8192, maximum size 8160`) | one primary table at 1294 columns, baselines at 7.5 KB — Postgres tabular storage needs the narrower per-shot row first (background telemetry as one vector column, which Postgres stores out of line at any width) |
| SQLite (the ADBC-bundled build) | 2000 columns per table (`SQLITE_MAX_COLUMN`) | 1294; half of every table is the TiledWriter's `ts_<key>` timestamp columns |
| DuckDB | no practical column limit | not chosen: Tiled gives it one pooled connection |

The same module fills null elements of a floating-point *array* column
with NaN before a PostgreSQL ingest: the ADBC PostgreSQL driver (1.11,
1.12) writes a NULL element as `0.0`, and Tiled's server turns NaN into
null on the way in (`deserialize_arrow` reads uploads through pandas), so
a missing telemetry sample inside a vector would otherwise read as zero.
NaN, `inf` and a NULL whole array survive the driver unchanged.

### Porting the SQL-stored runs to Parquet (`geecs-tiled-port-tables`, 0.111.0)

Runs registered before GeecsBluesky 0.110.0 have their stream tables in
`tabular.db`.  `geecs-tiled-port-tables` moves them to the present shape,
one table node at a time, through Tiled's API alone: read the table as the
server serves it, write `ScanNNN/ScanDataScanNNN-<stream>.parquet` beside
the s-file, delete the node and register it again under the same key as a
Parquet data source, read it back and compare.  Nothing in `tabular.db` is
modified or deleted — it is the rollback.  Run it from the worker's
checkout (the share is mounted there, the writer's `tiled_host_path` rule
gives the URI), with Tiled **up** (no restart, no cutover window — new
runs are Parquet already and never match the SQL mimetype):

```bash
cd <root>/qs-checkout/GeecsBluesky
cp ~/tiled/tabular.db ~/tiled/tabular.db.bak-$(date +%Y%m%d)-pre-port      # the SQL store is the rollback; keep a copy anyway
poetry run geecs-tiled-port-tables --dry-run \
  --alias 'Z:/data=/mnt/hdna2/data' --alias '/Volumes/hdna2/data=/mnt/hdna2/data'
poetry run geecs-tiled-port-tables --ledger ~/tiled/port-$(date +%Y%m%d).jsonl \
  --alias 'Z:/data=/mnt/hdna2/data' --alias '/Volumes/hdna2/data=/mnt/hdna2/data'
```

`--alias SRC=DST` maps the scan-folder roots other engine hosts recorded
(a Mac's `/Volumes/…`, Windows' `Z:/…`) onto this host's mount — the HTU
values above are the reference deployment's; a facility passes its own.
A run whose start document names no scan folder (the pre-claim
development runs of 2026-05..07) is skipped and reported; its table stays
in SQL storage, which the adapter override keeps readable, so the SQL
`writable_storage` entry stays in `config.yml` for them and for
`--tables appendable`.  `--limit N` / `--run UID` for a rehearsal; the
ledger (JSON lines, one record per node, the SQL data source as it was)
is what `--restore UID STREAM` re-registers from.  Rehearsed 2026-10-02 on
a throwaway Tiled over copies of both databases (§ "SQLite typed reads"
has the throwaway recipe) before the live run.

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
`library_version`; existing runs read back through
`geecs_data_utils.tiled_catalog.read_primary_scalars(run["primary"])` —
the pattern `tiled_catalog.py` / `tiled_export.py` use: the composite node's `internal` table via `.base`, **never**
`run["primary"].read()`, which downloads every camera stack and per-frame
attribute array and outer-joins their dimensions (a two-camera plugin run
took the worker host down, #834).  Ad-hoc `run["primary"]["data"]` does
**not** work under 0.2.14's composite-container layout; use `.base` for
raw node access.

**The web UI lives at `/ui`, not `/`** (verified live 0.2.14): the pip
wheel ships Tiled's built React catalog browser in `share/tiled/ui/`
(*outside* the Python package dir — easy to miss when searching the
package), and the server serves it at `http://192.168.6.14:8000/ui`. The
root `/` is only a minimal landing page. With
`allow_anonymous_access: false`, open `/ui?api_key=<key>` once — the
server moves the key into a cookie and strips the URL. The UI is a generic
catalog browser (uid-oriented; metadata, tables, array previews, downloads);
the scan-shaped quick-look workflow (day → Scan NNN → plot columns → drift)
is the Data Portal's job (GEECS-DataPortal).

### Client machines

`make_run_engine(tiled=True)` (the worker startup profile included) reads
`~/.config/geecs_python_api/config.ini` under `[tiled]` to decide whether
to **spool** the run's documents; the `geecs-tiled-writer` service on the
same host reads the same section for the catalog it registers them in
(`CLAUDE.md` § "Tiled: the spool and the writer service"):

```ini
[tiled]
uri = http://192.168.6.14:8000
api_key = <stable key>
```

---

## What Works

- The worker spools every run's documents; `geecs-tiled-writer` registers
  each run in Tiled at its close, off the engine thread ✓
- Run start/stop metadata written to catalog ✓
- Event documents (motor positions, detector scalars, timestamps) written —
  as one Parquet table per stream beside the s-file, typed, any width,
  NaN kept as NaN (0.110.0) ✓
- Scan number, scan folder, device list in run start metadata ✓
- Non-scalar device events include save directory and device `acq_timestamp` ✓
- DG645 shot control arm/disarm per step ✓
- Catalog readable from any network-connected Python session ✓
- Operator path complete — the web scanner (`GeecsScanner`) submits to
  the queueserver worker: shot control (trigger profiles) and
  setup/per-step/closeout actions all flow through it ✓
- Scalar s-file written by the engine from the run's own rows at the stop
  document (`callbacks.SFileCallback`) — no Tiled round trip ✓
- Hardware acceptance: `tests/test_phase0_hardware.py` (gated on
  `GEECS_HW=1`) runs the strict hook over a real camera
  against the live gateway — see its module
  docstring for invocation; run it to verify, no standing pass is
  recorded here

## Known Limitations

- **No TDMS output — a decided disposition, not an oversight**:
  on-shot TDMS was a poorly-implemented Master Control
  preservation and was not in use, so it was dropped outright.  Scalar
  s-files are written by the engine from the run's own rows; the offline
  re-export from Tiled (`geecs_data_utils.write_scalar_files_from_tiled`)
  runs the same join.  If LabVIEW tooling ever needs TDMS again, the
  natural shape is a post-scan Tiled→TDMS exporter alongside that
  re-export — analysis-side, no scanner integration required.
  Possible future work, not scheduled, not a gate.  The data-pipeline
  end state remains the open strategic question: does `ScanAnalysis`
  grow a Tiled reader, or keep reading exported s-files long-term?
- **Natively saved per-shot files are not served by Tiled** — only the
  file plugin's HDF5 stacks are (above: the stock HDF5 adapter; the one
  custom adapter on this server, § "SQLite typed reads", changes how the
  SQL tables are *read*, not what is served).  A device without a plugin writes its own per-shot
  files, and the run's events record the save directory and the device's
  `acq_timestamp`, not an external data source; readers join rows to files
  on disk by stamp (`geecs_data_utils.native_files`).  Serving those over
  HTTP would need the *worker* to emit stream resources for them plus a
  mimetype-matched adapter here — demand-driven work nobody has asked for,
  and moot for the vendor-only formats (HASO `.himg`) that have no Python
  reader at all.  Related undecided question: whether Bluesky native-save
  runs need a finalization/mover step for legacy filename compatibility,
  or whether direct native filenames are the canonical Bluesky path.
- **Tiled not yet read by ScanAnalysis** — post-scan analysis continues to
  use the file-based path (the s-file); `ScanAnalysis` itself does not
  read Tiled.

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
from geecs_data_utils.tiled_catalog import read_primary_scalars
df = read_primary_scalars(run["primary"])   # scalar table only — never run["primary"].read() (#834)

# Run hardware integration test (requires lab network)
cd GeecsBluesky
GEECS_HW=1 poetry run python -u -m pytest tests/test_phase0_hardware.py -m hardware -s
```
