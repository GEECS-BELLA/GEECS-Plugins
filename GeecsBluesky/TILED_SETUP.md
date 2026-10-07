# Tiled — the server and its clients

Tiled is the catalog of every GEECS Bluesky run: start/stop metadata, each
event stream's table, and the camera stacks, all queryable over HTTP from
any machine on the network.  The engine never talks to it — it spools each
run's documents to a file, and the `geecs-tiled-writer` service registers
the run at its close (`CLAUDE.md` § "Tiled: the spool and the writer
service"; the writer's deploy and heartbeat: `qserver/deploy/DEPLOYMENT.md`
§ The Tiled writer).  This page is the **server** side and the client
recipe.

## What the catalog holds

| Part | Where the bytes live | How Tiled serves it |
|---|---|---|
| Run metadata (start, stop, descriptors) | `~/tiled/catalog.db` (SQLite) | the catalog itself |
| Each stream's table (`primary`, `shots`, `baseline`, `<device>_stream`, …) | `ScanNNN/ScanDataScanNNN-<stream>.parquet` beside the s-file | registered from `readable_storage`, `application/x-parquet` |
| A camera's frames (the PVA file plugin's stacks) | `ScanNNN/<device>/<device>.h5` | registered from `readable_storage`, Tiled's stock HDF5 adapter |
| Tables written with the writer's `--tables appendable` | `~/tiled/tabular.db` (SQLite) | appendable SQL tables in `writable_storage` |

The scan folder is the record; Tiled is the index and the server over it.

## The server

Reference deployment (HTU): Ubuntu, the system Python, Tiled on the 0.2
line installed with `pip install --user`, served by a systemd unit `tiled`
running `tiled serve config ~/tiled/config.yml`.  Auth, host/port and the
trees all live in that config file, not in the unit.  `/fleet-status`
reports the running version (`/api/v1/`'s `library_version`).  The server
is pip-installed and unit-less as far as `deploy/` is concerned (no rendered
unit, no `site.env` key), so its settings live here.

```bash
python3 -m pip install --user 'tiled[server]>=0.2.14,<0.3'
```

A version move on a running host goes through § Upgrading, never this line
alone.

### `~/tiled/config.yml`

The catalog tree, with the data share in `readable_storage` as the Tiled
host mounts it (HTU's mount shown) — without it every Parquet table and
stack read answers 500, `Refusing to serve file://… because it is outside
the readable storage area for this server`.  `readable_storage` is an
argument of the catalog tree (a top-level key is refused by the config
schema):

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

The API key is `single_user_api_key` in the same file — stable, and
copied into every client's `config.ini` (below).

### The unit's environment

The stacks are written on Windows over SMB and read on Linux over the same
share; HDF5's file locking does not survive that path.  `sudo systemctl
edit tiled`:

```ini
[Service]
Environment=HDF5_USE_FILE_LOCKING=FALSE
```

then `sudo systemctl restart tiled`.

### Where the writer's paths come from

The writer registers each Parquet file and stack by a `file://` URI on the
**Tiled host's** mount.  When the writer runs where the share is mounted at
the same path, nothing is needed; otherwise `[Paths]
geecs_tiled_host_data_base_path` in the writer's `config.ini` names the
share as the Tiled host mounts it.

### Upgrading

```bash
sudo systemctl stop tiled
cp ~/tiled/catalog.db ~/tiled/catalog.db.bak-YYYYMMDD
cp ~/tiled/tabular.db ~/tiled/tabular.db.bak-YYYYMMDD
python3 -m pip install --user -U 'tiled[server]>=0.2.14,<0.3'
sudo systemctl start tiled
```

A version jump usually needs a catalog-schema migration: with
`init_if_not_exists: true` the service crash-loops until it is applied
(the journal shows `DatabaseUpgradeNeeded`, or `CalledProcessError` from
`tiled catalog init`).

```bash
python3 -m tiled catalog upgrade-database 'sqlite+aiosqlite:////home/<user>/tiled/catalog.db'
sudo systemctl start tiled
```

Check from any client: `/api/v1/` reports the new `library_version`, and a
recent run reads back through `read_primary_scalars` (below).

## Clients

`make_run_engine(tiled=True)` (the worker's startup profile included) and
the writer both read `~/.config/geecs_python_api/config.ini`:

```ini
[tiled]
uri = http://192.168.6.14:8000
api_key = <stable key>
```

The engine reads it only to decide whether to spool; the writer registers
into that catalog.

### Reading a run

```python
from tiled.client import from_uri
from geecs_data_utils.tiled_catalog import read_primary_scalars

c = from_uri("http://192.168.6.14:8000", api_key="<key>")
run = c.values().last()
run.metadata["start"]
df = read_primary_scalars(run["primary"])   # the per-shot table only
```

`read_primary_scalars` reads the composite node's `internal` table through
`.base`.  **Never `run["primary"].read()`**: it downloads every camera stack
and per-frame attribute array and outer-joins their dimensions, enough to
exhaust the worker host's memory.  `run["primary"]["data"]` does not exist
under the composite-container layout; use `.base` for raw node access.  `geecs_data_utils.tiled_catalog` / `tiled_export` are the
reference readers.

### The web UI

`http://192.168.6.14:8000/ui` (not `/`, which is a landing page): a generic
catalog browser — metadata, tables, array previews, downloads.  With
`allow_anonymous_access: false`, open `/ui?api_key=<key>` once; the server
moves the key into a cookie and strips the URL.  The scan-shaped workflow
(day → Scan NNN → plot columns → drift) is the Data Portal's.

## Limits

- **Per-shot files a device saves itself are not served** — only the file
  plugin's stacks and the stream tables are.  The run's events record each
  such device's save directory and `acq_timestamp`; readers join rows to
  files on disk by stamp (`geecs_data_utils.native_files`).
- **No TDMS.** Scalar s-files are written by the engine from the run's own
  rows; `geecs_data_utils.write_scalar_files_from_tiled` re-exports them
  from Tiled.
- **ScanAnalysis does not read Tiled**; it reads the s-file and the scan
  folder.
- **Appendable SQL tables read through the stock ADBC SQLite driver**,
  which types each column from the first 1024 rows: a float column null
  (NaN) for that long and valued later fails the whole read
  (bluesky/tiled#1558).  The default Parquet tables are not affected.

## Useful commands

```bash
sudo systemctl status tiled
sudo journalctl -u tiled --no-pager | tail -30
curl -s http://192.168.6.14:8000/api/v1/ | jq .library_version
```
