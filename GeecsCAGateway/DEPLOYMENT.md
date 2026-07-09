# GeecsCAGateway — Deployment

How the gateway is launched, where it currently runs, how clients point at it,
how to check it is alive, and what the production shape should become. The PV
semantics clients rely on are in `PV_CONTRACT.md`; design rationale is in
`DESIGN.md`.

---

## 1. Launching the gateway

```bash
cd GeecsCAGateway
poetry install                     # caproto + pydantic + mysql-connector; no ophyd/bluesky
poetry run python -m geecs_ca_gateway --experiment Undulator
```

(`geecs-ca-gateway` is also installed as a console script — same entry point.)

Actual CLI flags (`geecs_ca_gateway/__main__.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--experiment NAME` | *required* | GEECS experiment name; also the PV namespace prefix |
| `--all-variables` | off | Expose every device variable, not just the `get='yes'` monitoring set |
| `--no-settable` | off | Do *not* add settable (control-surface) variables to the subscribed set |
| `--include-disabled` | off | Include devices not `enabled` in the experiment |
| `--derived-channels PATH` | configs-repo convention if present | Load a geecs-schemas YAML/JSON derived-channel overlay and expose computed read-only numeric PVs |
| `--show-missing` | off | Keep the transport's "missing variable(s)" notices (quiet by default — subscribed-but-idle variables are normal for monitoring) |
| `--log-level LEVEL` | `INFO` | Python logging level |

Startup is fault-tolerant per device: unreachable devices are not fatal (their
PVs sit INVALID and reconnect automatically), a device whose UDP bind fails is
skipped loudly (readbacks still stream; its `:SP` puts raise), and a device
whose DB spec cannot be built is skipped with a warning. See
`PV_CONTRACT.md` §7.

### Config resolution

The gateway reads no config file of its own. Everything comes from:

1. **`~/.config/geecs_python_api/config.ini`** — `[Paths] geecs_data` points at
   the GEECS user-data directory (the same file every GEECS-Plugins package
   reads).
2. **`{geecs_data}/Configurations.INI`** — `[Database]` section
   (`ipaddress`, `port`, `name`, `user`, `password`) for the MySQL connection.
3. **The GEECS database** — the source of truth for the served set: devices,
   endpoints, variables, types, units, limits, settability. On a DB change,
   restart the gateway (this matches the existing master-GUI reboot pattern;
   hot reload is a later nicety).

Optional derived readbacks come from a schema-validated overlay file in the
configs repo. By convention, the gateway auto-loads:

```text
GEECS-Plugins-Configs/
  scanner_configs/
    experiments/
      <Experiment>/
        gateway/
          derived_channels.yaml
```

The file is not a replacement for the DB; it only adds computed numeric PVs on
top of the DB-backed raw device set. Same-device inputs are computed from one
coherent source frame:

```yaml
schema_version: 1
derived_channels:
  - device: TargetChamberPressure
    variable: Pressure
    expression: "10**(v - 6)"
    inputs:
      - symbol: v
        device: U_VacuumGauge
        variable: "AI_mean.Channel 0"
    egu: Torr
    precision: 6
    description: "Convectron pressure from U_VacuumGauge analog input 0"
```

Cross-device derived PVs use latest-value semantics and must declare
`stale_after`. The gateway recomputes when any input updates, and the status
loop marks the output `INVALID/UDF` if any input is missing or stale even when
all sources go quiet. These are advisory software-status PVs, not hard safety
interlocks:

```yaml
schema_version: 1
derived_channels:
  - device: LaserPermit
    variable: OK
    expression: "pressure < 1e-5 and ready > 0"
    inputs:
      - symbol: pressure
        device: TargetChamberPressure
        variable: Pressure
      - symbol: ready
        device: Amp4Shutter
        variable: Ready
    stale_after: 2.0
    description: "Latest-value advisory status for laser shots"
```

The configs repo root is resolved the same way as scanner configs:
`GEECS_SCANNER_CONFIG_DIR` may point directly at
`scanner_configs/experiments`, `GEECS_PLUGINS_CONFIGS` may point at the
configs repo root, or `~/.config/geecs_python_api/config.ini`
`[Paths] scanner_config_root_path` may point at the configs repo root. If the
conventional file is absent, the gateway starts normally with no derived PVs.
Use `--derived-channels /path/to/file.yaml` only to override the convention.

For the systemd service, keep a checked-out configs repo on the gateway host
and ensure one of those resolution paths is configured; then a normal restart
reloads both the DB-backed raw PVs and the derived overlay.

MySQL access uses the pure-Python connector (`use_pure=True`) — the C extension
has crashed silently on the lab Windows machines.

### Experiment selection and `subscribed_only` semantics

The default served set is the experiment's **per-shot monitoring subset**: each
enabled device's `get='yes'` variables from `expt_device_variable` (~377 PVs
for Undulator instead of the ~8600-variable firehose), **plus** each of those
devices' settable variables (the *control surface* — camera
`save`/`localsavingpath`, magnet setpoints — which clients need for writes
regardless of what is monitored; disable with `--no-settable`).

A device with **zero** `get='yes'` variables is *not* dropped: it keeps its
settable variables (an empty monitoring list is not "no PVs"), so the `:SP`
control surface of a device nobody logs per shot still exists. A device that
would expose nothing at all (nothing subscribed, nothing settable) is skipped
entirely — no idle TCP/UDP connections. Config building costs three batched
DB queries total (endpoints, variable metadata, get-list), not per-device
lookups, so startup is quick even for 100+ device experiments.

### Network scoping

Which subnet the CA server serves/beacons on is controlled by the standard
EPICS server env vars, set in the deployment environment:

```bash
export EPICS_CAS_INTF_ADDR_LIST="192.168.6.14"     # serve only on the lab interface
export EPICS_CAS_BEACON_ADDR_LIST="192.168.6.255"  # beacon to the lab subnet
```

Serving CA only on the intended subnet is the write-safety residual from
`DESIGN.md` — GEECS enforces value limits server-side, but there is no CA-level
auth.

---

## 2. Current dev deployment — and its risks

The gateway currently runs on the lab server **192.168.6.14**, alongside:

- the GEECS MySQL database (which the gateway itself reads at startup),
- the Tiled server (`http://192.168.6.14:8000`) and its metadata database.

This is fine for the pilot; name it for what it is — a stack of load-bearing
services on one box. Plainly:

- **Single point of failure.** One host outage takes down device PVs (Bluesky
  scans, Phoebus displays), the experiment DB, *and* Tiled data access at once.
  Nothing else in the lab restores any of them automatically.
- **Resource contention.** MySQL, Tiled ingestion, and the gateway's asyncio
  loop share CPU/RAM/disk I/O. The gateway is light (~100 updates/s), but a
  heavy Tiled write burst or a runaway query degrades everything together.
- **Upgrade coupling.** An OS upgrade or reboot for any one service (e.g. the
  22.04 migration) is downtime for all of them, and Python/library upgrades on
  a shared host risk cross-breaking environments. Keep each service in its own
  venv/poetry env at minimum.
- **Storage growth.** Tiled's data and metadata grow with every scan; MySQL
  grows slowly; logs grow steadily. Unmonitored, the shared disk filling up is
  the most likely way this box fails.

Storage placement rules:

- **Large scan/image files belong on the NetApp** (the data share) — Tiled
  should *reference* assets on NetApp, not ingest bulk image data onto the
  server's local disk.
- **The Tiled metadata database stays on proper local/server storage with
  backups.** Do **not** move a live database onto NetApp — SQLite (and DB
  engines generally) over SMB/NFS is a corruption lottery (locking semantics,
  cache coherence). Local disk + scheduled backup dumps is the right split:
  metadata local, assets on NetApp.
- The gateway itself is stateless (everything rebuilds from the DB at startup)
  — it needs no storage beyond logs.

---

## 3. Client-side recipe

Battle-tested on the Windows control machines (live sessions, 2026-07-06).

### Point CA clients at the gateway

Preferred — the shared GEECS config file, so the gateway host lives with the
other infrastructure addresses instead of in every shell:

```ini
# ~/.config/geecs_python_api/config.ini
[epics]
ca_addr_list = 192.168.6.14
# ca_auto_addr_list = NO    # optional; defaults to NO when applied from here
```

`geecs_bluesky` applies this **at package import** (before any CA library
loads — libca reads the env when its context is created): it sets
`EPICS_CA_ADDR_LIST` from `ca_addr_list`, and `EPICS_CA_AUTO_ADDR_LIST`
(default `NO` — a directed address list plus broadcast is rarely intended).

**An exported `EPICS_CA_ADDR_LIST` always wins** over the config value
(`os.environ.setdefault` semantics) — so a shell override for testing behaves
as expected, and a stale export can silently shadow the config: check the env
first when name resolution misbehaves.

### Switch the Scanner GUI onto the Bluesky/CA backend

```powershell
# PowerShell
$env:GEECS_USE_BLUESKY = "1"                          # 1/true/yes/on → BlueskyScanner
$env:GEECS_BLUESKY_ACQUISITION_MODE = "strict_shot_control"   # or free_run_time_sync
```

```bat
:: cmd.exe — note: no quotes, no $env:
set GEECS_USE_BLUESKY=1
set GEECS_BLUESKY_ACQUISITION_MODE=strict_shot_control
```

Both variables must be set in the **same shell that launches the GUI**
(PowerShell `$env:` and cmd `set` affect only that process tree, not the
system). `GEECS_BLUESKY_ACQUISITION_MODE` defaults to strict; strict requires a
reachable shot-control device with an `ARMED` state — use `free_run_time_sync`
for free-running trigger acquisition.

### Windows notes

- **The caRepeater PATH warning is benign.** On first CA use you may see
  `Unable to spawn caRepeater` / caRepeater-not-in-PATH. The executable ships
  *inside* the `epicscorelibs` wheel (which `aioca` bundles — no system EPICS
  install exists to put it on PATH). Everything works without the repeater
  when a single CA client runs per host; to silence the warning, add its
  directory to PATH. Locate it with:

  ```powershell
  poetry run python -c "import pathlib, epicscorelibs; print(next(pathlib.Path(epicscorelibs.__file__).parent.rglob('caRepeater*')))"
  ```

- **Firewall rules** — Channel Access uses:

  | Port | Protocol | Purpose |
  |---|---|---|
  | 5064 | UDP | name-resolution search (client → gateway) |
  | 5064 | TCP | circuit: data, monitors, puts |
  | 5065 | UDP | server beacons |

  Client machines need outbound UDP+TCP 5064 to the gateway host (and UDP 5065
  inbound for beacons, though clients work without them). Windows Defender
  prompts on first use — allow for the lab (private) network profile.

- **Reachability check** (TCP side only — `Test-NetConnection` cannot probe
  UDP, so name resolution can still fail with this green):

  ```powershell
  Test-NetConnection 192.168.6.14 -Port 5064
  ```

---

## 4. Smoke tests — "is it alive?"

Copy-pasteable, using the caproto CLI tools that ship with the gateway's own
environment (`poetry run` from `GeecsCAGateway/`; plain `caget`/`caput` from
EPICS base work identically). Substitute your experiment/device.

```bash
# 1. Gateway process up and serving? (heartbeat ticks every 5 s)
poetry run caproto-get Undulator:CAGateway:VERSION Undulator:CAGateway:UPTIME
poetry run caproto-monitor Undulator:CAGateway:HEARTBEAT   # Ctrl-C after a couple of ticks

# 2. Is a given device live behind it?
poetry run caproto-get Undulator:U_S1H:CONNECTED           # → Connected
poetry run caproto-get Undulator:CAGateway:DEVICES_CONNECTED

# 3. Readback streaming? (should tick with the device's ~1–5 Hz stream)
poetry run caproto-monitor Undulator:U_S1H:Current

# 3b. Derived readback loaded from the configs repo? (example: target chamber)
poetry run caproto-get -a Undulator:TargetChamberPressure:Pressure

# 4. Write path (pick a harmless settable variable; put-completion = GEECS
#    convergence, so this blocks until the device converges)
poetry run caproto-get Undulator:U_S1H:Current             # note the value
poetry run caproto-put Undulator:U_S1H:Current:SP <same value>
```

Interpretation:

- Step 1 fails → gateway down, or name resolution broken
  (`EPICS_CA_ADDR_LIST`, firewall — §3).
- Step 2 shows `Disconnected` / MAJOR → gateway fine, that device off or
  unreachable; readbacks are stale (INVALID) until it returns.
- Step 3 silent but step 2 `Connected` → the variable simply isn't streaming
  right now (analysis off, idle scope) — normal for a monitoring gateway.
- Step 4 raises → see `PV_CONTRACT.md` §7 (a put failure is a *GEECS* set
  failure or a startup UDP-bind skip, reported honestly).

No `python -m geecs_ca_gateway.smoke` helper is shipped: a useful one needs a
CA client dependency and network-timeout handling that don't fit in a
hermetically-testable ~60 lines, and the four commands above cover the need.
(Offline, `python -m geecs_ca_gateway.demo` remains the self-checking
no-hardware smoke test.)

---

## 5. Target production shape

### systemd unit

The unit for the lab deployment ships in
[`deploy/geecs-ca-gateway.service`](deploy/geecs-ca-gateway.service), with
install/verify/upgrade steps in [`deploy/README.md`](deploy/README.md):

```bash
sudo cp deploy/geecs-ca-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geecs-ca-gateway
```

- `Restart=on-failure` + `RestartSec=10` is enough: startup is cheap and
  idempotent (config rebuilds from the DB; clients' CA monitors reconnect
  automatically when the server returns). Config building is three batched DB
  queries, so PVs appear within seconds of start.
- A **restart is also the upgrade and the DB-resync mechanism**: `git pull &&
  poetry install && systemctl restart geecs-ca-gateway` — matching the
  existing GEECS master-GUI reboot pattern for device-set changes.
- **DB edits don't need shell access**: any CA client can write the
  `[Experiment:]CAGateway:RESTART` PV (devIocStats `SYSRESET` pattern) and
  the gateway exits with code 86, which the unit's `RestartForceExitStatus`
  turns into a relaunch — serving the freshly edited device/get-list set a
  few seconds later. A Phoebus button, a GUI menu action, or a one-liner:

  ```bash
  caproto-put Undulator:CAGateway:RESTART Restart
  ```
- Later sharding for fault isolation = more units with different
  `--experiment`/subsystem configs; CA name resolution makes this invisible to
  clients (DESIGN.md).

### Log expectations

journald captures stdout (`%(asctime)s %(levelname)s %(name)s: %(message)s`).
A healthy steady state is **quiet**. What you should see:

- Startup: `from_geecs_experiment(...): built N/M device spec(s)`,
  `serving N PV(s) across M device(s)`, `opened UDP to N/M device(s)`,
  `started N subscription supervisor(s)`, then one `X: subscription live` per
  device.
- State changes only, once per episode: `X: unreachable/dropped; retrying (up
  to every 30s)` (warning) and `X: reconnected` (info). No per-attempt
  tracebacks for devices that are simply off.
- One-time warnings worth acting on: `PV name collision` (fatal),
  `UDP bind/connect ... failed — starting without it` (that device's `:SP` is
  dead until restart), per-variable coercion/enum-mismatch warnings (DB type
  hygiene), `Discarding UDP reply ...` (a slow device blew its exe budget).

Anything *chatty* at steady state is a bug or a DB hygiene problem — the
"missing variable(s)" notices are already filtered by default.

### Archiver volume — pointer

When the Archiver Appliance lands, per-PV archive-rate control belongs in the
**MDEL/ADEL (archive event mask) split or the appliance's sampling policies —
not in the value deadband**, which must stay 0.0 (suppressing real
sub-tolerance motion from live monitors was a shipped production bug). The full
plan, including the `DBE_LOG`-gated-by-DB-tolerance option, is in `DESIGN.md`
→ "Honest gaps / next steps" → *Archive-rate control*.
