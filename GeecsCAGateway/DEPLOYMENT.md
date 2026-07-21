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

## 3. Client-side recipe — new-client onboarding

Everything a fresh client machine (lab network or VPN) needs, in the order a
new user wants it: raw PV access first, then the shared config file, then
Tiled readback. Battle-tested on the Windows control machines and over
routed VPN (live sessions since 2026-07-06).

**A client needs nothing from GEECS-Plugins.** That is the whole point of
the gateway: GEECS is consumed as standard EPICS PVs and standard Tiled
HTTP, so the monorepo is only required on machines that *submit scans*:

| Task | What you install | From this repo |
|---|---|---|
| Read / write PVs | any CA client — `pip install caproto`, pyepics, EPICS base `caget`, Phoebus | nothing |
| Live displays | Phoebus (point `EPICS_CA_ADDR_LIST` at the gateway) | nothing |
| Read scan data | `pip install "tiled[client]"` + the API key | nothing |
| Submit scans | GEECS-Console / GeecsBluesky | repo checkout + poetry |

### First contact — PVs with nothing but a CA client

No repo checkout and no config file required; any Channel Access client
works. The caproto CLI tools are the lightest install:

```bash
python -m venv ca && source ca/bin/activate
pip install caproto

export EPICS_CA_ADDR_LIST=192.168.6.14   # directed name search — routes over VPN
export EPICS_CA_AUTO_ADDR_LIST=NO        # don't also broadcast on your local subnet

caproto-get undulator:cagateway:version undulator:cagateway:uptime
caproto-monitor undulator:cagateway:heartbeat   # ticks every 5 s; Ctrl-C to stop
```

```powershell
# PowerShell equivalent (same shell that runs the client)
$env:EPICS_CA_ADDR_LIST = "192.168.6.14"
$env:EPICS_CA_AUTO_ADDR_LIST = "NO"
caproto-get undulator:cagateway:version
```

If the heartbeat ticks, you are connected — continue with the §4 smoke tests
to poke real device PVs.

PV names are `experiment:device:variable` with every component lowercased
from the GEECS names, and each settable variable additionally exposes
`…:SP` for writes — the normative naming/typing/alarm contract is
`PV_CONTRACT.md`. The served set is the experiment's `get='yes'` monitoring
subset plus each device's settable control surface (§1), so a variable
visible in GEECS Master Control is usually reachable here under its
lowercased name.

**Off-subnet (VPN) notes.** CA name search with an explicit address list is
*directed unicast* UDP, which routes over VPN — this is why the recipe above
works off-subnet. Server *beacons* ride UDP broadcast and do **not** cross
routed paths, so a long-lived display (Phoebus) that was searching while the
gateway was down can look stuck for minutes after the gateway returns;
restart the display (§5). The Python stack issues fresh searches per polling
cycle and recovers on its own. Keep `EPICS_CA_AUTO_ADDR_LIST=NO` — a
broadcast on your home/VPN subnet finds nothing and is never what you want.

### Point the Python stack at the gateway

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

### Running scans (dev branch)

The Scanner-GUI backend toggle formerly documented here
(`GEECS_USE_BLUESKY`, `GEECS_BLUESKY_ACQUISITION_MODE`) is **gone on
`dev`**: the Bluesky/CA path is the only engine, the legacy Scanner GUI is
un-launchable, and acquisition mode is declared per scan in the
`ScanRequest` itself (`acquisition: free_run | strict`) rather than by
environment variable — a request declares intent. Scans are submitted from
**GEECS-Console** (the PySide6 operator console) or headless via
`geecs_bluesky.GeecsSession.run(ScanRequest)`. (`master` still carries the
legacy scanner line and its env toggle.)

### Tiled — reading scan data back

Every Bluesky scan's documents (start/stop metadata plus the per-shot
scalar event table) land in the Tiled catalog on the same host,
`http://192.168.6.14:8000`. Client needs: `pip install "tiled[client]"`
and the API key — ask the gateway operator; the key is deliberately not in
this public repo.

```ini
# ~/.config/geecs_python_api/config.ini — the scanner reads this same section
[tiled]
uri = http://192.168.6.14:8000
api_key = <ask>
```

```python
from tiled.client import from_uri

c = from_uri("http://192.168.6.14:8000", api_key="<key>")
run = c.values().last()        # most recent scan
run.metadata["start"]          # scan number, device list, the ScanRequest…
df = run["primary"].read()     # per-shot scalar table (pandas DataFrame)
```

A generic web catalog browser is served at `http://192.168.6.14:8000/ui`
(first visit: `/ui?api_key=<key>` — the server moves the key into a cookie
and strips the URL). Server-side state and upgrade notes live in
`GeecsBluesky/TILED_SETUP.md`; the scan-shaped browsing workflow (day →
Scan NNN → plot columns) is GEECS-Console's scan browser.

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

Copy-pasteable, using the caproto CLI tools. Run them from the gateway's own
environment (`poetry run` from `GeecsCAGateway/`) or from any client venv
with `caproto` installed (§3 first-contact recipe — drop the `poetry run`
prefix); plain `caget`/`caput` from EPICS base work identically. Substitute
your experiment/device.

```bash
# 1. Gateway process up and serving? (heartbeat ticks every 5 s)
poetry run caproto-get undulator:cagateway:version undulator:cagateway:uptime
poetry run caproto-monitor undulator:cagateway:heartbeat   # Ctrl-C after a couple of ticks

# 2. Is a given device live behind it?
poetry run caproto-get undulator:u_s1h:connected           # → Connected
poetry run caproto-get undulator:cagateway:devices_connected

# 3. Readback streaming? (should tick with the device's ~1–5 Hz stream)
poetry run caproto-monitor undulator:u_s1h:current

# 3b. Derived readback loaded from the configs repo? (example: target chamber)
poetry run caproto-get -a undulator:targetchamberpressure:pressure

# 4. Write path (pick a harmless settable variable; put-completion = GEECS
#    convergence, so this blocks until the device converges)
poetry run caproto-get undulator:u_s1h:current             # note the value
poetry run caproto-put undulator:u_s1h:current:SP <same value>
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
  `[experiment:]cagateway:restart` PV (devIocStats `SYSRESET` pattern) and
  the gateway exits with code 86, which the unit's `RestartForceExitStatus`
  turns into a relaunch — serving the freshly edited device/get-list set a
  few seconds later. A Phoebus button, a GUI menu action, or a one-liner:

  ```bash
  caproto-put undulator:cagateway:restart Restart
  ```
- Later sharding for fault isolation = more units with different
  `--experiment`/subsystem configs; CA name resolution makes this invisible to
  clients (DESIGN.md).

### Host reboots and network mounts (verified live 2026-07-15)

The gateway host typically holds the configs repo on a NAS mount (the
derived-channels overlay resolves through it — §1). That makes the mount a
**boot-order dependency with a silent failure mode**: a hand-mounted share
(no fstab entry), or an fstab mount attempted while the NAS is down,
simply isn't there when the service starts — and the gateway then starts
*normally with no derived PVs* (by design). The only symptom is the
derived channels missing; every other PV is healthy. This exact chain
happened after a 2026-07-15 site power cycle where the NAS came up after
the gateway host.

Make the whole chain self-healing once:

```bash
# root-only credentials file (never put the password in fstab):
#   /root/.smb-credentials   (chmod 600)  containing username=/password=/domain=
# fstab entry — the three options are the point:
//<nas-host>/<share>  /mnt/<share>  cifs  credentials=/root/.smb-credentials,_netdev,nofail,x-systemd.automount  0  0
sudo systemctl daemon-reload
```

- `x-systemd.automount` mounts **on first access** instead of once at
  boot — a NAS that comes up hours after the host is a non-event.
- `nofail` keeps a dead NAS from wedging boot; `_netdev` orders after
  networking.
- Teach the unit the dependency so a missing mount *delays* the gateway
  instead of silently starting it configless:
  `sudo systemctl edit geecs-ca-gateway` →
  `[Unit]` / `RequiresMountsFor=/mnt/<share>`.

Verify the cycle without rebooting: `umount` the share, `systemctl start
mnt-<share>.automount` (at boot it arms itself; mid-session you start it
once), then `ls` the mountpoint — the listing should trigger the mount.
After fixing, `systemctl restart geecs-ca-gateway` and confirm a derived
PV answers.

One client-side footnote for off-subnet operators (VPN/routed): after a
gateway restart, long-lived displays (Phoebus) can look stuck for minutes
because CA beacons don't cross routed paths — restart the display. Details
in §3's off-subnet notes.

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
