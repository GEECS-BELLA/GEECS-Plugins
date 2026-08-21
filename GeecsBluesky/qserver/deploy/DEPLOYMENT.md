# GEECS Queueserver Deployment

How to install the bluesky-queueserver RE Manager as a service on a dedicated
Ubuntu host, what local services it depends on, and how an operator verifies
the manager is ready to accept scan requests. The launch mechanics are in
`../launch_re_manager.sh`; day-to-day operator checks and known failure modes
are in `../README.md`.

---

## 1. Host prerequisites

Use a dedicated Ubuntu 22.04 host on the lab network. The worker must be able
to reach:

- the Channel Access gateway host,
- the GEECS MySQL database,
- the Tiled server when Tiled publishing is enabled,
- the mounted GEECS data share used for scan-folder claim, ScanInfo writes,
  native assets, and legacy s-file export.

Install Python 3.11, Poetry, Redis, and a checkout of this repository:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv redis-server
sudo systemctl enable --now redis-server.service
```

Clone the repository if the host does not have it yet:

**Everything below runs AS THE SERVICE ACCOUNT** (the unit's `User=`,
`geecs` in this template — create it *with a login shell*, or substitute
`sudo -u geecs -H bash` wherever `sudo -u geecs -i` appears; a
`/usr/sbin/nologin` account fails `-i` with "account is currently not
available"): Poetry keys project virtualenvs under the
*invoking user's* cache, so an env installed by an admin account is
invisible to the service — the unit would crash-loop on a fresh,
dependency-less env while admin-side verification passes. Clone, install,
and verify as the service user (`sudo -u geecs -i` or a direct login):

```bash
sudo mkdir -p /opt/geecs && sudo chown geecs:geecs /opt/geecs
sudo -u geecs git clone https://github.com/GEECS-BELLA/GEECS-Plugins.git /opt/geecs/GEECS-Plugins
```

Install Poetry **as the service user** by the method approved for the host
image (note where it lands — the official installer uses `~/.local/bin`,
which systemd's service PATH does not include; the unit template therefore
takes poetry's absolute path). Then install the worker environment,
pointing poetry at 3.11 explicitly (jammy's default `python3` is 3.10,
which this package refuses — the repo's documented top environment
failure):

```bash
sudo -u geecs -i
cd /opt/geecs/GEECS-Plugins/GeecsBluesky
poetry env use python3.11
poetry install --extras "ca tiled qserver"
```

The `qserver` extra is the queueserver dependency bundle. If that extra has
not landed on the deployment branch yet, install the branch that provides it
before enabling the systemd unit.

Redis is provided by the Ubuntu `redis-server` package and should stay bound
to `127.0.0.1`; see `redis.conf-notes.md` for the Redis-specific notes and
the `vm.overcommit_memory=1` sysctl fix.

### Shared config

Create the standard GEECS config file for the service account:

```bash
sudo -u geecs sh -c 'mkdir -p "$HOME/.config/geecs_python_api"'
sudo -u geecs sh -c 'install -m 600 /dev/null "$HOME/.config/geecs_python_api/config.ini"'
sudo -u geecs sh -c 'editor "$HOME/.config/geecs_python_api/config.ini"'
```

Use the [Getting Started config.ini section](../../../docs/tutorials/getting_started.md)
as the key-by-key reference. Do not invent a queueserver-specific config
format; this worker reads the same `~/.config/geecs_python_api/config.ini`
contract as the rest of GEECS-Plugins.

At minimum, verify that the service account's config resolves:

- the GEECS data root on the mounted data share,
- the scanner configs repository,
- the scan-analysis configs path (`[Paths] scan_analysis_configs_path`) —
  required by optimize-mode requests whose evaluator uses `analyzers`
  (`BaseOptimizerConfig` refuses those without it; analyzer-free optimize
  requests and every other mode run fine, so the gap surfaces only on the
  first analyzer-based optimize submission),
- database credentials through the normal `Configurations.INI` chain,
- Tiled connection details when Tiled publishing is enabled,
- optional `[epics] ca_addr_list` if the unit-level `EPICS_CA_ADDR_LIST`
  override is not being used.

A missing key does not fail the service at startup — the worker loads and
the plan list populates, with only a warning in the journal (e.g. "Could
not determine base data path", "No Tiled URI configured — Tiled storage
disabled"). Grep the journal for `WARNING`/`ERROR` after the first
`environment open` rather than trusting a clean-looking `qserver status`
(2026-08-21 live-checkpoint lesson: three keys were discovered missing
one failed scan at a time).

The systemd unit sets `EPICS_CA_ADDR_LIST` explicitly. That is the deployment
standard because `systemctl cat geecs-qserver.service` then shows the active
Channel Access target without inspecting a private config file.

### Data share

Mount the production data share before starting the manager. The worker must
write to the same scan-folder tree used by the console path; otherwise scan
number claim, ScanInfo creation, native asset references, and s-file export
will fail or point at the wrong location.

Validate the mount as the service account before enabling the unit:

```bash
sudo -u geecs test -d /mnt/geecs-data
sudo -u geecs test -w /mnt/geecs-data
```

Replace `/mnt/geecs-data` with the path configured in `config.ini`.

---

## 2. Install the service unit

Edit the placeholder values in `geecs-qserver.service` for the host:

- `User=geecs` — replace only with the unprivileged service account created
  for this host.
- `WorkingDirectory=` — the `GeecsBluesky` checkout directory.
- `QS_STARTUP_DIR=` — the queueserver startup profile directory.
- `QS_EXPERIMENT=` — the GEECS experiment name served by this manager.
- `EPICS_CA_ADDR_LIST=` — the CA gateway host, for example `192.168.6.14`.
- `ExecStart=` — the service user's **absolute poetry path** (locate with
  `command -v poetry` as that user) followed by
  `run <checkout>/GeecsBluesky/qserver/launch_re_manager.sh`.

Install and start:

```bash
sudo cp geecs-qserver.service /etc/systemd/system/geecs-qserver.service
sudo systemctl daemon-reload
sudo systemctl enable --now geecs-qserver.service
```

Check the service and journal:

```bash
systemctl status geecs-qserver.service
journalctl -u geecs-qserver.service -n 100 --no-pager
```

The unit intentionally carries a commented-out
`Environment=QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER=` line. CurveZMQ control-plane
key management is still open; the design item is tracked in
`../../../Planning/cutover_strategy/02_queueserver_migration.md`.

### Document stream (no extra unit)

The launcher starts a `bluesky-0MQ-proxy` (5567 in / 5568 out) alongside
Redis when one is not already answering; the startup profile publishes
bluesky documents into it for GUI progress (see `../README.md`, "Document
stream"). The proxy runs inside the unit's cgroup on purpose — it is
stateless and dies with the unit; no second systemd unit is needed. Caveat
of that choice: systemd supervises only the main process, so a proxy that
dies *alone* is not restarted until the next manager restart — the symptom
is GUI progress going quiet while scans keep running fine (the proxy's
stderr goes to the journal; the worker also logs a warning at startup when
nothing answers on the publish port). Override the ports with
`Environment="QS_DOC_PROXY_IN=…"` / `QS_DOC_PROXY_OUT=…` (paired with
`QS_DOC_PUBLISH_ADDR` for the worker side), or disable with
`QS_DOC_PROXY=OFF` + `QS_DOC_PUBLISH_ADDR=OFF`.

### Network ports

Client machines (console GUIs) need to reach, on the worker host:

- **60615** — the RE Manager control socket (`bluesky-queueserver-api`),
- **60625** — the manager's console-output stream (log tail / failed-move
  reason lines; text, not documents),
- **5568** — the document-stream out port (live per-shot progress).

Redis (6379) stays loopback-only. The proxy's in port (5567) is only
ever used by the worker itself (same host) — nothing remote needs it,
though the proxy binds all interfaces; firewall it with the rest.

---

## 3. Verify the manager

Run these commands from a shell that has the `qserver` CLI available. The
Poetry environment from the checkout is the expected source:

```bash
sudo -u geecs -i
cd /opt/geecs/GEECS-Plugins/GeecsBluesky
poetry run qserver status
```

Expected: the manager responds, Redis is reachable, and the RE Manager state
is visible.

Open the worker environment:

```bash
poetry run qserver environment open
```

Then confirm the environment reaches the opened or idle state:

```bash
poetry run qserver status
```

If `environment open` fails, inspect the service journal first:

```bash
journalctl -u geecs-qserver.service -n 200 --no-pager
```

Follow the troubleshooting section in `../README.md` for the known
queueserver failure shapes — it is the single maintained list; symptoms and
fixes are not restated here.

After the environment is open, submit only the smoke-test queue items approved
for the current startup profile. Do not use a deployment host to discover
machine-control behavior ad hoc; the manager is the production execution
surface once this service is enabled.
