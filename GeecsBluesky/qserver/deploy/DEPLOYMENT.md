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

Install Poetry by the method approved for the host image, then install the
worker environment from the repository checkout:

```bash
cd /opt/geecs/GEECS-Plugins/GeecsBluesky
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
- database credentials through the normal `Configurations.INI` chain,
- Tiled connection details when Tiled publishing is enabled,
- optional `[epics] ca_addr_list` if the unit-level `EPICS_CA_ADDR_LIST`
  override is not being used.

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
- `ExecStart=` — the checked-out `qserver/launch_re_manager.sh`.

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

---

## 3. Verify the manager

Run these commands from a shell that has the `qserver` CLI available. The
Poetry environment from the checkout is the expected source:

```bash
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

Follow the troubleshooting notes in `../README.md` for the known queueserver
failure shapes:

- missing or incomplete `user_group_permissions.yaml`,
- a startup profile that does not define `RE = RunEngine({})` while the
  launcher uses `--keep-re`,
- attempts to run `qserver function execute` while the manager is not idle.

After the environment is open, submit only the smoke-test queue items approved
for the current startup profile. Do not use a deployment host to discover
machine-control behavior ad hoc; the manager is the production execution
surface once this service is enabled.
