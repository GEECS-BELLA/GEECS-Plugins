# Deploying the gateway as a service

`geecs-ca-gateway.service` is the systemd unit for the lab deployment
(serving on `192.168.6.14`). It is written against a generic `geecs` service
account — substitute the real account/paths on your box in the installed copy
(`User=`, `Environment=HOME=`, `WorkingDirectory=`, `ExecStart=`); site
specifics belong in `/etc/systemd/system`, not in this repo. The full
deployment context — config resolution, network scoping, the client-side
recipe, smoke tests, log expectations — is in
[`DEPLOYMENT.md`](../DEPLOYMENT.md).

## Install

From the repo checkout on the target box:

```bash
cd ~/GEECS-Plugins/GeecsCAGateway
poetry install                    # slim env: caproto + pydantic + mysql-connector

# Sanity-check the unit's paths against this box before installing:
which poetry                      # must match ExecStart's poetry path
poetry run python -m geecs_ca_gateway --experiment Undulator --log-level INFO
# ^ run once in the foreground; Ctrl-C once PVs are serving cleanly

sudo cp deploy/geecs-ca-gateway.service /etc/systemd/system/
sudoedit /etc/systemd/system/geecs-ca-gateway.service   # User= + the three paths
sudo systemctl daemon-reload
sudo systemctl enable --now geecs-ca-gateway
```

## Verify

```bash
systemctl status geecs-ca-gateway
journalctl -u geecs-ca-gateway -f     # expect the startup lines, then quiet
```

Then the smoke tests from `DEPLOYMENT.md` §4 — from a *different* machine, so
subnet scoping and firewalls are exercised:

```bash
caproto-get Undulator:CAGateway:VERSION Undulator:CAGateway:DEVICES_CONNECTED
caproto-monitor Undulator:CAGateway:HEARTBEAT    # ticks every 5 s
```

## Upgrade / resync with the GEECS DB

A restart is both the upgrade and the DB-resync mechanism (the served set
rebuilds from the database at startup):

```bash
cd ~/GEECS-Plugins && git pull
cd GeecsCAGateway && poetry install
sudo systemctl restart geecs-ca-gateway
```

**After a DB edit, no shell needed**: write the restart PV from any CA client
on the lab subnet (the unit's `RestartForceExitStatus=86` relaunches the
service, which rebuilds its config from the DB):

```bash
caproto-put Undulator:CAGateway:RESTART Restart
```

## CA alarm limits table

The curated alarm overlay lives in the existing GEECS MySQL database as
`ca_alarm_limits`. Apply the migration once per database before entering alarm
rows:

```bash
cd ~/GEECS-Plugins/GeecsCAGateway
mysql --host=<db-host> --user=<db-user> --password <db-name> \
  < deploy/ca_alarm_limits.sql
```

The table is optional from the gateway's perspective: if it is absent, startup
continues with no curated value alarms. After editing rows in `ca_alarm_limits`,
restart through the service or `Undulator:CAGateway:RESTART` so the gateway
reloads the DB-backed config.

To smoke-test one configured row, drive the readback into one of its alarm
bands and read the PV with status metadata:

```bash
caproto-get -a Undulator:U_S1H:Current
```

For example, with `high=2.5`, `hihi=4.5`, and the default high severity of
`MINOR`, a live `U_S1H:Current` value of `3 A` should report `HIGH` /
`MINOR_ALARM` in CA clients such as Phoebus. If you edit the DB row and do not
see the metadata or severity change, restart the gateway so it reloads the
DB-backed config.

## Changing what is served

Edit `ExecStart` in the unit (`--all-variables`, `--no-settable`,
`--include-disabled`, an explicit `--derived-channels /path/to/file.yaml`
override, a different `--experiment`), then
`sudo systemctl daemon-reload && sudo systemctl restart geecs-ca-gateway`.
Serving a second experiment = a second copy of the unit with a different name
and `--experiment`; CA name resolution makes the split invisible to clients.

Derived channels normally do not require an `ExecStart` edit: put
`derived_channels.yaml` under
`GEECS-Plugins-Configs/scanner_configs/experiments/<Experiment>/gateway/` and
restart the gateway. The service discovers that file through
`GEECS_SCANNER_CONFIG_DIR`, `GEECS_PLUGINS_CONFIGS`, or
`config.ini [Paths] scanner_config_root_path`.
