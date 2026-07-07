# Deploying the gateway as a service

`geecs-ca-gateway.service` is the systemd unit for the current lab deployment
(user `abmx`, checkout at `/home/abmx/GEECS-Plugins`, serving on
`192.168.6.14`). The full deployment context — config resolution, network
scoping, the client-side recipe, smoke tests, log expectations — is in
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

## Changing what is served

Edit `ExecStart` in the unit (`--all-variables`, `--no-settable`,
`--include-disabled`, a different `--experiment`), then
`sudo systemctl daemon-reload && sudo systemctl restart geecs-ca-gateway`.
Serving a second experiment = a second copy of the unit with a different name
and `--experiment`; CA name resolution makes the split invisible to clients.
