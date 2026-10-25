# Deploying the PV bridge (`geecs-panels`)

The bridge is GeecsPanels' process: the per-kind pages and the `/api`
routes the web repository's pages read (device streams, puts, the
journal). It runs on the services host behind the front door, which
serves it at `/panels` (port 8500 on `GEECS_SERVICES_HOST`).

Not wired into `deploy/bootstrap_host.sh` yet: the steps below are by hand.

## Its own account

The bridge writes to devices, so it runs as **its own service account**
(`GEECS_PANELS_USER` in `site.env`), never `GEECS_SERVICE_USER`. The camera
IOCs' access security lets the scan worker's CA user write a camera that a
run holds. Every other unit runs as that one account, so a bridge on it
would pass the run lock (`Planning/device_panels.md` § 6). The IOCs see
this account as the CA user of every put the bridge makes. Who asked is
the bridge's journal: one row per put, naming the operator.

The account has no login, no home and no `config.ini`. Everything it needs
comes from `site.env` (the experiment, the CA address list, the configs
root) and its command line.

```
sudo useradd --system --no-create-home --shell /usr/sbin/nologin \
    --gid "$GEECS_SERVICE_USER" "$GEECS_PANELS_USER"
```

The unit adds the service account's group (`SupplementaryGroups=`), so the
bridge can read the clone and its venv. The clone's parent directories
must let that group in (`chmod g+rx`, e.g. on a 0750 home).

## Install

As the service account:

```
git clone "$GEECS_REPO_URL" "$GEECS_CHECKOUT_ROOT/panels-checkout"
cd "$GEECS_CHECKOUT_ROOT/panels-checkout/GeecsPanels"
"$GEECS_POETRY" config virtualenvs.in-project true --local
"$GEECS_POETRY" install
```

The venv must live inside the package (`.venv`). The unit runs
`.venv/bin/geecs-panels` directly, because the bridge's account has no
poetry home to find a venv in.

Then render and install the unit (root), from any clone:

```
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
sudo install -m 0644 ~/deploy-staging/geecs-panels.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now geecs-panels
```

A `site.env` without `GEECS_PANELS_USER` renders no panels unit.

## The IOCs' side

Each IOC's access file must grant this account where the bridge writes
(geecs-epics `access.acf`). The camera's acquisition group stays the
worker's while a run holds the lock. The bridge refuses those puts first
(423), and the IOC refuses them too.

## State

The put journal is `/var/lib/geecs-panels/journal.db`, the unit's
`StateDirectory`; `--journal PATH` keeps it elsewhere. Back it up: who
wrote what to which device is recorded there and nowhere else.

## Check

- `GET /health`
- `GET /api/devices`: the devices, and `errors` naming any it could not read.
- `GET /api/journal?entries=10`
- Through the front door: `/panels/health`, `/panels/api/devices`.

`journalctl -u geecs-panels` shows every run-lock release, with the
operator's name.
