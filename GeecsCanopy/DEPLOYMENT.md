# GeecsCanopy — deployment runbook

One instance per facility, on the services host (Linux, x86: the image is
amd64 only). Everything below runs from the host's canopy clone
(`<GEECS_CHECKOUT_ROOT>/canopy-checkout`); `dc` abbreviates
`sudo -u <service account> docker compose --env-file /etc/geecs/site.env --project-directory /etc/geecs/canopy --project-name geecs-canopy`:
the account the unit runs as, since compose reads the 0600 `secrets.env`
itself, and the env file feeds the compose file's `${VAR}`s as the unit's
`EnvironmentFile=` does.

## 1. What runs

| Container | Listens | Notes |
|---|---|---|
| `api` | `GEECS_CANOPY_BIND:GEECS_CANOPY_PORT` | `uvicorn geecs_canopy.app:app --root-path $GEECS_CANOPY_ROOT_PATH`: Canopy with the GEECS device module at `/geecs` (section 3a); Raccoon loads `geecs_canopy.directory:GeecsDirectoryAdapter` |
| `worker` ×2 | — | `arq canopy.worker.WorkerSettings`: Squirrel's snapshot jobs; no worker, no snapshots |
| `postgres` 16 | 127.0.0.1:5433 | data in `/var/lib/geecs-canopy/pgdata` (the unit's state dir) |
| `redis` 7 | 127.0.0.1:6380 | the arq queue only, no persistence; never the queueserver's Redis |

All four use host networking (CA needs UDP). Postgres and Redis live in this
project, as the archiver keeps its own store: Canopy's state is its own, and
its upgrade runbook below dumps and migrates one database nobody else uses.
The front door forwards `/canopy/` to the api; the WebSocket is
`/canopy/hare/v1/ws/pvs` (the proxy must pass `Upgrade`). CORS is `*` in
Canopy itself.

## 2. First install

```bash
GeecsCanopy/deploy/render_conf.sh /etc/geecs/site.env ~/deploy-staging/canopy
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging GeecsCanopy/deploy/geecs-canopy.service
# root: the lines render_conf.sh printed (the conf, secrets.env, the state and backup
# folders owned by the service account; postgres runs as it), then fill secrets.env
# (each value: openssl rand -hex 32)
sudo install -m 0644 ~/deploy-staging/geecs-canopy.service /etc/systemd/system/ && sudo systemctl daemon-reload
dc build                                   # the derived image, FROM pin.py's digest
dc run --rm api alembic upgrade head       # creates Canopy's schema (postgres starts first)
sudo systemctl enable --now geecs-canopy
```

A configs share that mounts after the containers start stays invisible to
them (500 `adapter_misconfigured` on unscoped queries): restart the unit
after remounting it. Check: `curl -s -H "X-API-Key: <display key>" http://<bind>:<port>/hare/v1/health/summary`
answers `"backend": "epics"`. The default backend is `simulation`, which
serves sine waves and still passes `/health`: the compose file sets
`CANOPY_EPICS__BACKEND=epics`, and this check is part of every deploy.

## 3. Keys

Two env-layer keys in `secrets.env` (`CANOPY_API_KEYS`): the **display**
key (`read@*`, every page and front end) and the **ops** key (`admin@*`).
Every other key is minted into Postgres, printed once:

```bash
dc run --rm -T api python scripts/create_key.py "<app>" -r        # add -w only for a service key
```

**Rule (condition (a)): no operator or front-end key ever carries Canopy
`write`.** A key's `write` is key-wide: it reaches Hare's `PUT` and
Squirrel's restore directly, around every guard of the GEECS module. Write
keys are service keys only (the module itself, an ops script), never on a
page, never handed to a person. Revocation is not instant (keys are cached).

## 3a. The device module (`/geecs`)

The api command is the wrapper `geecs_canopy.app:app`: Canopy's app,
unpatched, with the module mounted in-process at `/geecs`. Its devices come
from data only: the family files (GEECS-Schemas) and the experiment's
`native_devices.yaml` + `panel_devices.yaml`, joined by GEECS device name
as the directory joins them; never the GEECS DB. Only `@*` keys reach
`/geecs` (Canopy's module set is fixed).

| Route (read access) | Answers |
|---|---|
| `GET /geecs/v1/devices` | the roster, each device's family, axes and state |
| `GET /geecs/v1/devices/{name}` | every role per axis: PVs, Hare's value, severity, a verdict (`live`, `stale`, `off`, `not answering`, `unreachable`, `error`, `unknown`; `not running` waits for a service-status source) |
| `POST /geecs/v1/devices/{name}/put` | `{role, axis?, value, operator, confirm?}` through the gates (`geecs_canopy/gates.py`): policy 403, allow list 403, hazard 428, run lock 423, bounds 422, then Hare (its own status on refusal); an `attempted` journal row before Hare (503 `journal_unavailable`, nothing sent, when it cannot be written) and one outcome row per request |
| `GET /geecs/v1/journal?entries=N` | the newest N `JournalEntry` rows (GEECS-Schemas), oldest first |

Freshness is judged continuously: the module subscribes every monitored
role and each `ioc`'s heartbeat as an internal Hare client and keeps when
each counter last advanced.

**Settings** (compose environment): `GEECS_CANOPY_EXPERIMENT`,
`GEECS_CANOPY_CONFIGS_ROOT` (the configs repo's root, mounted
`/opt/geecs-configs`), `GEECS_CANOPY_STATE_DIR` (the journal's folder,
`/var/lib/geecs-canopy/module` on the host, owned by the image's `appuser`,
uid 1000, group the service account); optional `GEECS_CANOPY_STALE_AFTER_S` (3), `_PUT_TIMEOUT_S`
(5), `_SETTLE_TIMEOUT_S` (30: a move's put-callback, not Hare's 5 s
connect timeout), `_RELOAD_S` (10: the device documents' re-read).

**Who may write: `<experiment>/operator_policy.yaml`** in the configs repo
(`OperatorPolicy`, GEECS-Schemas). No file, no grant: every put is denied.

```yaml
schema_version: 1
grants:
  - operators: [A. Operator, B. Operator]
    families: [motor_record]          # every motor, within each allow list
  - operators: [A. Operator]
    devices: [U_ChicaneInner]
    roles: [current_sp, reset]        # empty: every role the device allows
```

**The policy guards against mistakes, not against users.** The operator
is the name the request carries, and nothing binds it to a person: anyone
holding the display key can name a granted operator. The authenticated
part is the Canopy key, journaled by name. What does hold for every
caller is the allow list, the hazard confirm, the run lock and the bounds,
and that no page key reaches Hare's or Squirrel's own write routes.
Binding the operator to a person needs a key per operator or the front
door's user header (an upstream candidate, §7.4 of the decision page).
Operator and page keys are `read` only (section 3): the module authorizes
its own writes and reaches Hare in-process, so a person never needs
Canopy `write`.

**The journal** is SQLite (`journal.sqlite3` in the state folder), one
`JournalEntry` outcome row per request: the only record of who wrote what,
so it has its own backup. A put that passed every gate first leaves an
`attempted` row, and no put reaches Hare unless that row was written (a
read-only, full or locked journal refuses writes with 503
`journal_unavailable`). Its outcome row names it in `attempt`. **An
`attempted` row no outcome row names is a write that may have reached the
device with no recorded answer**: the module died, or the journal failed,
during the put (the put's response then carries `journalError`). Read the
device's value before repeating it. The Postgres dump does not include
the journal. Nightly, beside
that dump, from the service account's crontab (the folder's group is the
service account's; the file is created group-readable):
`dc exec -T api python -c "import sqlite3; sqlite3.connect('/var/lib/geecs-canopy-module/journal.sqlite3').backup(sqlite3.connect('/var/lib/geecs-canopy-module/journal.bak'))"`,
then `cp /var/lib/geecs-canopy/module/journal.bak /var/backups/geecs-canopy/journal-$(date +%F).sqlite3`. The
Postgres path, a `geecs` schema with its own Alembic tree and version
table (Canopy's tree is one for all its modules), is not built.

**Squirrel.** Its registry is **derived, never hand-edited**: push 2
reconciles it idempotently from the family files and the device documents
(the GeecsArchiver pattern). Squirrel's restore job is **never the executor
of an operator's restore**: its worker writes with its own EPICS adapter,
around every guard and unattributed. A restore is a module route (push 2)
that reads the saved values from Squirrel and runs each row through
`gates.put_role`, the same function the put route calls (lock, bounds,
confirm and journal checked at execution). Squirrel's restore endpoint
stays reachable to service keys only.

**The self-test.** `geecs_canopy/canopy_surface.py` lists every
Canopy-internal name the package uses and the image they were verified
at. The wrapper checks them, the pin and the installed commit before
importing Canopy, and refuses to start naming the rollback (section 5).

## 4. The GEECS DB (transitional)

The directory's gateway source (`elementType=gateway`) lists what the CA
gateway serves for the LabVIEW devices, with the gateway's own read-only
queries. It goes away with the CA gateway. It connects as a dedicated
read-only user, `GEECS_CANOPY_DB_URL` in `secrets.env`; this package never
writes the GEECS DB, so the maintainer creates the user, for example:

```sql
CREATE USER 'canopy_ro'@'<services-host>' IDENTIFIED BY '<password>';
GRANT SELECT ON <database>.expt_device          TO 'canopy_ro'@'<services-host>';
GRANT SELECT ON <database>.expt_device_variable TO 'canopy_ro'@'<services-host>';
GRANT SELECT ON <database>.device               TO 'canopy_ro'@'<services-host>';
GRANT SELECT ON <database>.devicetype_variable  TO 'canopy_ro'@'<services-host>';
GRANT SELECT ON <database>.variable             TO 'canopy_ro'@'<services-host>';
GRANT SELECT ON <database>.choice               TO 'canopy_ro'@'<services-host>';
```

A site without the DB sets `GEECS_CANOPY_DB=off`: unscoped queries list the
native devices, `elementType=gateway` answers 503 `directory_unavailable`.

## 5. Upgrade and rollback

The pin is `geecs_canopy/pin.py` (digest + commit), bumped in a reviewed PR
after the same steps on the sim. An older image cannot run on a newer
schema, so every bump is:

```bash
dc run --rm api alembic current                     # note the revision: the rollback target
dc exec -T postgres sh -c 'pg_dump -p 5433 -U "$POSTGRES_USER" -Fc "$POSTGRES_DB"' \
  | sudo -u <service account> tee /var/backups/geecs-canopy/pre-<new-commit>.dump >/dev/null
git pull && GeecsCanopy/deploy/render_conf.sh /etc/geecs/site.env ~/deploy-staging/canopy   # + install
dc build && dc run --rm api alembic upgrade head
sudo systemctl restart geecs-canopy                 # then the backend check of section 2
```

Rollback, rehearsed on the sim before the bump: with the **newer** image
still built, `dc run --rm api alembic downgrade <noted revision>` (every
migration at `0a5ee23` has a real `downgrade()`), then check out the
previous pin, re-render, `dc build`, restart. If a downgrade fails, stop the
unit and restore the dump (`pg_restore --clean -p 5433 -U canopy -d canopy`).
The wrapper app is the api command, so its self-test failing takes Hare,
Squirrel and Mole down with it: the rollback is then the unwrapped
`uvicorn canopy.main:app` command (edit the api `command:` in the
installed compose file) plus the previous pin. A pin bump re-verifies
`canopy_surface.py`'s names against the new commit and moves its
`VERIFIED_IMAGE` in the same PR.

**Routine backup:** the same `pg_dump` nightly from the service account's
own crontab (it owns the backup folder; no `sudo` there), keeping a week. Postgres holds Squirrel's registry and
snapshots and the minted keys: state that exists nowhere else.

## 6. Development

- **Canopy's contract suite**, inside the derived image (the image ships
  no `tests/` tree and no pytest):
  `git clone https://github.com/slaclab/canopy && git -C canopy checkout <CANOPY_COMMIT>`, then
  `docker run --rm --platform linux/amd64 -u root -v $PWD/GeecsCanopy:/pkg:ro -v $PWD/canopy:/canopy:ro -e CANOPY_SOURCE=/canopy -e PYTHONDONTWRITEBYTECODE=1 -w /pkg geecs-canopy:<commit> sh -c 'pip --python /app/.venv/bin/python install -q pytest pytest-asyncio && /app/.venv/bin/python -m pytest -p no:cacheprovider tests/test_directory.py -q'`.
- **Mac bench variant** (Docker Desktop has no Linux host networking): a
  bridge network, `platform: linux/amd64`, published loopback ports, and
  `EPICS_CA_ADDR_LIST` naming each sim IOC as `host.docker.internal:<port>`
  (`EPICS_PVA_NAME_SERVERS` likewise). The exercise's runtime directory is
  that variant; it is a dev deviation, never a deployment.
