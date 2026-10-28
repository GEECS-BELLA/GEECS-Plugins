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
| `api` | `GEECS_CANOPY_BIND:GEECS_CANOPY_PORT` | `uvicorn canopy.main:app --root-path $GEECS_CANOPY_ROOT_PATH`; Raccoon loads `geecs_canopy.directory:GeecsDirectoryAdapter` |
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
Once lane B's wrapper app is the api command, its self-test failing takes
Hare, Squirrel and Mole down with it: the rollback is then the unwrapped
`uvicorn canopy.main:app` command plus the previous pin.

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
