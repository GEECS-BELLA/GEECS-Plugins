# GeecsArchiver — Deployment

How the Archiver Appliance is installed on a service host, how its archive
set is onboarded, and how it is operated. The plan and the pilot record are
in `PLAN.md`; the appliance's own documentation is
<https://epicsarchiver.readthedocs.io/>. Every facility value below comes
from the host's `site.env` (`docs/platform/site_profile.md`) — nothing in
this runbook is a value to type.

---

## 1. What runs

One systemd unit, `geecs-archiver`, running upstream's official container
(`ghcr.io/archiver-appliance/epicsarchiverap:singletomcat-<version>`, pinned
in `deploy/compose.yaml.in`) under `docker compose` with host networking.
The appliance is a Channel Access client of the CA gateway and nothing
else: it reads `EPICS_CA_ADDR_LIST` from `site.env` like every other CA
client and reconnects on its own through gateway restarts.

| | |
|---|---|
| Port | 17665 — management UI/API (`/mgmt`) and data retrieval (`/retrieval`) |
| Conf | `/etc/geecs/archiver/` — `compose.yaml`, `appliances.xml` (rendered), `policies.py`, `server.xml`, `context.xml`, `archappl.properties` (static) |
| Configuration store | `/var/lib/geecs-archiver/archappl.jdbm2` (the unit's `StateDirectory`; a cache — the record is §3) |
| Archive data | `GEECS_ARCHIVER_DATA_ROOT/sts` (short-term, hourly partitions) → `GEECS_ARCHIVER_DATA_ROOT/lts` (long-term, yearly); no medium-term store |
| Logs | `journalctl -u geecs-archiver` (Tomcat's stdout) |
| Health | `geecs-archiver status`; `curl http://<host>:17665/mgmt/bpl/getApplianceMetrics` |

---

## 2. Install

> **Container recipe unverified until Phase 3.** The 2026-10-02 pilot ran the
> appliance from tarballs as the service account. Everything container-specific
> below — `docker compose` as the service account, Tomcat running as a non-root
> uid inside upstream's image (its `work/`, `temp/`, `logs/` directories), the
> bind-mounted `server.xml`/`context.xml`, JDBM2 persistence through the state
> directory, a re-added PV picking up its existing partitions — is designed from
> upstream's image and documentation and **has not run yet**. The first
> production install is its test; fix the recipe there and drop this note.

### Prerequisites (root, once per host)

```bash
sudo apt install docker.io docker-compose-v2        # Engine + the compose plugin; never Docker Desktop
sudo usermod -aG docker <service account>            # the unit runs compose as the service account
sudo systemctl enable --now docker
```

Log the service account out and in (or `newgrp docker`) so the group
applies. Outbound HTTPS to `ghcr.io` is needed for the first pull and for
upgrades.

### The clone and its environment (as the service account)

The archiver gets its own clone, `<root>/archiver-checkout`, per the fleet
map's one-clone-per-service rule:

```bash
git clone https://github.com/GEECS-BELLA/GEECS-Plugins.git <root>/archiver-checkout
cd <root>/archiver-checkout/GeecsArchiver
( . ../deploy/site_env_lib.sh; load_site_env /etc/geecs/site.env; "$GEECS_POETRY" install )
```

(`deploy/bootstrap_host.sh --only archiver` does both once the archiver is
wired into it; until then, this is the install.)

### Render the conf and the unit

`site.env` needs the archiver's three keys beside the fleet's usual ones:

```ini
GEECS_ARCHIVER_HOST=<the address every client reaches the appliance at>
GEECS_ARCHIVER_DATA_ROOT=/srv/geecs-archiver
GEECS_ARCHIVER_JAVA_OPTS=-Xmx2g
```

```bash
cd <root>/archiver-checkout
GeecsArchiver/deploy/render_conf.sh /etc/geecs/site.env ~/deploy-staging/archiver
deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging GeecsArchiver/deploy/geecs-archiver.service
```

Both are unprivileged and print the root lines. In substance:

```bash
sudo install -d -m 0755 /etc/geecs/archiver
sudo install -m 0644 ~/deploy-staging/archiver/* /etc/geecs/archiver/
sudo install -d -o <service account> -g <service account> -m 0750 "$GEECS_ARCHIVER_DATA_ROOT"/sts "$GEECS_ARCHIVER_DATA_ROOT"/lts
sudo install -m 0644 ~/deploy-staging/geecs-archiver.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now geecs-archiver
```

`render_conf.sh` fills the service account's uid/gid into the compose file
(the container's Tomcat runs as that user) — render **on the service
host**, not on a laptop; it warns when the account does not exist.

### First start

```bash
journalctl -u geecs-archiver -f          # the four web apps deploy in ~1 min
curl -s http://localhost:17665/mgmt/bpl/getVersions
```

`getVersions` answering with four matching version strings is "up". The
management UI is `http://<host>:17665/mgmt/ui/index.html`.

---

## 3. Onboarding the archive set

The archive set is **derived**, not maintained: `geecs-archiver onboard`
reads the experiment from the GEECS database with the same queries the CA
gateway serves from and reconciles the appliance with the result. Run it
from any machine with database access and a `config.ini` naming the
appliance:

```ini
[archiver]
url = http://<host>:17665
```

```bash
geecs-archiver list --experiment Undulator              # what the rule wants (database only)
geecs-archiver onboard --experiment Undulator --dry-run # the plan: archive / resume / retune / pause
geecs-archiver onboard --experiment Undulator           # apply, then wait for the new PVs to connect
```

What the rule includes and excludes is `geecs_archiver.archive_set`'s
docstring; the experiment's exceptions go in the configs repo as
`scanner_configs/experiments/<Experiment>/archiver/archive_policy.yaml`
(a `geecs_schemas.ArchivePolicy`; absent = defaults). **Rolling out:**
start with the policy's `exclude` set to everything but a curated dozen,
watch a real lab week (disk growth, engine CPU, retrieval latency in
`status`), then widen.

`onboard` never deletes: a PV the rule stops wanting is **paused**, and only
PVs under this experiment's prefix are touched. Pausing more than ten PVs in
one run needs `--yes` (a half-empty device table must not pause the
experiment silently). A PV someone archived by hand from the appliance's UI
is paused on the next run unless it is listed in the policy's `include` —
the rule plus the committed policy is the configuration of record, so put
the exception there, not in the appliance.

**The drift alarm.** A requested PV the gateway does not serve never answers
on CA, so the appliance cannot complete its archive request: it stays in the
appliance's **never-connected list** (`getNeverConnectedPVs`) indefinitely.
`onboard` reads that list on **every** run — not only after a submission —
prints each wanted PV found there as `! never connected`, and exits 1. Fix
the rule or the gateway, not the appliance. Exit status: 0 ok · 1 drift
(or a refused request) · 2 usage (no experiment/URL, or an unguarded mass
pause) · 3 the appliance did not answer.

**After a GEECS database change** (a device added, a `get` flag edited):
restart the gateway (its restart PV) and rerun `onboard`. Two commands, one
change. Rerunning when nothing changed is a no-op.

---

## 4. Operations

- **Health.** `geecs-archiver status` (versions, PV counts, event and data
  rates, this experiment's disconnected PVs); the management UI; the unit's
  journal. The appliance's disconnected-PV list is the archiver-side
  mirror of the gateway's `connected` PVs.
- **Gateway restart.** Normal. The appliance logs the disconnect, reconnects
  by itself and records the outage as connection-lost / regained fields on
  the first sample after (pilot: a 14 s gap). No action.
- **Disk.** The one metric that pages. Watch `GEECS_ARCHIVER_DATA_ROOT`
  (yearly files under `lts/`); the ETL's out-of-space default *deletes
  source streams*, so alert well before. Sizing: `PLAN.md` §5.
- **Configuration backup.** The store under `/var/lib/geecs-archiver` is a
  cache; the record is the rule plus the committed policy (its `include`
  list is where a hand-added PV belongs). `geecs-archiver export-config
  --out <path>` snapshots the appliance's own view (JSON; `importConfig`
  restores it) for a like-for-like restore after a disk loss. A nightly systemd timer writing into
  `{experiment}/archiver/` on the data share is the intended shape.
- **Upgrade.** Bump the tag in `deploy/compose.yaml.in` in a reviewed PR,
  pull the clone, re-render, reinstall `/etc/geecs/archiver`, then
  `sudo systemctl restart geecs-archiver` (compose pulls the new image on
  the way up). Read upstream's release notes for persistence-format or
  Tomcat changes first. Weekly upstream images are never deployed.
- **Stop / start.** `sudo systemctl stop geecs-archiver` runs `compose
  down` (Tomcat flushes the short-term store on shutdown; the unit allows
  180 s).

---

## 5. Reading the history

- **Phoebus Data Browser**, in the facility's `settings.ini`:
  ```ini
  org.csstudio.trends.databrowser3/urls=pbraw\://<host>:17665/retrieval|GEECS archiver
  org.csstudio.trends.databrowser3/archives=pbraw\://<host>:17665/retrieval|GEECS archiver
  ```
  Restart Phoebus; right-click any live PV → Data Browser.
- **Browser:** `http://<host>:17665/retrieval/ui/viewer/archViewer.html`.
- **Python:** `GET /retrieval/data/getData.json?pv=<pv>&from=<ISO>&to=<ISO>`
  → `[{"meta": {...}, "data": [{"secs", "nanos", "val", "severity", "status"}, …]}]`.
  Stored timestamps are the gateway's GEECS timestamps, not receive time.

---

## 6. Troubleshooting

| Symptom | Meaning |
|---|---|
| `onboard` exits 1 with `! never connected` PVs | The rule derives a PV the gateway does not serve (the appliance's never-connected list). Compare with the gateway's log at startup; the served-set rule changed on one side |
| `onboard` exits 2, "refusing to pause N PVs" | The derived set shrank by more than the guard — usually a database maintenance state. Check `geecs-archiver list`; rerun with `--yes` only if the shrink is real |
| `onboard` exits 3 | The appliance did not answer (`geecs-archiver status`; the unit's journal) |
| Engine log: `Cannot determine DBR type for pv …localsavingpath` | A `path` (char-array) PV reached the appliance. The rule excludes them; something submitted it by hand — pause it |
| Appliance up, every PV disconnected | `EPICS_CA_ADDR_LIST` in `site.env` does not name the gateway, or the gateway is down (`caget <exp>:cagateway:heartbeat` from the host) |
| `docker compose up` fails on `17665` in use | A hand-started pilot (`pkill -f quickstart_tomcat`) or another service on the port |
| Permission denied writing `storage/…` or `persistence/…` | The data root or state directory is not owned by the service account, or the compose `user:` ids are stale — re-render on the host |
| `onboard`: `no appliance URL` | Set `config.ini [archiver] url` (or `GEECS_ARCHIVER_URL`) |
| Startup samples dated days ago | By design: a PV's first stored sample carries its own last-change timestamp, flagged `startup` |

---

## 7. Second facility

Copy `deploy/site.env.example`, set every value (the archiver keys
included), run the install above, then `geecs-archiver onboard
--experiment <theirs>`. One appliance per facility; the per-experiment PV
prefix keeps the archives disjoint, and a Data Browser that reaches both
networks can list both `pbraw://` URLs.
