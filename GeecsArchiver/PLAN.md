# GeecsArchiver — EPICS Archiver Appliance plan

Continuous, between-scan history of GEECS control variables: the niche
the legacy NI SVE → Citadel path fills today and the one piece of the
EPICS migration not yet stood up. Bluesky/Tiled records *per-run* scan
data only — "what was the chiller doing last Tuesday at 03:00" has no
answer in our stack until this lands.

**Status (2026-10-02): a plan awaiting the maintainer's rulings (§11),
not a deployment record.** It supersedes the July 2026 draft (the gateway package's ARCHIVER.md,
now this file's git history); §1 lists what
changed and §10 what still stands. When the phases execute, the
operational content moves into this package's `DEPLOYMENT.md` and this
file is deleted — plans do not outlive their landing (the 2026-09
`Planning/` prune is the precedent).

Companion contracts: `GeecsCAGateway/PV_CONTRACT.md` (what the appliance,
as a CA client, may rely on), `docs/platform/site_profile.md` (one home
per facility value — the rule this package is the first service to join
*by design* rather than by retrofit), `docs/platform/fleet_map.md` (where
the new row goes).

---

## 0. What we are deploying

The [EPICS Archiver Appliance](https://github.com/archiver-appliance/epicsarchiverap)
is the accelerator community's standard PV archiver (SLAC/BNL/ALS
lineage; current stable **2.4.1, July 2026**). Four Java web apps —
`mgmt`, `engine`, `etl`, `retrieval` — in Apache Tomcat. Nobody writes
Java: configuration is an XML appliance identity, a Python-syntax
`policies.py`, a properties file and environment variables, plus a REST
API ("BPL") and a web UI for PV management.

It is a plain Channel Access client. The gateway already serves each
experiment's monitoring set with GEECS timestamps, correct types,
units/precision, and INVALID/COMM alarm transitions on device
disconnect — exactly the interface the archiver was built to consume.
**No GEECS-specific code sits in the archive path.** The only new code in
this repository is a small Python package (§2): the PV-onboarding tool,
a typed client of the mgmt API, a retrieval helper, and the deploy
recipe.

### Non-goals

- **Images and arrays.** Off CA by design; the PVA gateways own them.
  Scalars only.
- **Per-shot correlation.** Tiled's job. The archiver stores independent
  time series.
- **Replacing s-files or Tiled.** This adds the continuous record; the
  scan-data path is untouched. GeecsBluesky's rule stands: the archiver
  is never a source of s-file columns and is never read in the scan path.
- **Cross-facility archiving.** One appliance per facility (§6); no
  archiver ever reaches across networks.
- **Retiring Citadel on day one.** It runs in parallel until the
  appliance has archived through real lab weeks, device power-cycles and
  gateway restarts included.

---

## 1. What changed since the July draft

| July 2026 draft | Now |
|---|---|
| Appliance 2.3.1; JDK 21; Tomcat 10/11 | **2.4.1** (2026-07-21): JCA update fixing PVs that showed "archiving" but were never connected (sharpens our pilot criterion, §8); 2.4.0 added S3/tar object storage and ADEL/NELM for PVXS servers; the 2.4 line documents a Parquet file backend. Build toolchain is Java 21; upstream says the *next* release needs Java 25 — irrelevant to a pulled image, relevant to a native install |
| "No official container image → native Tomcat + systemd" | **Official images exist**: the release workflow publishes `ghcr.io/archiver-appliance/epicsarchiverap:<target>-<tag>` for `singletomcat`, `mgmt`, `etl`, `engine`, `retrieval` (linux/amd64 + arm64; Tomcat 11 + Temurin JDK 21 baked), plus weekly builds. The native decision's premise is gone (§3) |
| New ABMX box sized for gateway + archiver + Tiled (§10 of the draft) | Superseded by the consolidated services box the maintainer chose in 2026-08/09 (128 GB, Ubuntu Server 24.04; archiver + queueserver + MCP + Tiled + analysis; the central lab server keeps the CA gateway and MySQL). This plan chooses no hardware; it states storage needs (§5) |
| Config persistence in the GEECS MySQL (own `archappl` schema) | The appliance's config is **reconstructible** from this package's onboarding rule, so persistence need only survive restarts → a local file (§4). The LabVIEW-critical DB box is not touched |
| Bespoke deploy on one box | The **site profile** exists (`site.env`, `render_units.sh`, `bootstrap_host.sh`, `/fleet-status`, 2026-09): every facility value this service needs gets a key there, and a second facility is a copied `site.env` (§6) |
| "A future archiver pulling from both CA servers" (gateway `DEPLOYMENT.md` §5) | The facilities sit on **different networks**: one appliance per facility, consolidation only at the viewer (§6). Amend that sentence when this lands |
| Tiled on SQLite | Postgres is coming to the worker for Tiled (#1020 arc). Irrelevant to the appliance — its JDBC persistence speaks MySQL/MariaDB (or SQLite with an extra jar), never Postgres — which is one more reason not to tie its config to a database at all |

---

## 2. Shape in the repository — **RULING 1: a new package, `GeecsArchiver/`**

Recommended: a new top-level package, not a module in `GeecsCAGateway/`
and not a bare recipe under `deploy/`.

- **It is a client of the gateway's contract.** The gateway's rule is
  that nothing imports its code; every consumer reads `PV_CONTRACT.md`
  and talks CA. The archiver's Python depends on **GEECS-Core** (the DB
  and `pv_naming`) and **GEECS-Schemas** (the overlay model, §7) — never
  on the gateway package.
- **It is a service family**, with its own clone, unit, runbook and
  fleet-map row — the one-clone-per-service rule gives it
  `<root>/archiver-checkout`.
- **It is small** — GEECS-LogTriage-sized: a few modules, a CLI, a
  `deploy/` directory, tests that need no hardware.

```
GeecsArchiver/
  pyproject.toml            geecs-archiver — python >=3.11,<3.12; geecs-core, geecs-schemas, pydantic, httpx
  README.md  CLAUDE.md  CHANGELOG.md
  PLAN.md                   this file (deleted when DEPLOYMENT.md carries the record)
  DEPLOYMENT.md             the runbook; grows through Phases 2–4
  geecs_archiver/
    config.py               the client-side [archiver] url key and the mgmt / retrieval URL derivation
    archive_set.py          the rule: which PVs an experiment archives (GeecsDb + pv_naming + the overlay)
    mgmt_client.py          typed client of the mgmt BPL: getAllPVs, getPVStatus, archivePV,
                            pauseArchivingPV, resumeArchivingPV, changeArchivalParams, exportConfig,
                            getApplianceMetrics
    onboard.py  cli.py      geecs-archiver onboard | status | export-config   (§7)
    retrieval.py            fetch(pv, start, end) -> DataFrame over getData.json   (Phase 5)
  deploy/
    geecs-archiver.service  unit TEMPLATE (@PLACEHOLDER@ + ${VAR}) wrapping the container (§3)
    compose.yaml.in  appliances.xml.in  server.xml  policies.py  archappl.properties  context.xml
    render_conf.sh          renders the .in files from site.env; the unit goes through render_units.sh
  tests/
```

Bookkeeping the repo requires of a new package: the root `CLAUDE.md`
repository map and dependency graph, the `CHANGELOG.md` list
(`scripts/doc_audit.py` checks both), `scripts/check.sh` `OWN_ENV_PKGS`
and the CI matrix leg, `deploy/README.md`'s template list.

**Alternatives considered.** (a) A module in the gateway package — wrong
direction of dependency and it drags caproto into a list-building tool.
(b) A recipe under `deploy/` only — there is real Python here (the
onboarding rule, the mgmt client) that needs a home with tests.

---

## 3. Runtime — **RULING 2: the official container, under systemd**

| | Native (JDK + Tomcat tarball + four WARs) | Official image (`singletomcat`) |
|---|---|---|
| Matches the fleet's habits | Yes — bare systemd like every other unit | A systemd unit that runs `docker compose`; the fleet's first container |
| Upstream support | Documented, but Tomcat 11 is not in Ubuntu 24.04's packages (noble ships tomcat10): tarball + hand-rolled layout + `deployMultipleTomcats.py` or the single-JVM flag | The documented, CI-built artifact; Tomcat 11 + JDK 21 pinned by upstream; our only file is the compose + conf |
| JVM/Tomcat ops burden (the draft's main fear) | Ours: JDK upgrades, Tomcat CVEs, connector/pool config, the MySQL connector jar | Upstream's: an upgrade is a tag bump |
| Upgrade | Swap WARs; re-check Tomcat/JDK floors (next release: Java 25) | `docker compose pull` after a reviewed PR bumps the tag |
| Isolation | `MemoryMax=` on the unit | cgroup limits on the container + the unit |
| New runtime on the box | None | Docker Engine (`docker.io` + `docker-compose-v2` from Ubuntu packages — never Docker Desktop) |

The July decision for native rested on "no official image, so Docker
means owning a Dockerfile". That premise is gone, and what remains of
the native path is exactly the unfamiliar operational surface the draft
wanted to avoid. **Recommendation: the official image, pinned by tag in
the repo's compose template** (the image tag is a fleet pin like a
package version, not a site value).

Specifics the pilot (§8, Phase 1) verifies:

- **`network_mode: host`.** The CA client inside uses `EPICS_CA_ADDR_LIST`
  (unicast to the gateway; `EPICS_CA_AUTO_ADDR_LIST=NO`), so no NAT or
  broadcast question arises, intra-appliance URLs and browser-facing
  URLs are the same address, and the fleet port is what it says. The
  alternative — bridge networking with `17665:8080` — works for unicast
  CA on Linux but splits `appliances.xml` into inside/outside addresses;
  not worth it.
- **Port 17665 for mgmt + retrieval** — the appliance's conventional
  port, what every Phoebus example assumes. The image's Tomcat listens
  on 8080, so `deploy/server.xml` (stock Tomcat 11 `server.xml` with the
  connector on 17665, optionally `address=` the lab interface) is
  bind-mounted over it. One file, upstream-shaped.
- **Who runs it.** The unit runs as the service account (the site
  profile's `is_unit_template` requires `User=@SERVICE_USER@`), which
  joins the `docker` group on this box; the container runs Tomcat as that
  same uid (`user:` in compose) so archive files on the host belong to
  the service account. The Phase 3 install checks the image's Tomcat
  work/temp directories tolerate a non-root uid (the pilot ran tarballs,
  §12); rootless Docker is the fallback.
- **Heap and memory.** `JAVA_OPTS=-Xmx2g` (generous for O(1000) PVs; the
  appliance is built for millions) and a container `mem_limit`, so the
  JVM can never squeeze Tiled or the worker.
- **Environment through the container**: `EPICS_CA_ADDR_LIST`,
  `EPICS_CA_AUTO_ADDR_LIST`, `TZ` from `site.env`; `ARCHAPPL_APPLIANCES`,
  `ARCHAPPL_POLICIES`, `ARCHAPPL_PROPERTIES_FILENAME`, `ARCHAPPL_MYIDENTITY`,
  `ARCHAPPL_SHORT_TERM_FOLDER`, `ARCHAPPL_LONG_TERM_FOLDER`,
  `ARCHAPPL_PERSISTENCE_LAYER` (+ `_JDBM2FILENAME`) set in the compose
  template.

The site profile's "containers deferred" note names the HTTP-shaped
services as the only candidates and says nobody runs a container
platform here. The archiver is HTTP-shaped, and "platform" here is one
Engine package and one unit. Amend that note when this lands: the
archiver is the exception, and this table is why.

---

## 4. Configuration persistence — **RULING 3: a JDBM2 file, plus a nightly config export**

The appliance keeps its PV list and per-PV type info ("typeInfo") in a
pluggable persistence layer. Options at 2.4.1:

| Layer | Needs | Verdict |
|---|---|---|
| `InMemoryPersistence` | nothing | Pilot only — lost on restart |
| `JDBM2Persistence` | one env var naming a file | **Recommended** — one file under the unit's `StateDirectory` (`/var/lib/geecs-archiver/persistence.jdbm2`); no database, no jar, no credentials |
| SQLite over JDBC | the xerial jar in Tomcat's lib + `context.xml` | Not in the official image → bespoke; no |
| MariaDB sidecar (upstream's `docker-compose.single.yml` shape) | a second container with a volume; `mysqldump` backups | The fallback if JDBM2 misbehaves — upstream-documented, self-contained |
| Schema in the GEECS MySQL (the July plan) | a schema + user on the LabVIEW-critical DB box, per facility | No: couples the archiver's config to the box this plan deliberately leaves alone |

**Why the smallest option is safe here.** The configuration of record is
*not* the appliance's store. It is the onboarding rule in this package
plus the committed overlay (§7), re-applied idempotently by
`geecs-archiver onboard`; and the mgmt API's `exportConfig` /
`importConfig` give a byte-exact JSON snapshot for the cases where
re-deriving is not what you want (a PV paused by hand, a one-off
sampling tweak). Persistence therefore only has to survive restarts.
Phase 3 adds a systemd timer: nightly `geecs-archiver export-config` to
`{experiment}/archiver/` on the data share (the logbook's
`{experiment}/logbook/` mirror is the precedent for a non-scan tree).

Two facts to verify in the pilot, not assume: the archive **data** of a
PV re-added with the same policy is picked up from the existing PB
partitions (the store URL and name-to-path mapping are deterministic:
`siteNameSpaceSeparators=[\:\-]` in `archappl.properties` turns
`undulator:u_s1h:current` into `undulator/u_s1h/current…`); and the
official image's baked `context.xml` declares a MariaDB JNDI datasource
(`jdbc:mariadb://mariadb:3306/archappl`) that nothing looks up under
JDBM2 — confirm the logs show no connection attempts, else bind-mount a
`context.xml` without the `<Resource>` (keep `<Loader delegate="true"/>`,
which Hazelcast needs).

---

## 5. Storage

- **Tiers.** STS on tmpfs (`PARTITION_HOUR`, upstream's `hold`/`gather`
  defaults) → LTS on the box's mirrored local disk (`PARTITION_YEAR`).
  **MTS skipped**: at our volume a two-store ladder is enough and one
  fewer ETL hop. Both folders live under `GEECS_ARCHIVER_DATA_ROOT`
  (`site.env`, §6), STS as a symlink into tmpfs per upstream's guide.
- **Format.** Protocol-buffer files (the default). The 2.4 Parquet
  backend uses the same partitioning and is readable by standard tools —
  attractive for pandas, but Phoebus/`pbraw` and the retrieval API are
  format-agnostic, so PB stays for v1 and Parquet is a later, reversible
  choice per store.
- **Sizing** (HTU/Undulator today: ~650 served PVs, ~480 readbacks). The
  gateway suppresses exact repeats, so a static PV costs nothing.

| Case | Rate | Volume |
|---|---|---|
| Physical ceiling: every PV changes every 5 Hz frame, no throttle | ~2.4 k events/s × ~15 B | ~3 GB/day |
| Default policy, MONITOR at 1 s, *everything* still changing | ≤1 event/s/PV | ~0.6 GB/day |
| Realistic (analysis scalars while beam runs + a few noisy readbacks) | tens of PVs, part of the day | single-digit GB/month |

  The pilot replaces the guess with a measurement (the appliance reports
  per-PV event rates and storage rates) before the full set is committed.
  A 2 TB mirror is years in every row.
- **The lever we never pull:** the gateway's monitor deadband stays 0.0
  (`PV_CONTRACT.md` §6; inheriting the DB tolerance there suppressed
  real sub-tolerance motion from s-files — a shipped bug fixed in
  gateway 0.5.1). Archive-rate control is the appliance's: per-PV
  `samplingperiod` → the overlay's skip list → ETL decimation
  (`firstSample_3600` and friends) — all archiver-side; later, if ever,
  the `DBE_LOG`/ADEL split in the gateway (`DESIGN.md`).
- **Disk is the metric that pages.** The ETL's
  `OutOfSpaceHandling` default deletes *source* streams when the
  destination is full; the alert threshold sits well before that.
- **NetApp as LTS** stays gated on the ETL-on-SMB test (run the ETL
  against a share-mounted LTS for a week or two, watch for stalls or
  corruption) — unchanged from the draft. The mirror is the default.
- **Backups.** Config: the nightly export (§4). Data: LTS is append-only
  yearly files per PV — an `rsync` to the share is the whole recipe when
  someone wants one; the tape/HPSS tangent stays parked.

---

## 6. Site profile integration — the flexibility this plan exists for

Every facility value the archiver needs has one home. New `site.env`
keys (`deploy/site.env.example` grows them with the usual line-by-line
comments; `docs/platform/site_profile.md` lists them):

```ini
# ---- archiver -----------------------------------------------------------
# The appliance's own address as every client reaches it (install-time):
# rendered into appliances.xml and into the client config.ini [archiver] url;
# the Phoebus Data Browser datasource is pbraw://<this>:17665/retrieval.
GEECS_ARCHIVER_HOST=192.168.6.14
# Where the archive lives on this host (install-time): <root>/sts (a tmpfs
# symlink) and <root>/lts on the mirrored local disk — never a live store on SMB.
GEECS_ARCHIVER_DATA_ROOT=/srv/geecs-archiver
# JVM options for the engine's Tomcat (runtime); -Xmx is the heap ceiling.
GEECS_ARCHIVER_JAVA_OPTS=-Xmx2g
```

Reused, not duplicated: `EPICS_CA_ADDR_LIST` / `EPICS_CA_AUTO_ADDR_LIST`
(the appliance reads the same names every CA client does — nothing is
re-mapped), `GEECS_EXPERIMENT` (the PV prefix the onboarding rule
filters on), `TZ`, `GEECS_SERVICE_USER` / `GEECS_SERVICE_HOME` /
`GEECS_CHECKOUT_ROOT` / `GEECS_POETRY`.

Fixed fleet facts, not site values: port 17665, `StateDirectory=
geecs-archiver`, the image tag, the appliance identity `appliance0`.

**Client side**: one key, `config.ini [archiver] url =
http://<host>:17665`, read by the CLI, the retrieval helper, the portal
and logbook links (Phase 5) and a later MCP domain; the bootstrap renders
it from `GEECS_ARCHIVER_HOST` like the other fleet endpoints.
`docs/tutorials/getting_started.md`'s key table gains the row. Phoebus:
`org.csstudio.trends.databrowser3/urls=pbraw://<host>:17665/retrieval` in
that facility's `settings.ini`.

**A second facility** — a different network, its own gateway, its own
MySQL — copies `site.env.example`, changes every value, keeps every key,
runs `deploy/bootstrap_host.sh --only archiver` and then
`geecs-archiver onboard --experiment <theirs>`. Nothing in the repository
changes. The per-experiment PV prefix (`undulator:` here, something else
there) keeps the two archives' names disjoint by construction, so a
viewer that reaches both networks can list both `pbraw://` URLs in one
Data Browser — consolidation at the viewer, never at the archive, which
is also the gateway `DEPLOYMENT.md` §5 doctrine ("monitoring layer, not
access layer") read for a world where the networks do not meet.

**Bootstrap and render** (*done in the fleet-wiring PR, 2026-10-02*).
`bootstrap_host.sh` gains the `archiver` service: clone `archiver-checkout`, its poetry env, a Docker Engine
prerequisite check shaped like the Redis one (judge the packaged unit,
not just a binary), the image pull, and the conf render into staging;
the root steps gain `install -d /etc/geecs/archiver` and the conf
install. `render_units.sh` gains the unit template. The non-unit files
(`compose.yaml.in`, `appliances.xml.in`) are rendered by the package's
own `deploy/render_conf.sh` — `render_units.sh` refuses anything that is
not a unit template by design, and should keep doing so.

**Observability.** `scripts/fleet_status.sh` gains an `Archiver` row from
`GET /mgmt/bpl/getVersions` + `getApplianceMetrics` (version, PV count, disconnected
count — the archiver-side mirror of the gateway's `connected` PVs) and
`17665` in `FLEET_PORTS`; `scripts/lab_status.sh` probes the port;
`docs/platform/fleet_map.md` gets the row and a diagram node; the Data
Flow Map gets its `INFO` entry; the `/fleet-status` skill lists the row.

---

## 7. The archive set — **RULING 4** (where the rule lives) **and RULING 5** (setpoints)

The rule, in `geecs_archiver/archive_set.py`, from the same two batched
GEECS-Core queries the gateway uses (`GeecsDb.get_experiment_devices`,
`get_experiment_device_variables`) and `geecs_core.pv_naming`:

- **Include** the readback PVs of every enabled device's `get='yes'`
  variables; every device's `connected` status PV (state changes only;
  the uptime history we have never had); derived channels declared in
  the experiment's `gateway/derived_channels.yaml` (the gateway's own
  overlay file, read at its conventional path — one declaration, two
  consumers).
- **Exclude** `systimestamp` / `acq_timestamp` (advance every frame by
  design — pure disk burn), the `cagateway:*` diagnostics, and the
  `path`-typed long-string PVs (the appliance cannot type the gateway's
  char-array channels — §12).
- **Setpoints (`:SP`) — RULING 5.** The draft excluded them ("the
  readback reflects converged state"). Recommended now: **include**. A
  setpoint changes only when someone puts to it, so it costs nothing,
  and it carries the operator's *intent* and the refused-write alarm
  history (`PV_CONTRACT.md` §7) that a readback never shows.

**Why the rule can live here without importing the gateway (RULING 4).**
Two packages applying one DB rule is a drift risk; the runtime truth
check closes it: a PV the rule wants and the gateway does not serve
never answers on CA, so the appliance cannot finish its archive request
and keeps it on its **never-connected list** — which `onboard` reads on
every run and fails on. (The pieces both sides need — the configs-repo
lookup, the status-PV name, the derived-channel name parts — live in
GEECS-Core and GEECS-Schemas, so the only thing left to drift is the
get-list/settable rule itself.) That check is needed
anyway, and it makes drift loud at the moment it happens. If it bites
repeatedly, the upgrade path is the one GEECS-Core already took for
`variable_types`: move the served-set rule into the shared library and
have both the gateway and this package import it. The third option —
the gateway writing a manifest file at startup for this tool to read —
ties onboarding to the gateway's host and is not recommended.

**Curation overlay** — optional, per experiment, in the configs repo
beside the gateway's: `scanner_configs/experiments/<Experiment>/archiver/
archive_policy.yaml`, validated by `geecs_schemas.archive_policy.ArchivePolicy`
(schema_version 1 — the config vocabulary's home is GEECS-Schemas):

```yaml
schema_version: 1
exclude: ["*:uc_*:image_size*"]            # PV globs the rule would otherwise include
include: ["undulator:cagateway:devices_connected"]   # beyond the rule; never paused
sampling_overrides:
  - match: "*:u_vacuumgauge:*"
    sampling_period: 10                     # sent with the request: the overlay is the one table
include_setpoints: true
```

**Idempotent by construction.** `geecs-archiver onboard --experiment X`
diffs the desired set against `getAllPVs`: new PVs → `archivePV` (bulk
JSON); PVs the rule no longer wants → `pauseArchivingPV` (data is never
deleted by this tool); a policy mismatch → `changeArchivalParams`.
`--dry-run` prints the diff; the exit status is non-zero when any PV is
never-connected. Rerunning it after a DB edit becomes the same reflex as
restarting the gateway — two commands, one change.

`deploy/policies.py` is site-neutral (folders from the environment) and
names only the stores: one policy, STS → LTS, with a 1 s MONITOR fallback
for a request that carries no sampling. Every request from this tool does
carry its sampling (the appliance's user-specified sampling wins over the
policy), so the overlay is the one table. `getFieldsArchivedAsPartOfStream`
is empty — no EPICS record fields exist behind these PVs; `.DESC` is read
by the appliance at runtime for display where the gateway serves one.

---

## 8. Phases and gates

```
Phase 0  rulings (this document)                —            then the skeleton PR
Phase 1  pilot: official image, in-memory       ~half a day  proves CAJ↔caproto; measures rates; throwaway
Phase 2  the package + deploy recipe (one PR)   ~2 days      skeleton, mgmt client, rule + tests, templates,
                                                             bootstrap/render/fleet-status, docs, CI leg
Phase 3  production on the services box         ~1 day       JDBM2, STS tmpfs + LTS mirror, disk alert,
                                                             nightly export, Phoebus line
Phase 4  onboarding at scale                    ~1 day + a lab week of observation
Phase 5  consumers                              ongoing      pandas helper, portal/logbook history links, MCP read domain
Phase 6  soak → Citadel retirement              the maintainer's call, outside the repo
```

**Phase 1 — pilot (throwaway).** On a Linux box on the lab subnet with
Docker Engine — **RULING 6: which box.** The interim services host
(192.168.6.14) is the obvious candidate but needs Docker installed on
the box that also runs LabVIEW's MySQL; any scratch Ubuntu machine or VM
on the subnet is equally good; **not** Docker Desktop on a Mac over VPN
(its UDP proxy already broke PVA search replies in this fleet, and CA
search is UDP too).

```bash
docker run --rm --network host \
  -e EPICS_CA_ADDR_LIST=192.168.6.14 -e EPICS_CA_AUTO_ADDR_LIST=NO \
  -e ARCHAPPL_PERSISTENCE_LAYER=org.epics.archiverappliance.config.persistence.InMemoryPersistence \
  -v "$PWD/server.xml:/usr/local/tomcat/conf/server.xml:ro" \
  ghcr.io/archiver-appliance/epicsarchiverap:singletomcat-2.4.1
# then http://<box>:17665/mgmt/ui/index.html
```

Archive a deliberately diverse handful:

| PV | What it tests |
|---|---|
| `undulator:cagateway:heartbeat` | a guaranteed 5 s change rate — liveness with no hardware dependence |
| `undulator:u_s1h:current` | float readback with units/precision metadata |
| a camera analysis scalar | NaN handling (a failed analysis publishes NaN) |
| any `…:connected` | DBR_ENUM archiving + label retrieval + the COMM alarm transition |
| a `…:localsavingpath` | long-string char-array channels (`PV_CONTRACT.md` §4) |
| any `…:SP` | the setpoint half of RULING 5, including a refused put's alarm |

Success criteria, all required: each PV reaches "Being archived" **and
connected** within ~5 minutes; retrieval returns **GEECS timestamps**
(the gateway's timestamp ladder, not receive time); a **gateway restart**
mid-archive is logged, survived, resumed, and shows as an honest gap; a
**device power-cycle** does the same via the INVALID/COMM transition;
**Phoebus Data Browser** plots history behind a live trace with the one
settings line; the non-root-uid, JNDI-datasource and re-add-same-PV
checks of §3–§4 pass; and `getPVDetails` on the full desired set
(submitted in dry-run through the tool's first cut, or by hand) yields
the event-rate numbers §5 is waiting for.

Bail-out, unchanged: a *protocol-level* CAJ↔caproto disagreement (not a
config mistake) stops the arc; the fallback is the lightweight Python
CA→time-series collector sketched before the gateway existed. Expectation:
not exercised — caproto's server already serves libca clients (aioca,
Phoebus, the Windows control machines) daily.

**Phase 2 — the package.** *Built 2026-10-02 (the PR carrying this plan): the
package, its tests, the deploy templates and the schema kind; the fleet wiring
of §6 follows in its own PR.* Everything in §2, §6 and §7 except the live
host steps, as one PR with docs and bookkeeping in the same change;
tests need no hardware (the rule against DB fixtures, the mgmt client
against a recorded BPL, the renderer against a sample `site.env`); the
fleet-status row is live-verified once Phase 3 exists.

**Phase 3 — production.** The bootstrap *is* the install
(`--only archiver`), then the printed root lines. If the services box
slips past the pilot, the pilot container may keep running on the
interim host with JDBM2 persistence and a disk watch as a stopgap (that
box has 100 GB; single-digit GB/month) — the maintainer's call, named
here so it is a decision rather than a drift.

**Phase 4 — onboarding at scale.** A curated dozen → a real lab week
(disk growth, engine CPU, retrieval latency) → the full set. The
maintainer's existing `get='yes'` curation in the GEECS DB remains the
primary lever for *what* is archived; the overlay is for exceptions.

**Phase 5 — consumers.** Phoebus (the settings line joins the Phoebus
recipe in `docs/geecs_gateway/`); a retrieval helper in this package → pandas
over `getData.json`; **history links** from the Data Portal's run page
and the logbook's scan card — the device's PVs over the scan's time
window, a deep link into the retrieval UI or a CSV (small, and the first
place operators will meet the archive); an MCP `archiver/` read domain
(history for a PV over a window) — the MCP is an experiment and gates
nothing. Grafana only if a wall-dashboard need appears.

---

## 9. Risk register

| Risk | Exposure | Mitigation |
|---|---|---|
| caproto↔CAJ interop defect | Low; the one untested seam | Phase 1 exists to retire it; bail-out defined; 2.4.1's "archiving but never connected" fix makes the criterion honest |
| Docker Engine as a new runtime on the box | Certain, small | Ubuntu packages, one unit, one container in the whole fleet; stated as the exception in the site profile |
| JDBM2 is old embedded-database code | Medium | Config is reconstructible (§4); the MariaDB sidecar is the upstream-shaped fallback |
| Upstream image changes under a tag | Low | Pin by tag (digest if it ever moves); an upgrade is a reviewed PR; weekly images are never deployed |
| Unauthenticated mgmt UI and API | Real | Lab subnet only (the stance CA itself takes); optional `address=` on the connector; never port-forwarded |
| Disk growth surprises | Medium | §5 ceiling + a measured lab week before the full set; the alert is a Phase 3 deliverable, ahead of the ETL's delete-source default |
| Name-to-path mapping | Low | Our components are `[a-z0-9_]`; the appliance splits on `:`/`-`, so a facility whose experiment name carried a dash would merely nest one level deeper |
| Project health | Low | Active releases through 2026, multi-lab userbase; data in an open, documented format (PB; Parquet available) — not a Citadel-style lock-in |

---

## 10. What the July draft decided that still stands

Scalars only; per-shot correlation is Tiled's; the gateway deadband stays
0.0 and archive-rate control is the appliance's; `connected` PVs are
archived; the timestamp churn is excluded; the appliance reconnects
through gateway restarts on its own (a restart is a gap, not an
incident); a DB change is "restart the gateway, rerun onboarding";
Citadel runs in parallel until soak; disk is the metric that pages; the
pilot is throwaway and its value is the verdict.

---

## 11. Rulings needed before the skeleton PR

1. **Package.** New `GeecsArchiver/` (recommended) — vs a gateway module or a `deploy/`-only recipe.
2. **Runtime.** The official `singletomcat` image under systemd, host networking, port 17665 (recommended) — vs native Tomcat 11 + JDK from tarballs.
3. **Persistence.** JDBM2 file in the state directory + nightly `exportConfig` (recommended) — vs a MariaDB sidecar — vs a schema in the GEECS MySQL.
4. **Where the archive-set rule lives.** In this package, with the never-connected check as the drift alarm (recommended) — vs moving the served-set rule into GEECS-Core now — vs a gateway-written manifest.
5. **Setpoints.** Archive `:SP` (recommended) — vs the draft's exclusion.
6. **Pilot host.** Install Docker Engine on the interim services host — vs a scratch Linux box/VM on the lab subnet.

---

## 12. Phase 1 results — pilot run 2026-10-02 on the interim services host

Run as the service account with no sudo: Temurin JDK 21.0.12 + Tomcat
11.0.26 + the 2.4.1 release tarball unpacked under the account's home,
`quickstart.sh` (hostname patched to the box's address) under
`setsid nohup`, `JAVA_OPTS=-Xmx768m`, in-memory persistence,
`EPICS_CA_ADDR_LIST=192.168.6.14`. Up in ~40 s; all four WARs report
2.4.1. Nine PVs submitted at 10:28 PDT, "Being archived" + connected
within ~70 s. **Verdict: the CAJ↔caproto seam works; the bail-out is
retired.**

| Check | Result |
|---|---|
| Float readback (`u_s1h:current`, EGU/PREC metadata) | PASS — stored sample `10:28:24.229` equals the CA-side timestamp to the millisecond |
| 1 Hz camera analysis scalars (`uc_alineebeam3:centroidx`, `meancounts`) | PASS — device timestamps (`.985/.986`) preserved, not receive time |
| Gateway heartbeat (5 s cadence) | PASS |
| Enums: `…:connected` (status), `pulsewire_dg645:inhibit` (device enum) | PASS |
| Setpoint `u_s1h:current:SP` | PASS |
| Plain `string` (`pulsewire_dg645:trigger_source`) | PASS — `DBR_SCALAR_STRING` |
| **Long-string `path` (`uc_alineebeam3:localsavingpath`, char array 512)** | **FAIL** — engine `MetaGet`: "Cannot determine DBR type"; archive request aborted. The 110 `path` PVs are **excluded from the archive set** (§7) until someone cares; they change rarely and carry no physics. Upstream question, not ours to fix now |
| **Gateway restart** (`caput …:cagateway:restart Restart`, 10:32:13) | PASS — gateway back with 109 devices in ~15 s; the appliance showed 8 disconnected PVs then 0 on its own; the archive carries the outage as `cnxlostepsecs=…334 / cnxregainedepsecs=…348` (14 s) on the first sample after, and the heartbeat series resumes from the restarted counter |
| "Startup" samples | As designed: each PV's first stored sample carries the PV's *own* last-change timestamp (the gateway's previous start on 09-29, a DG645 change on 09-30), flagged `startup=true` — days-old dates, not a clock fault |
| Retrieval | `getData.json` serves EGU/PREC/DESC metadata and severity/status per sample; `Never`-connected and currently-disconnected lists answer correctly |
| NaN from a failed analysis | NOT exercised (the camera was analysing throughout) |
| Device power-cycle | NOT exercised (no hands in the lab) — Phase 4's lab week |
| Phoebus Data Browser | Owner's visual check pending; settings line: `org.csstudio.trends.databrowser3/urls=pbraw://192.168.6.14:17665/retrieval` |
| Full-set event-rate measurement (3,244 readbacks) | NOT run: the list is generated on the host (`fullset_readbacks.json`, 1,866 float / 1,065 enum / 114 status / 110 path / 89 string; 2,842 `:SP`), but submitting thousands of monitors against the production gateway during the day was held back as a load decision for the owner |

Footprint at 9 PVs: JVM RSS 735 MB (the heap cap, as expected — the
appliance pre-sizes buffers), 2.7 events/s, 0.01 GB/day — two 1 Hz
camera scalars and the heartbeat account for nearly all of it, which is
the §5 picture in miniature.

Host facts that bind Phase 3: Ubuntu 22.04, 4 cores, 15 GB RAM (~7 GB
free), 39 GB free disk, Java 11 only, **no Docker**, no passwordless
sudo, outbound HTTPS to GitHub and ghcr.io works. RULING 6 resolved by
circumstance: the pilot ran from tarballs; the container runtime waits
for the production install.

The pilot is left running (in-memory, 9 PVs, heap-capped) for the
owner's look at `http://192.168.6.14:17665/mgmt/ui/index.html`; stopping
it is `pkill -f quickstart_tomcat` as the service account, and
`~/archiver-pilot/` is the only residue.

---

## References (upstream, as read 2026-10-02 at tag 2.4.1)

- Releases: <https://github.com/archiver-appliance/epicsarchiverap/releases> (2.4.1, 2026-07-21; weekly pre-releases)
- Release announcement: <https://epics.anl.gov/tech-talk/2026/msg00804.php>
- Docs: <https://epicsarchiver.readthedocs.io/> — sysadmin install guide, `env-vars`, `create-policy-file`, `sqlite`, `backingupconfig`, `parquet`, `storage_plugins`
- Image workflow: `.github/workflows/publish-image.yml` (`ghcr.io/archiver-appliance/epicsarchiverap`, targets `singletomcat|mgmt|etl|engine|retrieval`, tag `<target>-<git tag>`); `Dockerfile` (Tomcat 11, Temurin JDK 21, the `ARCHAPPL_*` defaults and the baked `context.xml`)
- Persistence classes: `src/main/org/epics/archiverappliance/config/persistence/` (`InMemory`, `JDBM2`, `MySQL`, `Redis`); `ARCHAPPL_PERSISTENCE_LAYER_JDBM2FILENAME`
- Sample `appliances.xml` (single appliance, 17665 + cluster 16670) and `policies.py`: `src/sitespecific/tests/classpathfiles/`
- Phoebus Data Browser datasource: <https://github.com/sasaki77/archiverappliance-datasource> (Grafana, optional)
