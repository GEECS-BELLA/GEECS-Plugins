# EPICS Archiver Appliance — Deployment Plan

Continuous, between-scan archiving of GEECS control variables: the niche the
legacy NI SVE → Citadel path fills today, and the one thing in the EPICS
migration story we have not yet stood up. Bluesky/Tiled captures *per-run scan
data only* — "what was the chiller temperature doing last Tuesday at 3am" has
no answer in our stack until this lands.

This document is the plan, not (yet) the deployment record. It follows the
same structure as the gateway's own rollout: small verified phases, each with
explicit success criteria and a bail-out point. Companion docs:
`PV_CONTRACT.md` (what the archiver, as a CA client, may rely on),
`DEPLOYMENT.md` (the gateway service it archives from), `DESIGN.md`
(why archive-rate control lives here and not in the gateway deadband).

---

## 0. What we are deploying, honestly

The [EPICS Archiver Appliance](https://github.com/archiver-appliance/epicsarchiverap)
is the accelerator community's standard PV archiver (SLAC/BNL/ALS lineage,
actively maintained — latest stable **2.3.1, March 2026**). It is a **Java web
application**: four WARs (`mgmt`, `engine`, `etl`, `retrieval`) running in
Apache Tomcat.

- **What that means for us:** this is *deploy-and-configure* work, not Java
  programming. Nobody writes Java. Configuration is a handful of files
  (an XML appliance identity, a Python-syntax `policies.py`, env vars) plus a
  REST API and web UI for PV management.
- **What it costs:** one more always-on service, in a runtime (Tomcat/JVM)
  we have no prior operational experience with. Logs, heap sizing, and
  failure modes are unfamiliar territory. This is the real price of the
  "best tool, not shortcut" choice, and it is the main thing the pilot
  phase must de-risk.
- **Version requirements** (current as of 2.3.1): **JDK 21+** and
  **Tomcat 10 or 11** (2.2.0 dropped Tomcat 9). Both install from stock
  Ubuntu 22.04 packages / upstream tarballs. Note most community Docker
  images and older blog-style guides are Tomcat-9-era — trust the
  [official docs](https://epicsarchiver.readthedocs.io/) over anything
  that predates 2.2.0.

### Why this is now a stock integration

The appliance is a plain Channel Access client. The gateway already serves the
Undulator experiment's monitoring set (~650 PVs) from `192.168.6.14` with GEECS
timestamps, correct types, units/precision metadata, and INVALID/COMM alarm
transitions on device disconnect — exactly the interface the archiver was
built to consume. No GEECS-specific code is required in the archive path.

### Non-goals

- **Images / arrays** — off CA by design (`DESIGN.md`); a future distributed
  PVA workstream. The archiver here is scalars only.
- **Per-shot correlation** — that is Bluesky/Tiled's job. The archiver stores
  independent time series; joining them to shots stays out of scope.
- **Replacing s-files or Tiled** — this adds the *continuous* record, it
  replaces nothing in the scan-data path.
- **Retiring Citadel on day one** — the SVE/Citadel path keeps running in
  parallel until the appliance has demonstrably archived through real lab
  weeks (including device power-cycles and gateway restarts).

---

## 1. Decisions to make up front

### Host topology: pilot on the existing box, production on a new one

**Decided (2026-07-07).** The existing server (`abmx`, 192.168.6.14 — MySQL +
Tiled + gateway) has only **100 GB of local storage**, which disqualifies it
as the long-term home of anything whose steady-state job is to grow disk.
A new box is being procured (§10 for the example config), and the split is:

- **New box: gateway + archiver + Tiled** — the modern stack, co-located
  with the mirrored terabytes its two storage-growing services (Tiled,
  archiver LTS) need. Give it a stable hostname/IP before anything points
  at it; clients get repointed exactly once.
- **Existing abmx: MySQL only.** The GEECS DB is small and slow-growing —
  100 GB is generous for a dedicated DB box. The criticality alignment is
  the real win: abmx becomes the boring, stable, *LabVIEW-critical* box
  nobody touches, and a failure of the new (experimental-stack) box leaves
  LabVIEW GEECS entirely unaffected.
- **MySQL does not move.** It is wired into every GEECS config and the
  LabVIEW device layer; relocating it touches everything for zero benefit.
  The archiver's config schema (§1 below) lives in this existing MySQL,
  reached over the LAN like every other client. abmx's MySQL dumps should
  land *off-box* (NetApp or the new box) so a dead abmx disk is not a dead
  experiment database.

Phase 1 (the throwaway pilot) does not wait on procurement — it can run on
abmx or any Linux box on the lab subnet, since it stores nothing worth
keeping. Phase 2 lands on the new box.

Archiver-specific notes that hold wherever it runs:

- **Disk is the resource that matters.** The archiver is the first service
  whose *steady-state job* is to grow local storage (§5 for the math). Disk
  monitoring stops being hygiene and becomes a requirement.
- **JVM memory.** The engine holds per-PV buffers; for O(500) PVs the
  defaults are fine (this thing is built for millions of PVs), but Tomcat's
  heap should be capped explicitly (`-Xmx1G` is generous for our scale) so
  a JVM can never squeeze Tiled or the gateway.

### Tiled migration (small, one idle window)

Moving Tiled to the new box is deliberately part of this plan — it removes
the 100 GB time bomb. Steps: stop `tiled.service` on abmx → copy the metadata
database and data directory to the new box's mirror → mount the NetApp data
share on the new box (Tiled *references* image assets there; the mount is
required, not optional — an absent mount also degrades scan-number claiming)
→ replicate the systemd unit → update the `tiled_uri` clients use. Keep the
storage placement rules of `DEPLOYMENT.md` §2: metadata on local (mirrored)
disk, bulk assets on NetApp, never a live database on SMB.

### Install route: native Tomcat + systemd (recommended), not Docker

Docker was the initial lean when this was queued, so the reversal deserves
an explanation:

| | Native (tarball + systemd) | Docker |
|---|---|---|
| Matches existing ops on the box | Yes — tiled.service pattern, journald logs | No — new runtime, separate log/restart discipline |
| Official upstream support | Yes — release tarball + `quickstart.sh`, documented site-install layout | No official production image; community images (pklaus, lnls-sirius) are Tomcat-9-era and stale vs 2.2+ |
| Isolation of the JVM | systemd unit limits (`MemoryMax`, etc.) | Container limits (marginally cleaner) |
| Upgrade | swap WARs / rerun install script | rebuild image ourselves (we'd own the Dockerfile — bespoke surface) |

The deciding argument: with no maintained official image, Docker here means
*maintaining our own Dockerfile* — bespoke surface pretending to be a
shortcut. Native + systemd is one more unit file in a pattern the box already
uses. If Phase 1 reveals the native install is genuinely painful, Docker
remains available as the fallback, with our own (small) Dockerfile as the
accepted cost.

### Appliance config persistence: the existing MySQL on abmx

The quickstart mode holds PV configs in memory (lost on restart — fine for
the pilot, disqualifying for production). Production installs persist config
in a MySQL schema. We already run MySQL on abmx for GEECS; the appliance
gets its **own schema and own user** (e.g. `archappl`), reached over the LAN
— so the GEECS database and the archiver never share tables or credentials,
and abmx stays the one DB box. This is the single biggest "production-ize"
step between Phase 1 and Phase 2. (The config schema is tiny — PV list +
policies, not data — so the LAN hop costs nothing.)

---

## 2. Phase 1 — Quickstart pilot (throwaway, ~an afternoon)

Goal: **prove the caproto gateway ↔ archiver engine interop end-to-end**
before investing in any production plumbing. The appliance's engine is a
pure-Java CA client (CAJ); the gateway is a pure-Python CA server (caproto).
Both are spec-compliant and this pairing is expected to just work — but it is
the one genuinely untested seam in the whole plan, so it goes first.

On abmx (or any Linux box on the lab subnet):

```bash
# prerequisites
sudo apt install openjdk-21-jre-headless
mkdir ~/archiver-pilot && cd ~/archiver-pilot
# download archappl_v2.3.1.tar.gz (github releases) + apache-tomcat-11.x tarball
tar xzf archappl_v2.3.1.tar.gz
export EPICS_CA_ADDR_LIST=192.168.6.14
export EPICS_CA_AUTO_ADDR_LIST=NO
./quickstart.sh apache-tomcat-11.*.tar.gz
# wait 2–5 min, then: http://192.168.6.14:17665/mgmt/ui/index.html
```

Archive a deliberately diverse handful through the mgmt UI:

| PV | What it tests |
|---|---|
| `Undulator:CAGateway:HEARTBEAT` | guaranteed 5 s change rate — liveness with zero hardware dependence |
| `Undulator:U_S1H:Current` | float readback with units/precision metadata |
| a camera analysis scalar (e.g. `…:centroidx`) | NaN handling (failed analysis publishes NaN) |
| any enum PV (e.g. a `…:CONNECTED` or an on/off device var) | DBR_ENUM archiving + label retrieval |
| a `path`-typed PV (`…:localsavingpath`) | long-string char-array channels (`PV_CONTRACT.md` §4) |

**Success criteria (all must hold):**

1. Each PV goes green ("Being archived") in the mgmt UI within ~5 minutes
   (the appliance samples event rates before committing a policy).
2. Retrieval returns data with **GEECS timestamps** (the gateway's timestamp
   ladder, not archiver receive time):
   `http://…:17665/retrieval/data/getData.json?pv=Undulator:CAGateway:HEARTBEAT&from=…&to=…`
3. **Restart the gateway** while archiving: the appliance must log the
   disconnect, survive it, resume on reconnect, and the retrieved series must
   show the gap honestly (disconnect/reconnect events are first-class in the
   PB data model).
4. **Power-cycle or unplug a real device**: same expectation, driven by the
   gateway's INVALID/COMM alarm transition instead of a CA disconnect.
5. Phoebus Data Browser plots history for one of these PVs (§6 for the
   one-line datasource setting) — this is the user-visible payoff and the
   moment the whole chain is demonstrated.

**Bail-out:** if the CAJ↔caproto seam shows real interop trouble (not
config mistakes — actual protocol disagreements), stop and reassess. The
fallback direction is the previously-sketched lightweight Python CA→TSDB
collector: less capable, but Python-native and solo-maintainable. Do not
grind against a protocol-level incompatibility in a Java stack we cannot
patch. (Expectation: this bail-out is not exercised; caproto's server side
already interoperates with libca-based clients daily via aioca/Phoebus.)

Everything from the quickstart is discarded afterwards — its value is the
verdict, not the install.

---

## 3. Phase 2 — Production install on the new box

Only entered if Phase 1 passes and the new box (§1, §10) is racked. On the
new box, Ubuntu LTS with the data mirror mounted; then, in order:

1. **MySQL schema + user** for appliance config (`archappl`) on abmx's
   existing MySQL, per the
   [official install guide](https://epicsarchiver.readthedocs.io/en/latest/sysadmin/installguide.html).
   MySQL connector jar goes into Tomcat's lib.
2. **Site install layout** under `/opt/archappl` (or the box's convention):
   Tomcat 11, the four WARs deployed via the release's install scripts,
   `appliances.xml` declaring a **single appliance** with identity
   `appliance0`, all URLs bound to the new box's (stable) address.
3. **Storage tiers** on the data mirror (see §5 for sizing):
   - STS (short-term, minutes–hours): tmpfs or the mirror
   - MTS (medium-term, days): the mirror
   - LTS (long-term): the mirror.
     PB files are plain append-only files, so *unlike* the Tiled SQLite
     database, moving LTS to NetApp later is not automatically forbidden —
     but it stays local until someone demonstrates the ETL job behaves on
     SMB. Storage placement rules in `DEPLOYMENT.md` §2 still apply.
4. **Environment:** `EPICS_CA_ADDR_LIST=<gateway host>`,
   `EPICS_CA_AUTO_ADDR_LIST=NO` in the service environment — explicit,
   same-box or not, so the archiver can never wander off looking for PVs
   by broadcast. (Once the gateway also moves to this box, that is
   localhost/the box's own address — set it explicitly anyway.)
5. **systemd unit** (`archappl.service`) modeled on `tiled.service` /
   `DEPLOYMENT.md` §5: `Restart=on-failure`, journald capture, explicit
   `-Xmx` heap cap, `After=network-online.target` (its MySQL is remote).
6. **Config-DB backup**: the archiver's MySQL schema joins whatever dump
   schedule the GEECS DB uses (it is tiny — PV list + policies, not data).

The gateway's own move to this box is the same recipe as its abmx deploy
(`DEPLOYMENT.md`: clone + poetry install + config.ini + systemd unit), plus
a one-time client repoint: `[epics] ca_addr_list` in the shared config.ini
on the Windows machines, and `EPICS_CAS_INTF_ADDR_LIST` in the unit. It can
move before, with, or after the archiver — the archiver reconnects either
way.

Ordering note vs the gateway-as-a-service work: the archiver does not
*require* the gateway to be under systemd first (CA clients reconnect), but
production archiving against a hand-launched gateway is silly — land the
gateway unit first. The two efforts touch disjoint files and can proceed in
parallel.

---

## 4. Phase 3 — PV onboarding at scale

Manual UI submission is fine for five PVs, not for 650. The appliance's mgmt
REST API takes bulk JSON:

```
POST /mgmt/bpl/archivePV
[{"pv": "Undulator:U_S1H:Current", "samplingmethod": "MONITOR", "samplingperiod": "1.0"}, …]
```

Deliverable: a small module in this package (e.g.
`geecs_ca_gateway/archiver_onboarding.py` + a console script) that derives
the archive list from **the same DB enumeration the gateway serves from**
(`GatewayConfig.from_geecs_experiment`) — single source of truth, so the
archive set tracks the served set by construction. It should:

- **Include:** readback PVs of the monitoring (`get='yes'`) set.
- **Exclude by default:** the churn set — `…:systimestamp` /
  `…:acq_timestamp` (advance every frame by design; archiving them is pure
  disk burn), `CAGateway:UPTIME`/`HEARTBEAT` (diagnostics, not physics),
  and `:SP` setpoints (the readback already reflects converged state;
  revisit if setpoint-vs-readback history proves useful).
- **Archive `…:CONNECTED` PVs** — cheap (state changes only) and exactly
  the uptime/health history we have never had.
- Be **idempotent** (safe to rerun after devices are added to the DB;
  already-archived PVs are skipped) — rerunning it after a DB change is
  then the same reflex as restarting the gateway.
- Support a **curation overlay** (skip/rate-override lists) in a small
  checked-in file, not code.

Sam's existing `get='yes'` curation in the GEECS DB remains the primary
lever for *what* is archived — the same ~30–60 min of domain work that
already curates the gateway's served set, with no second list to maintain.

Rollout: submit a curated ~dozen first, watch a real lab week (disk growth,
engine CPU, retrieval latency), then submit the full set.

---

## 5. Storage math and archive-rate control

Honest numbers, using the Undulator monitoring set (~480 readback PVs):

- **Static PVs cost ~nothing.** The gateway suppresses exact repeats at the
  monitor level, so an unchanging variable generates zero CA events and
  zero archive bytes. Most of the set is static most of the time.
- **Worst case** (every PV changing every 5 Hz frame): ~2.4 k events/s at
  ~15 bytes/event (PB scalar double) ≈ **3 GB/day ≈ 90 GB/month**. This is
  the physically-possible ceiling, not a projection.
- **With MONITOR sampling at 1 s** (the proposed default policy): the
  appliance throttles each PV to ≤1 stored event/s regardless of the 5 Hz
  stream → ceiling drops to ~0.6 GB/day ≈ **18 GB/month**, again only if
  *everything* changes continuously.
- **Realistic projection:** the continuously-changing population is the
  camera-analysis scalars while beam is running plus a handful of noisy
  readbacks — order tens of PVs for fractions of the day. Single-digit
  GB/month is the expectation; the pilot week (Phase 3) replaces this
  guess with a measurement before the full set is committed.

Levers, in the order to reach for them: per-PV `samplingperiod` in the
onboarding policy → skip-list (§4) → the appliance's ETL-stage decimation
(post-processing operators like `firstSample_3600` for LTS) — all archiver-side.

**The lever we must not reach for:** the gateway's monitor deadband. It
stays 0.0. Archive-rate control belongs in the appliance's sampling
policies (or later, a `DBE_LOG`/ADEL split in the gateway — `DESIGN.md`
"Archive-rate control"). Wiring the DB `tolerance` into the value deadband
suppressed real sub-tolerance motion from scan rows and s-files — a shipped
production bug (fixed in 0.5.1). `PV_CONTRACT.md` §6 pins this.

---

## 6. Consumers

**Phoebus Data Browser** — the primary UI. One settings line (goes in the
same `settings.ini` used for the VPN/Mac launch recipe):

```ini
org.csstudio.trends.databrowser3/urls=pbraw://192.168.6.14:17665/retrieval
```

Right-click any live PV in a display → open in Data Browser → history fills
in behind the live trace. This is the "Citadel viewer, but standard" moment.

**Python/pandas** — the retrieval endpoint speaks JSON/CSV over HTTP
(`/retrieval/data/getData.json?pv=…&from=…&to=…`), so ad-hoc analysis needs
`requests` + `pandas`, nothing appliance-specific. If usage grows, wrap the
two common queries in a small helper here — after real usage patterns exist,
not before.

**Grafana** — a maintained
[archiver datasource plugin](https://github.com/sasaki77/archiverappliance-datasource)
exists. Optional, later, only if a wall-dashboard need appears.

---

## 7. Operations runbook (grows during Phases 2–3)

- **Health:** mgmt UI front page (appliance metrics, disconnected-PV count);
  `curl http://…:17665/mgmt/bpl/getApplianceMetrics`. The engine's
  disconnected-PV list is the archiver-side mirror of the gateway's
  `CONNECTED` PVs.
- **Disk:** the one metric that pages. Alert threshold on the LTS partition
  before the ETL jobs start failing, not after.
- **Gateway restart:** expected-normal event. Appliance reconnects on its
  own; a restart shows as a short gap in every series. No archiver action.
- **DB change / new devices:** restart gateway (existing pattern) → rerun
  the onboarding script (idempotent). Two commands, same reflex.
- **Appliance upgrade:** stop Tomcat, swap WARs from the new release
  tarball, start. Config lives in MySQL and survives. Read release notes
  for PB-format or Tomcat-floor changes (2.2.0 broke Tomcat 9; assume
  such breaks recur).
- **What we deliberately do not monitor yet:** JVM internals, per-PV
  storage rates, ETL timing. Add instrumentation when a real incident
  demands it, not preemptively.

---

## 8. Risk register

| Risk | Exposure | Mitigation |
|---|---|---|
| caproto↔CAJ interop defect | Low, but the one untested seam | Phase 1 exists solely to retire this; bail-out defined |
| Unfamiliar JVM/Tomcat ops | Certain, cost unknown | Pilot on quickstart first; heap-capped systemd unit; logs in journald like everything else |
| Disk growth surprises | Medium | §5 math + measured pilot week before full onboarding; disk alert is a Phase 2 deliverable |
| Box concentration | Reduced by the two-box split (§1) | abmx = MySQL only (LabVIEW-critical, untouched); new box = the modern stack, whose failure LabVIEW never notices |
| Existing abmx disk (100 GB) | Real today | Tiled moves to the new box's mirror (§1); MySQL alone is comfortable in 100 GB; dumps go off-box |
| Mgmt UI/REST has no auth | Real | Lab-subnet exposure only (same stance as CA itself); no port-forwarding to it |
| Appliance project health | Low — active releases through 2026, multi-lab userbase | Data is in a documented open format (PB, protobuf-based; Parquet backend landing in 2.3+) — not a Citadel-style lock-in |

---

## 9. Sequencing summary

```
Phase 1  quickstart pilot          ~afternoon   proves interop; throwaway
Phase 2  production install        ~1–2 days    MySQL config, tiers, systemd
Phase 3  bulk onboarding           ~1 day + a lab week of observation
Phase 4  runbook + consumers       ongoing      Phoebus datasource, disk alert
```

Each phase ends at a stable, useful state; nothing before Phase 3 commits
more than an afternoon. The gateway needs **zero code changes** for any of
this — the onboarding script (Phase 3) is the only new code in the repo,
and it is a client of existing config machinery.

---

## 10. Hardware — the new box

Decided 2026-07-07: buy from ABMX (the vendor of the existing server —
convenience premium accepted deliberately). Example configuration, sized for
gateway + archiver + Tiled with a decade of storage headroom:

**ABMX 1267S3CL** (1U, 4× hot-swap 3.5" bays, ~$4.5–5k configured):

| Item | Pick | Rationale |
|---|---|---|
| CPU | Xeon E-2434 (stock option) | The whole stack is I/O-light; 4 cores is genuinely enough |
| RAM | 64 GB DDR5 ECC (2×32 GB) | ECC for a data-integrity box; 2 DIMM slots left free |
| OS drive | 1 TB M.2 NVMe | OS + service venvs, separate from the data pair |
| Data | 2× 4 TB SATA SSD, bays 1–2 | The mirror: Tiled metadata + data, archiver STS/MTS/LTS |
| Bays 3–4 | empty | Growth is a hot-swap insert, not a new server |
| OS | none shipped | Ubuntu LTS, self-installed |

Included for free on this chassis: a full BMC (ASPEED AST2600, IPMI 2.0 with
HTML5 KVM) — remote console over the lab network/VPN with no license fee.

Setup rules that are not optional:

- **Build the mirror in Linux — `mdadm` RAID1 or a ZFS mirror — during the
  Ubuntu install. Do not enable the Intel chipset RAID in the BIOS.**
  Chipset fake-RAID is opaque like hardware RAID *and* motherboard-bound
  like software RAID; a plain mdadm/ZFS mirror can be read on any other
  Linux machine, which is exactly the property the box holding the lab's
  history needs.
- Redundant PSU deliberately skipped (single point of acceptable failure;
  the LabVIEW-critical services live elsewhere). Mirrored data disks are
  the one redundancy this box must have: it holds the primary copy of the
  continuous archive and Tiled metadata — data that exists nowhere else.
- Give it its stable hostname/IP at install time (§1) — everything
  downstream (CA address lists, `tiled_uri`, `appliances.xml`) hard-codes
  it once.
