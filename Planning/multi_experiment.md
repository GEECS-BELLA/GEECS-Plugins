# Multi-experiment stand-up — the brief (integration branch `feat/multi-experiment`)

Scaffolding per `Planning/README.md`: the promotion PR to `master` deletes this
file after moving anything load-bearing into docstrings and CLAUDE.md files.
Design history: the 2026-10-15 brainstorm (hub model) and the maintainer's
rulings of 2026-10-31, both recorded here. The coordinator runs the slices in
order, one PR each into this branch, with the `/land` ritual (scope check,
`/check`, `scripts/commit.sh`, fresh-context adversarial review with
dispositions, CI green), and merges a lane PR after its review is dispositioned
(maintainer's rule for integration branches). `master` is merged forward into
this branch at least weekly. Promotion to `master` is the maintainer's merge.

## Goal

Two experiments run at once on the same hub with no code changes between
them: the main experiment (Undulator) as deployed today, and the bench
experiment, PulsedWire, as a second, independent set of service instances. A
new experiment is one verb: `experiment up <Name>`; a dead one is
`experiment down <Name>`. Standing up a second experiment from the site profile
with zero code changes is also the repo's own 1.0 criterion (root `CLAUDE.md`
§ "Release & Versioning").

## Facts and rulings (maintainer, 2026-10-31)

- **The hub is the services box**: every scan host, one box. Scan mode requires
  the worker to share the gateway's host (`plans/scan_mode.py::check_gateway_host`).
- **Gateway: one instance per experiment on its own CA server port**, templated
  like the worker. No gateway code change in this arc; clients list two
  `host:port` pairs in their CA address lists. Consolidation into one process
  serving the live set (multi-roster + a reload guard that refuses while a run
  is open) is a LATER arc, not this one.
- **PulsedWire in the DB**: an experiment named approximately "test stand"
  holds the pulsed-wire devices. The maintainer ALSO enabled those devices
  under Undulator so the Bluesky stack could drive them ("hacked into place").
  Slice 1 reads the exact name and both device sets, read-only. The overlap
  check (slice 5) must REPORT the duplicates; removing them from Undulator is
  the maintainer's DB edit, never an agent's (auto mode blocks DB writes anyway).
- **Per instance**: queueserver worker (own ZMQ ports, Redis prefix, state
  dir, spool, `scan_mode.json`), Tiled writer, scanner (own port), logbook
  (own port). **Shared as-is**: Tiled, the Data Portal (already
  `?experiment=`-aware), the archiver (onboard already prefix-scoped), Canopy.
- **Lanes may deploy PulsedWire's units on the services box.** They never
  stop, restart, edit or re-render an Undulator unit, never run an Undulator
  scan, and never write to the DB. A PulsedWire `count` with no movable
  devices is allowed as the connectivity proof; a PulsedWire `sweep` waits
  for the maintainer's go (it moves bench hardware).
- **Facility values have one home** (root `CLAUDE.md`): per-instance values
  live in `/etc/geecs/experiments/<Name>.env`, consumed by template units via
  `EnvironmentFile=/etc/geecs/experiments/%i.env`; committed files carry
  placeholders and an example only. No experiment name, host or port in code.
- **Additive only.** The existing non-templated Undulator units keep working
  unchanged; migrating Undulator onto the templated form is a later,
  maintainer-run step with its own runbook line.

## Slices

1. **Inventory (no code).** Read-only: the DB's experiment rows and the two
   device sets (`GeecsDb`, over VPN after `/lab-status`), the units running on
   the services box (`/fleet-status`), every hard-coded port and state path in
   the qserver startup, the Tiled writer, the scanner and the logbook, and the
   gateway's CA server port option. Write the port plan and the per-instance
   env schema into § Status below. ≤ 1 session.
2. **Template units + rendering.** `geecs-qserver@.service`,
   `geecs-tiled-writer@.service`, `geecs-scanner@.service`,
   `geecs-logbook@.service`, `geecs-ca-gateway@.service` beside the existing
   units; `deploy/render_units.sh` renders them; `deploy/bootstrap_host.sh
   --experiment <Name>` installs one instance; `deploy/experiments.env.example`.
   Pure deploy templating; tests for the renderer where it has them.
3. **Per-instance code paths.** Every port, prefix and state path the
   services read comes from the environment (the per-instance env file), with
   today's values as defaults so the Undulator units are unchanged. The
   gateway takes its CA server port from config. The worker's CA address list
   names its own gateway port. Tests pin the defaults.
4. **The verb.** `deploy/experiment.sh up|down|list <Name>`: writes the env
   file from the example (ports allocated from a documented scheme), scaffolds
   `<configs>/<Name>/` from GEECS-Schemas' empty documents when absent, renders
   and enables the instance's units, prints the ports and the scanner URL;
   `down` stops and disables them and leaves data and configs in place; `list`
   shows the live instances. A Python helper in GEECS-Core is acceptable for
   the configs scaffold and the overlap check; the orchestration stays a shell
   script beside `bootstrap_host.sh`.
5. **Overlap check.** A read-only `GeecsDb` query listing devices enabled in
   more than one live experiment; `experiment up` prints it as a warning naming
   each device and both experiments; a test over `FakeGeecsServer`/a fake DB.
6. **Deploy PulsedWire on the services box** (the proof): env file, units up,
   the second gateway on its port, the worker connected, a PulsedWire `count`
   through the PulsedWire scanner instance. Record the evidence in § Status
   (ports, unit states, `/fleet-status` output) with no host names or
   addresses in the record.
7. **Docs + promotion body.** `docs/platform/site_profile.md` and
   `fleet_map.md` gain the instance model; each touched package's `CLAUDE.md`
   states the rule it keeps (one sentence, no pointer to this file); the
   promotion PR body lists the lane PRs and the decisions still open. No
   version bumps.

## Proof of done

PulsedWire runs as a second instance set beside the live Undulator instance
for a week; `/fleet-status` shows both; the maintainer observes an Undulator
scan unaffected; `experiment down PulsedWire` and `up` again round-trips.

## Out of scope

The single live-set gateway; a scanner with an experiment picker (per-instance
scanners first); migrating Undulator to the templated units; containers or
Kubernetes; any DB write; anything that touches the Undulator instance.

## Size and cost rules

A slice ≤ ~600 changed lines; split when larger. Reviews bounded (~25–30 tool
calls); fix confirmations to a fresh small verifier. No GitHub issues, no PR
comments outside the lane PRs. Public repo: no hosts, addresses or account
names in any committed file or PR body.

## Status

(The coordinator appends dated entries here: the inventory's findings, the
port plan, the env schema, each slice's PR number, the deploy evidence.)

### 2026-10-31 — slice 1, the inventory (read-only)

**DB** (`GeecsDb`, SELECT only). Four `expt` rows: `kHzLPA`, `LaserUpstairs`,
`Test stand default`, `Undulator`. The bench experiment's exact DB name is
**`Test stand default`** (spaces): shot-control device `pulsewire-DG645`,
its own multicast group (the same remote port as Undulator), and an
`expt.RootPath` whose last component is **`TestStand`**. Its enabled set is
five devices — `pulsewire-DG645`, `pulsewire-ESP302-sensor`,
`pulsewire-Hexapod`, `pulsewire-picomotor8742-quads`, `pulsewire-picoscope` —
and **all five are also enabled under Undulator** (118 enabled devices).
DB-wide, those five are the only devices enabled in more than one experiment.

**Services box** (`/fleet-status` + read-only `systemctl`/`ss`): the
Undulator worker family runs here non-templated (`geecs-qserver`,
`-qserver-ready`, `-tiled-writer`, `-scanner`, `-logbook`, plus the portal,
MCP, archiver, Tiled, Redis). **No CA gateway runs on the box**: the
Undulator gateway is still on the old services host, and the box's
`site.env` carries that host's `EPICS_CAS_INTF_ADDR_LIST`. Listening today:
5065/udp, 5075/5076 (PVA), 5567/5568, 6379, 8000, 8200, 8300, 8400, 17665,
60615, 60625 (and the archiver's 10xxx/16670). Nothing on 5064 or 51xx.

**Code: what is fixed per process today**

| Service | Value | Where | Instance-able by |
|---|---|---|---|
| RE Manager | 0MQ control 60615 / info 60625 | stock defaults | stock env `QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER` / `..._INFO_...` (no code) |
| RE Manager | Redis prefix `qs_default` | stock default | `--redis-name-prefix` — the launcher must pass it (code) |
| RE Manager | `existing_plans_and_devices.yaml` | written into the shared startup dir | `--existing-plans-devices` — launcher (code) |
| launcher | doc proxy 5567→5568 | `QS_DOC_PROXY_IN/OUT` | env (exists) |
| worker | doc publish `localhost:5567` | `QS_DOC_PUBLISH_ADDR` | env (exists) |
| worker | experiment | `QS_EXPERIMENT` | env (exists) |
| worker | spool + `scan_mode.json` | unit `Environment=` under `/var/lib/geecs-tiled-writer` | template `%i` (unit) |
| readiness | manager address | `QS_CONTROL_ADDR`, else loopback:60615 | env (exists) |
| Tiled writer | state dir | `GEECS_TILED_WRITER_STATE` | template `%i` (unit) |
| scanner | HTTP 8300 | `--port` default | unit arg (unit) |
| scanner | manager control/info/doc | `config.ini [qserver]` only (`qs_client.read_qserver_config`) | **env override needed (code)** — config.ini is per account, shared by both instances |
| scanner | state dir, writer heartbeat | `$STATE_DIRECTORY`, `GEECS_TILED_WRITER_STATE` | template `%i` (unit) |
| logbook | HTTP 8400, notes DB | `--port` arg; `$STATE_DIRECTORY/logbook.db` | unit arg + `StateDirectory=geecs-logbook/%i` (unit) |
| CA gateway | CA server port 5064 | caproto reads `EPICS_CA_SERVER_PORT` | env — but **only in the gateway's process**: the same key in the worker's env would re-port every port-less address-list entry, so the instance env carries `GEECS_CA_SERVER_PORT` and the template's `ExecStart=/usr/bin/env EPICS_CA_SERVER_PORT=${GEECS_CA_SERVER_PORT} …` hands it over (`Environment=` does not expand `${}`). The ruling "no gateway code change" wins: slice 3's "from config" is met by the instance file |
| CA gateway | serving interface / beacons | `EPICS_CAS_INTF_ADDR_LIST` / `..._BEACON_...` | instance env (overrides site.env) |
| worker | CA address list | `EPICS_CA_ADDR_LIST` (env wins over config.ini) | instance env: `<box>:<GEECS_CA_SERVER_PORT>` first, then the IOC hosts — every instance unit reads the file, so the scanner's readback PVs (a CA client too) follow it |
| fleet probe | manager address | `scripts/qserver_probe.py` reads config.ini `[qserver]` (one per account) | per-instance probe needed for "`/fleet-status` shows both" (slice 6/7) |

The PV prefix is `normalize_component(experiment)` — `test_stand_default:` vs
`undulator:` — so two gateways never serve one PV name even for the five
shared devices (each device then carries one TCP subscription per gateway).

**Port plan** (slot `N` = 1..9 per extra instance; Undulator is slot 0 =
today's values, untouched): CA server `5064+100N`; 0MQ control `60615+100N`,
info `60625+100N`; doc proxy in/out `5567+100N` / `5568+100N`; scanner
`8300+10N`; logbook `8400+10N`; Redis prefix `qs_<instance>`. PulsedWire =
slot 1: 5164, 60715, 60725, 5667/5668, 8310, 8410 — all free on the box.

**Per-instance env schema** (`/etc/geecs/experiments/<Instance>.env`, read by
the template units AFTER `site.env`, so it overrides): `GEECS_INSTANCE`
(= `%i`), `GEECS_INSTANCE_SLOT`, `GEECS_EXPERIMENT` + `QS_EXPERIMENT` (the
DB name, spaces allowed — the instance name is a slug), `GEECS_CA_SERVER_PORT`,
`EPICS_CAS_INTF_ADDR_LIST`, `EPICS_CAS_BEACON_ADDR_LIST`, `EPICS_CA_ADDR_LIST`,
`QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER`, `QSERVER_ZMQ_INFO_ADDRESS_FOR_SERVER`,
`QS_REDIS_PREFIX`, `QS_DOC_PROXY_IN`, `QS_DOC_PROXY_OUT`, `QS_DOC_PUBLISH_ADDR`,
`QS_CONTROL_ADDR`, `QS_INFO_ADDR`, `QS_DOC_ADDR` (the clients' side),
`GEECS_SCANNER_PORT`, `GEECS_LOGBOOK_PORT`.
Paired keys (`QS_DOC_PUBLISH_ADDR` = proxy in, `QS_DOC_ADDR` = proxy out,
`QS_CONTROL_ADDR`/`QS_INFO_ADDR` = the manager's two sockets) restate the slot:
slice 4's verb derives every port from `GEECS_INSTANCE_SLOT`, never as
independent inputs. The client keys are new `QS_*` names, not the existing
site key `GEECS_QS_DOC_ADDR`: the main scanner reads site.env too, and must
not start honouring a key it ignores today.

**Open — the maintainer's site.env edit, not a lane's:** the archiver and
Canopy take `EPICS_CA_ADDR_LIST` from site.env, so they reach a second
gateway only when the site list gains its `host:port` entry.

**Open — needs the maintainer before slice 6** (slices 2–5 do not depend on
it): the data folder. A scan claims `<data root>/<experiment>/Y…`, so a
worker on `QS_EXPERIMENT=Test stand default` would start a new
`Test stand default/` tree beside the existing `TestStand/` one (which holds
2026-09/10 days), and the configs resolver and logbook key on the same name.
Options: (a) accept the new tree; (b) rename the DB experiment to a slug
(the maintainer's DB edit); (c) a later per-instance data-folder key. The
scanner configs repo has no directory for either name yet (slice 4's
scaffold would create it).

### 2026-10-31 — slice 2, template units + rendering

Six instance templates beside the main units — `geecs-ca-gateway@`,
`geecs-qserver@`, `geecs-qserver-ready@`, `geecs-tiled-writer@`,
`geecs-scanner@`, `geecs-logbook@` — each reading `site.env` then
`/etc/geecs/experiments/%i.env` (`@EXPERIMENTS_DIR@`, knob
`GEECS_EXPERIMENTS_DIR`); state under `/var/lib/geecs-experiments/%i/`
(`engine/` shared by worker + writer, `scanner/`, `logbook/`); the gateway's
port handed over in its `ExecStart` alone. `render_units.sh` renders them;
`bootstrap_host.sh --experiment <Instance> [--instance-env FILE]` stages only
them and prints the `enable --now <unit>@<Instance>` line (existing clones used
as they are; config.ini untouched); `deploy/experiments.env.example` is the
schema with slot 1 as the worked example; `check_instance_env`
(`site_env_lib.sh`) refuses a missing key, a leftover placeholder, a name
mismatch and a process-wide `EPICS_CA_SERVER_PORT`. The units consume
`QS_EXISTING_PLANS_DEVICES`, `QS_REDIS_PREFIX`, `QS_INFO_ADDR`, `QS_DOC_ADDR`
that slice 3 teaches the code.

### 2026-10-31 — slice 3, per-instance code paths

Slice 2 merged as #1225. Its review added an optional `GEECS_INSTANCE_CHECKOUT_ROOT`
(site.env, render-time): instances may run from clones of their own, so a deploy
from this branch never moves a main clone. Slice 3 teaches the code the keys the
units carry, every default unchanged:
- `launch_re_manager.sh` passes `--redis-name-prefix` / `--existing-plans-devices`
  only when `QS_REDIS_PREFIX` / `QS_EXISTING_PLANS_DEVICES` are set. The
  0MQ addresses are the manager's own `QSERVER_ZMQ_*_FOR_SERVER`.
- `qs_client.read_qserver_config`: `QS_CONTROL_ADDR` / `QS_INFO_ADDR` /
  `QS_DOC_ADDR` override config.ini's `[qserver]` per field (all three are
  needed when it is absent). The scanner reaches its own manager this way.
- The gateway: no code. Its port is caproto's `EPICS_CA_SERVER_PORT`, set
  in its unit alone and pinned by a gateway test. The worker's CA list
  is `EPICS_CA_ADDR_LIST` from the instance file, which already wins over
  config.ini.
- `bootstrap_host.sh --experiment` renders nothing for qserver or scanner
  when the clone's code does not read those keys. That closes #1225's
  HIGH finding.
