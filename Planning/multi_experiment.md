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
