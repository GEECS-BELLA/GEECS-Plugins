# Changelog — geecs-pva-gateway

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-09-04

### Added

- **`geecs_pva_gateway.fleet` — the fleet roster from the DB, with a
  "not deployed" state.** `fleet_roster(experiment)` returns the
  experiment's camera servers (endpoint IPs hosting enabled image
  devices — the same camera test `PvaGatewayConfig` applies per host,
  now via the shared `config.image_variables`), each marked `deployed`
  by membership in the client `config.ini` `[pva] addr_list` (the
  clients' `EPICS_PVA_ADDR_LIST`, mirroring `[epics] ca_addr_list`;
  absent key = all deployed). A roster host missing from the list hosts
  cameras only nominally — no instance was ever installed — so a failed
  probe there is not an outage. Site-profile arc Phase 4 (the PVA-roster
  literal; plan `Planning/site_profile/00_overview.md`).

### Changed

- **`deploy/gen_fleet_status.py` takes `--experiment`** (default: the
  config.ini experiment) and writes `fleet_status_<experiment>.bob` from
  the DB roster; the hand-curated `HOSTS`/`EXPERIMENT` literals are gone.
  Not-deployed hosts render as a labelled row with no live PVs and no
  restart button. `deploy/fleet_status.bob` is replaced by
  `deploy/fleet_status_undulator.bob` (HTU, the reference deployment:
  11 camera servers in the DB, 9 deployed — the two hosts that dropped
  out of the old 13-host list no longer have image devices).
- `scripts/fleet_status.sh` reads the roster live from the DB through
  this package's env instead of grepping the checked-in screen, and
  prints not-deployed hosts as `[ -- ]` rows (previously `[DOWN]` every
  run).

## [0.4.5] - 2026-08-29

### Fixed

- **`deploy/launch.bat` installs GEECS-Core** — the pull-on-restart
  package list predated the geecs-core split (2026-08-20), so the first
  post-split fleet restart installed a gateway that imports `geecs_core`
  without installing it: an NSSM crash loop (found live on the canary
  during the capture-arc 0.4.4 rollout; healed by installing GEECS-Core
  into the venv, which persists across restarts). **Fleet operational
  note: every box bootstrapped before this fix carries the old local
  `launch.bat` — do NOT `:restart` such a box onto a post-split source
  clone until it has GEECS-Core in its venv (one console pip line) or
  has been re-bootstrapped (which copies the fixed launcher).**

## [0.4.4] — 2026-08-27

### Fixed

- **Timestamp plausibility checked post-epoch-conversion** in
  `_frame_timestamp` (parity with the CA gateway's PV_CONTRACT ladder):
  a LabVIEW value in `(0, offset]` previously became a *negative* Unix
  timestamp on the published NTNDArray instead of falling through to
  receive time — poisoning any consumer keying frames on the PVA
  timestamp (the capture daemon's dedupe, the analysis `acq_timestamp`
  join). Found by the capture-arc audit (2026-08-27); reaches the fleet
  on next service restart (pull-on-restart launcher).

## [0.4.3] — 2026-08-20

### Changed

- Transport, DB, and naming imports now come from the new **geecs-core**
  package (`geecs_core.transport` / `geecs_core.db` / `geecs_core.pv_naming`);
  `geecs-ca-gateway` remains a dep only for `config.effective_vartype`. No
  behavior change.

## [0.4.2] — 2026-08-18

### Fixed

- `DEPLOYMENT.md` smoke test corrected from a bare p4p `get` to a held
  `monitor`: a bare `get` returns the cached value immediately (startup
  placeholder or stale last frame) and never waits out the gating
  round-trip, so the old snippet read `(1, 1)` on a healthy fresh instance
  — verified empirically during review of the docs-site image page, which
  now documents the same behavior.

## [0.4.1] — 2026-08-18

### Changed

- Docs truth-up after the fleet rollout completed (2026-08, 9 boxes at
  uniform 0.4.0): `DEPLOYMENT.md` drops the canary/pilot phase framing
  (fleet-wide is the current state; per-host restart example; machine-account
  share access stated as production-validated with the domain-account
  fallback demoted to contingency; `HOSTS` defined as the should-run roster —
  DB-derived, then hand-curated), and `CLAUDE.md` gains the
  load-bearing-production statement plus
  the missing `gen_fleet_status.py` / `test_entrypoint.py` layout entries.

## [0.4.0] — 2026-08-17

### Added

- `bootstrap.ps1` installs Python 3.11 itself when `py -3.11` is missing
  (silent all-users install of 3.11.9 — the last 3.11 with a binary
  installer — from python.org), so a bare camera server needs no manual
  Python setup before onboarding.
- `deploy/gen_fleet_status.py`: checked-in generator for `fleet_status.bob`;
  its `HOSTS` list is the fleet roster of record (DB snapshot documented in
  the docstring) — update the list, rerun, commit both files.

### Changed

- `fleet_status.bob` grew from the canary-only row to the full fleet: all 13
  camera-hosting endpoints from the experiment DB (enabled devices with
  image-typed variables, grouped by endpoint IP), one
  version/heartbeat/restart row each. Now generated, not hand-edited.
- `DEPLOYMENT.md`: console-first onboarding is now the documented preferred
  path (script, `-Source`, and `-ConfigSource` all straight off the share as
  UNC paths — no local staging; SSH documented as the fallback needing a
  local clone + local INI copy), and the client-access section notes that
  PVA broadcast discovery is subnet-local — the fleet spans several lab
  subnets, so clients wanting cameras fleet-wide carry the full address
  list even on-site.

### Fixed

- Both bootstrap downloads (`python installer`, `nssm.zip`) now pass
  `curl -f`, so an HTTP-level failure (404, proxy/captive-portal page) fails
  the download step loudly instead of surfacing later as a corrupt file.

## [0.3.0] — 2026-07-25

### Changed

- **Pull-on-restart installs from the lab's shared GEECS-Plugins clone instead
  of a wheel drop** (owner decision: reuse the "Active Version" pattern GEECS
  itself launches from). `launch.bat` reinstalls the four intra-repo packages (incl. the
  transitive `GEECS-Schemas`) from
  `GEECS_PVA_SOURCE` (`--no-deps --no-build-isolation`; poetry-core installed
  at bootstrap so restarts need no internet); the clone's checked-out commit
  is the fleet pin — rollout = `git pull` there + restart PVs, rollback =
  `git checkout <rev>` + restarts. Replaces the never-deployed wheel/CURRENT
  machinery; bootstrap's `-WheelShare` becomes `-SourceShare`.

## [0.2.1] — 2026-07-25

### Added

- `bootstrap.ps1 -ConfigSource <path>`: copies `Configurations.INI` into the
  service profile during bootstrap, making a box start-ready in one command
  (console: point at the share; SSH: point at a scp'd local copy).
- `DEPLOYMENT.md` "Client access" section: on-subnet clients need nothing
  (UDP broadcast search); routed/VPN clients list camera-server IPs in
  `EPICS_PVA_ADDR_LIST` / Phoebus `epics_pva_addr_list`.

## [0.2.0] — 2026-07-25

### Added

- **Deployment machinery (PR ladder rung C).**
  - Writable `{exp}:pvagateway:{host}:restart` PV: any put triggers a clean
    shutdown with exit code 86 (`RESTART_EXIT_CODE`) — the fleet rollout
    mechanism (NSSM relaunches, which re-resolves DB config and re-pins the
    wheel). Mirrors the CA gateway's `CAGateway:RESTART` pattern.
  - `deploy/bootstrap.ps1` — one-time per-box setup: layout, venv + install,
    firewall, NSSM fetch + service registration. The service runs LocalSystem
    with `USERPROFILE` overridden to a service-owned profile dir, solving the
    session-0 mapped-drive and LocalSystem-home problems without service
    account passwords.
  - `deploy/launch.bat` — pull-on-restart: installs the wheels listed in
    `CURRENT` on the share (`--no-deps`: monorepo wheels carry unresolvable
    path metadata; external deps freeze at bootstrap), falling through to the
    installed versions when the share is unreachable.
  - `deploy/fleet_status.bob` — Phoebus fleet screen (version / heartbeat /
    confirm-dialog restart per host).
  - `DEPLOYMENT.md` rewritten as the scripted runbook (bootstrap, rollout,
    smoke, instance-PV table).

## [0.1.0] — 2026-07-25

### Added

- **Initial package: the PVA peer of GeecsCAGateway, serving GEECS camera
  images as NTNDArray PVs from the camera servers themselves.** Verified
  architecture from the 2026-07-25 live pilot (UC_Amp2_IR_input at 1–5 Hz over
  VPN).
  - `PvaGatewayConfig.from_geecs_experiment` — DB-scoped served set: enabled
    devices on this host's endpoint IP with image-typed variables (via the
    gateway's `effective_vartype`, including the choice-descriptor quirk).
  - `GeecsPvaGateway` — one process, N cameras: **per-variable** gated +
    supervised GEECS TCP subscriptions (start on a variable's first PVA
    client / stop on its last; reconnect with 0.5→30 s backoff on socket
    drops), IMAQ decode off the event loop, latest-wins frame slots, GEECS
    timestamp ladder (`acq_timestamp` → `systimestamp`, LabVIEW→Unix),
    startup collision guard on normalized PV names, and per-instance
    `version`/`heartbeat` identity PVs.
  - CLI `geecs-pva-gateway --experiment NAME [--host IP] [--devices A,B]
    [--list]`.
  - Hermetic test suite: binary wire-format push server + isolated PVA server
    (`isolate=True`), covering frame flow, timestamps, gating both directions,
    and reconnect-after-drop.
