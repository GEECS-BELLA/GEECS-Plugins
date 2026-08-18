# Changelog — geecs-pva-gateway

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.1] — 2026-08-18

### Changed

- Docs truth-up after the fleet rollout completed (2026-08, 9 boxes at
  uniform 0.4.0): `DEPLOYMENT.md` drops the canary/pilot phase framing
  (fleet-wide is the current state; per-host restart example; machine-account
  share access stated as production-validated with the domain-account
  fallback demoted to contingency; `HOSTS` defined as the deployed-instance
  roster), and `CLAUDE.md` gains the load-bearing-production statement plus
  the missing `gen_fleet_status.py` / `test_entrypoint.py` layout entries.

## [0.4.0] — 2026-07-31

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
