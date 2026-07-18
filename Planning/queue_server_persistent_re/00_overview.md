# QueueServer and Persistent RunEngine — Overview

**Status:** planning
**Branch model:** long-running experimental GeecsBluesky branch, with small
independent pieces split out when useful
**Started:** 2026-06-30

This document captures the path from the current GUI-owned
`BlueskyScanner` bridge to a persistent Bluesky QueueServer deployment for
GEECS. The goal is to make the Bluesky backend look like a normal beamline
control service: one scan authority owns the RunEngine, the scan-critical
ophyd-async objects, and the GEECS device clients; GUI and notebook clients
submit work to that authority instead of creating hardware-facing objects
themselves.

Bluesky support is still experimental in this repository. During this work,
`GeecsBluesky` internals may change aggressively. The contracts to preserve are
the ones that matter outside this package: GEECS scan-folder discipline,
schema-versioned Bluesky event documents, native GEECS file-saving semantics,
and the external-asset direction already proven for native camera files.

---

## Thesis: "one scan authority, persistent device clients"

The current `BlueskyScanner` creates a `RunEngine` inside the scanner process,
then creates/connects GEECS ophyd-async devices at scan start and disconnects
them at scan completion. That is a good experimental bridge, but it is not the
desired long-term ownership model.

The target model is:

```text
GEECS Scanner GUI / notebooks / automation clients
        |
        | QueueServer API on the private lab network
        v
QueueServer / RE Manager
        |
        v
RE Worker process
  - persistent RunEngine
  - persistent GEECS device registry
  - scan plans and plan factories
  - Tiled/document callbacks
  - scan lifecycle/status reporting
        |
        v
GEECS LabVIEW device servers and native file saving
```

The RE Worker is the scan authority. It owns the RunEngine and the
scan-critical ophyd-async objects. The ophyd-async objects are **not** separate
distributed services; they are Python client proxies over the existing
distributed GEECS/LabVIEW device layer.

The GUI becomes a client. It submits a serializable scan request, monitors
status, and sends pause/resume/abort requests through QueueServer. It does not
own scan-critical UDP/TCP clients, shot-control setters, or native-save
configuration.

---

## Current State

As of this planning checkpoint:

- `BlueskyScanner` is a `ScanManager`-compatible object used by the GUI when
  Bluesky mode is enabled.
- `BlueskyScanner` creates one `RunEngine` at initialization. The RunEngine's
  internal asyncio loop persists for that scanner object.
- Detectors, motors, and shot-control UDP clients are created from the GEECS
  database at scan start and disconnected in the scan cleanup path.
- Plans are already mostly reusable: `geecs_step_scan`,
  `geecs_free_run_step_scan`, `geecs_single_shot`, `geecs_t0_sync`, and
  `geecs_run_wrapper` hold the important acquisition contracts.
- Tiled integration, scan-number claim, native save-path handling, and external
  asset document emission are implemented in the scanner path.
- `GeecsDb` is already the authoritative source for resolving device names to
  hosts, ports, and device types.

This is enough to run hardware scans, but it keeps scan authority inside the
GUI process and makes device lifetime per-scan rather than service-scoped.

---

## Target State

The target deployment has one long-lived QueueServer/RE Worker for the active
GEECS Bluesky backend.

The worker startup code should:

- create or keep the production RunEngine instance;
- subscribe Tiled/document callbacks;
- create a `GeecsDeviceRegistry`;
- expose queue-callable scan plans such as `geecs_gui_scan(request)`;
- expose only the plans and devices intended for client submission.

The device registry should:

- use `GeecsDb` as the device catalog;
- lazily resolve and connect GEECS devices requested by a run;
- keep successful connections warm when safe;
- reconnect or invalidate unhealthy clients;
- make run-specific configuration explicit and resettable.

The first production-style deployment can colocate QueueServer on the existing
Linux Tiled/GEECS-DB host because Bluesky usage is experimental and there is no
meaningful Tiled load yet. This is a phase-1 deployment shortcut, not a design
requirement. QueueServer should still run in its own environment with separate
logs/config so it can move later to a dedicated Linux control machine without a
code redesign.

---

## Locked Starting Assumptions

These are working assumptions for the first implementation. Revisit them only
with new operational evidence.

1. **GEECS DB remains the device catalog.** Do not introduce a parallel static
   device inventory unless the DB lacks information required by the registry.

2. **Ophyd-async objects live in the RE Worker.** They are client proxies used
   by Bluesky plans, not distributed device servers. The distributed hardware
   layer remains the GEECS/LabVIEW device system.

3. **Do not eagerly connect every known device by default.** A setup may have
   roughly 100 GEECS devices while a scan uses about 30. The registry should
   know how to resolve all devices but connect the subset needed by a run,
   keeping warm clients where useful.

4. **One active scan authority.** The first deployment should assume one
   QueueServer/RE Worker controls scan-critical writes. Multiple workers
   touching the same hardware require an explicit locking/authority model and
   are out of scope for the first pass.

5. **Passive archiving is out of scope.** A future ambient device archiver may
   reuse GEECS transport code, but it should not be owned by the scan RunEngine
   and should not block QueueServer deployment.

6. **Heavy nonscalar data stays out of RunEngine events.** Native GEECS file
   saving remains the data plane for images/TDMS/etc. Bluesky documents carry
   scalar metadata, timestamps, save-path context, and external asset
   references.

7. **Private-network deployment first.** The lab network/firewall is the
   primary access boundary. Use QueueServer's ZMQ API on approved local subnets
   first; defer `bluesky-httpserver`, REST auth, or domain/SSO integration until
   there is a concrete need.

8. **Tiled is a separate role even if colocated.** Tiled may share the same
   Linux host during phase 1, but QueueServer should treat it as an external
   service and remain movable.

---

## Design Pieces

### Piece 1 — Serializable scan request model

Define a queue-facing scan request model independent of the GUI's current
`ScanExecutionConfig` object graph. It should include:

- scan mode (`standard`, `noscan`, later additional modes);
- scan variable and position/wait-time settings;
- repetition-rate/acquisition-mode settings;
- save-device configuration;
- shot-control configuration reference or payload;
- optional description/user metadata;
- experiment/data-session context needed by scan-number claim.

The current `BlueskyScanner.reinitialize(exec_config)` duck-typing can remain
temporarily, but QueueServer plans need a request that can cross a process/API
boundary as plain JSON-compatible data.

### Piece 2 — Plan factory extraction

Move the scanner-specific logic that turns a scan request into motors,
detectors, metadata, and a plan into a reusable module. `BlueskyScanner` and
QueueServer startup code should both call this layer during migration.

This extraction should preserve the existing plan contracts:

- `geecs_run_wrapper` claims scan numbers and sets scan metadata;
- strict shot-control uses plan-owned single-shot when `ARMED` is available;
- free-run mode uses reference-paced event creation and t0 sync;
- native file-saving devices receive scan-specific save paths;
- external asset definitions are keyed by GEECS database device type.

### Piece 3 — Persistent `GeecsDeviceRegistry`

Introduce a registry owned by the RE Worker. It should be responsible for
device lookup, lazy construction, connection reuse, and health handling.

The registry should separate:

```text
device catalog/spec:
  device name, host, port, device type, available variables

live client:
  UDP/TCP transport, ophyd-async device object, connection state

run role/view:
  detector role, selected variables, save path, shot-id/t0 state,
  reference relationship, external asset logging
```

The registry does not need to solve all of this in one PR, but this separation
is the key to safely reusing devices across runs.

### Piece 4 — Explicit run-scoped device state

Persistent devices are only safe if run-specific state cannot leak between
runs. The first implementation must identify and reset/stage:

- selected variable list and Bluesky data keys;
- detector role (`triggered`, `reference`, `contributor`, `snapshot`);
- nonscalar save enablement and save paths;
- external asset logging roots and scan number;
- shot-id tracker state;
- t0/reference-device relationships;
- pending asset documents;
- TCP queue/cache assumptions that must start clean for a run.

If some existing device classes are too scan-shaped to make persistent safely,
use persistent lower-level GEECS clients/transports first and create ephemeral
per-run Bluesky readable views on top.

### Piece 5 — QueueServer startup module

Add a startup module or startup directory for QueueServer that:

- constructs the production RunEngine and subscriptions;
- creates the `GeecsDeviceRegistry`;
- exposes queue-callable plans;
- loads or validates configuration paths;
- writes the `existing_plans_and_devices.yaml` and permissions files needed by
  QueueServer.

For production-style QueueServer deployments, the RunEngine and subscriptions
should be defined in startup code rather than relying on minimal built-in
RunEngine configuration.

### Piece 6 — GUI QueueServer facade

Add a scanner facade that implements the GUI-facing scanner API but delegates
to QueueServer:

```text
reinitialize(exec_config) -> build/store serializable request
start_scan_thread()      -> submit/start queue item
stop_scanning_thread()   -> request pause/abort/stop through QueueServer
is_scanning_active()     -> query QueueServer/RE state
estimate_current_completion() -> derive from lifecycle/doc stream
```

The existing in-process `BlueskyScanner` can stay as a development/hardware
test backend until the QueueServer path is proven.

### Piece 7 — Deployment service configuration

Document and eventually add deployment artifacts for the lab control host:

- dedicated Python environment;
- QueueServer config YAML;
- Redis bound to localhost;
- ZMQ control/info ports exposed only to approved lab subnets;
- separate logs from Tiled/GEECS-DB services;
- systemd or equivalent process supervision;
- restart procedure that does not disturb Tiled/GEECS-DB;
- migration note for moving QueueServer to a dedicated host later.

The exact Linux host, service account, and authentication model are deferred.
For phase 1, a private-network deployment with host firewall rules is enough.

---

## Deployment Posture

### Phase 1 — Colocated experimental service

Run QueueServer on the existing Linux Tiled/GEECS-DB host.

This is acceptable because:

- Bluesky usage is experimental;
- there is no significant live Tiled load yet;
- the host already has relevant lab-network access;
- it avoids waiting for procurement or reformatting a dedicated machine.

Guardrails:

- separate environment and service definition;
- separate logs/config from Tiled and database services;
- no hardcoded assumption that Tiled/DB are localhost except in local
  deployment config;
- Redis remains local-only;
- ZMQ ports are opened only to approved lab subnets.

### Phase 2 — Dedicated control host

Move QueueServer to a modest Linux control machine once the workflow is proven
or the service becomes important enough to deserve failure isolation.

A suitable host does not need to be large: a reliable mini PC with wired
Ethernet, NVMe storage, 16-32 GB RAM, static IP/DNS, UPS, and no aggressive
sleep/update policy should be enough. Reliability and network placement matter
more than CPU.

### Avoid cloud scan authority

Cloud-hosted control is not a target. It adds VPN/routing/latency/failure modes
to the process that must own aborts, hardware writes, data mounts, and device
reachability. Cloud services may be reasonable later for dashboards or
replicated metadata, but not for the active GEECS scan authority.

---

## Validation Plan

Validation should progress from fake devices to real hardware:

1. QueueServer starts locally and opens an RE Worker with the GEECS startup
   module.
2. Fake GEECS devices can be resolved, connected, scanned, and reused across
   multiple queued plans.
3. Worker close/reopen recreates registry state cleanly.
4. NOSCAN hardware scan succeeds through QueueServer.
5. STANDARD hardware scan succeeds through QueueServer.
6. Strict shot-control scan with `ARMED` succeeds and disarms on abort.
7. Free-run mode still emits schema-v1 companion columns and reference metadata.
8. Tiled receives start/descriptor/event/stop documents.
9. Native file-saving devices write files in the expected scan folder.
10. External asset datum IDs and Resource/Datum documents are emitted for
    registered device types.
11. GUI facade can start, monitor, and abort a QueueServer-backed scan.

---

## Abort Risk

| Risk | Level | Mitigation |
|---|---|---|
| Persistent device state leaks between runs | High | Make run-scoped configuration explicit; add tests for consecutive scans with different device subsets. |
| QueueServer abort semantics are worse than current GUI bridge | Medium | Validate abort/disarm early with fake devices and then hardware. |
| GUI lifecycle/progress reporting loses needed detail | Medium | Keep current lifecycle callback behavior as a compatibility target until a better document/status stream replaces it. |
| Device registry design becomes too abstract | Medium | Start with the minimum registry needed for the current scanner path; defer broad catalog features. |
| Colocated deployment couples QueueServer to Tiled/DB load | Low initially | Keep services separate and move QueueServer to a dedicated host when load or reliability demands it. |
| QueueServer security model is underdeveloped | Low initially | Private-network deployment first; restrict ports by subnet; add QueueServer permissions/auth layers later if needed. |

---

## Out of Scope

- Passive ambient archiving of all GEECS variables.
- Analysis pipeline design and derived-run strategy.
- Replacing Tiled.
- Replacing GEECS/LabVIEW device servers.
- Making ophyd-async objects into distributed services.
- Cloud-hosted scan authority.
- Multiple independent RE Workers controlling the same hardware.
- Full domain/SSO integration for phase 1.

---

## Open Questions

- Should the first QueueServer client use the ZMQ API directly, or should a
  thin local helper hide that from the GUI code?
- What is the minimum GUI status stream needed to preserve today's
  `ScanLifecycleEvent` behavior?
- Should persistent registry entries cache full ophyd-async devices, lower-level
  GEECS transports, or both?
- Which current device state belongs in `stage`/`unstage`, and which belongs in
  an explicit `configure_for_run(...)` helper?
- How should QueueServer user-group permissions be represented while the lab
  still uses a shared domain account?
- Should the local in-process `BlueskyScanner` remain indefinitely as a
  developer/hardware-test backend, or be retired once QueueServer is stable?
- What is the right branch strategy once implementation begins: one long-lived
  branch for the full migration, or small branches for request model, registry,
  and startup code?

---

## First Implementation Slice

The most useful first slice is not systemd or authentication. It is a local
QueueServer smoke path with fake devices:

1. Add the serializable scan request model.
2. Extract a plan factory that can be called without GUI objects.
3. Add a minimal `GeecsDeviceRegistry` that resolves and caches fake/real device
   clients.
4. Add QueueServer startup code exposing one `geecs_gui_scan(request)` plan.
5. Prove two queued scans can run back-to-back with different device subsets
   without leaking run-specific state.

After that, deployment and GUI integration will have something concrete to
wrap.
