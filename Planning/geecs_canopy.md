# GeecsCanopy: is SLAC's Canopy our display backend?

Status: exercise brief, 2026-10-26, provisional until the lanes report; the
write-up and recommendation are appended as §7. Delete this file when the arc
it describes is finished or abandoned (`Planning/README.md`).

## 1. What is being evaluated

[slaclab/canopy](https://github.com/slaclab/canopy): the backend of SLAC's
control-system display platform, by the PyDM team, public since 2026-10-21,
pre-1.0, no release yet. One FastAPI process ("modular monolith") plus an
arq worker, Postgres 16 and Redis 7. Modules, each at `/<module>/v1/` behind
`X-API-Key` keys with `read`/`write`/`admin` and per-module scopes:
**Hare** (live PVs: WebSocket subscribe with 100 ms batched diffs and a
heartbeat, REST read, PUT with put-callback; CA via aioca, pvAccess via p4p),
**Squirrel** (a PV registry with device/setpoint/readback names, snapshots:
save, restore a subset, compare, restore hooks), **Mole** (archiver read API,
pluggable), **Raccoon** (PV directory, pluggable), Vole, Fox, Jobs, Magpie,
Converter. Image `ghcr.io/slaclab/canopy`, tags = short commit SHAs only
(`0a5ee23` = main 2026-10-23); the reference production compose runs the app
with host networking. Licence: BSD-3 with an enhancements clause (§7 says what
it means).

The question: do we build the GEECS display backend **on** Canopy, **beside**
it, or without it. The previous push (geecs-web + GEECS-Plugins #1153, parked)
built the same layer; it re-implemented what SLAC published four days earlier.

## 2. The rule: from Canopy outward

The exercise designs from Canopy's shapes toward GEECS, never the reverse.
Canopy's wire frames, key model, module layout and adapter protocols are
taken as given; GEECS is expressed as data (the family files), as adapters
Canopy already allows, and as the smallest module that adds what Canopy
lacks. The parked bridge (`Planning/epics_estate.md` Appendix A, the panels'
`CLAUDE.md` "Rules this package keeps", geecs-web `docs/mvp-bench.md`) is
reference for **rules and test cases only**: a lane may ask "does Canopy keep
rule 1?" and never "how do we make Canopy emit a DeviceFrame?". Where Canopy
falls short, the answer is a candidate upstream contribution, phrased so SLAC
could accept it, not a fork.

## 3. Questions the exercise must answer

1. Does Canopy run against our IOCs (geecs-epics' simulator host) with no
   patch: CA scalars, enums, waveforms; pvAccess; the camera's NTNDArray
   (`valueKind: image` is reserved in the source, NTNDArray maps to
   `unsupported` today)?
2. Freshness: given per-PV `tsNs`, `connected`, the 5 s heartbeat and 100 ms
   batching, can a client judge staleness by the producing plugin's counter
   (Appendix A rule 1), and what must it do on reconnect?
3. Identity and refusal: is the API key the only identity on a write; what
   does a refused put look like (`pv_write_failed` 502 vs an IOC clamp); what
   does `wait: false` hide?
4. Snapshots: does Squirrel carry our save/diff/restore-subset/unreachable
   semantics (steps 10–14 below), and what is a "device" to it?
5. Directory: can Raccoon answer device → PVs for a GEECS device from the
   GEECS DB, the native registry and the family files, through its adapter
   protocol (dotted-path load from an installed package)?
6. Device module: can a facility module mount in-process (module set is a
   fixed tuple in `main.py`, pinned by a test), or must it be a sidecar?
7. Operations: CPU/memory, what Canopy needs that we do not run today
   (Postgres, Redis, a worker, keys), the front-door prefix, the image pin
   policy for a pre-1.0 service.

## 4. The judging checklist

The behavioural bar is the previous gate's 17-step click-through (geecs-web
`docs/mvp-bench.md`, 17/18 on the sim). For each step the write-up says:
Canopy gives it as-is / the GEECS module must add it / impossible.

| # | Step | Canopy module expected to answer |
|---|---|---|
| 01 | Experiment home: roster by family, alarms, active snapshot, last commands | Raccoon + Squirrel + module (journal) |
| 02 | Camera page live, counter advancing | Hare (scalars); image path open (Q1) |
| 03 | Crosshair set and dragged (OVER1 PVs) | Hare put |
| 04 | Composed panel with `show_when` | front end (out of scope) over Hare |
| 05 | Motor tweak, journaled with operator | Hare put + module (identity, journal) |
| 06 | Hazard confirm dialog | module (family `confirm`) |
| 07 | Supply setpoint accepted, refused by bounds, refused by IOC | Hare 502 vs module bounds (rule 11) |
| 08 | `denied` from the device's allow list | module |
| 09 | Stale page: controls close when the IOC stops | Hare `connected` + heartbeat (Q2) |
| 10–14 | Snapshot save, active verdict, diff after a move, restore subset, verdict after | Squirrel (Q4) |
| 15–16 | Scan submit through preflight, pause/resume/stop | GeecsScanner, beside Canopy (not judged) |
| 17 | Three panel tabs at once without starving requests | Hare: one WebSocket per page |

## 5. The lanes (Opus, fresh sessions, one per piece)

- **C1, Canopy on the sim.** Run the pinned image with its Postgres and Redis
  on this Mac, EPICS backend at the simulator's CA ports (and the camera's
  pvAccess name server), keys from its scripts. Exercise Hare (subscribe to
  motor, supply, camera PVs; a motor put and its readback; a refused put on
  the FAST-PS sim; `wait: false`), Squirrel (register, save, move, compare,
  restore a subset, a failure on an unreachable PV), Mole (adapter interface
  only unless the archiver pilot is reachable read-only off the lab network),
  Raccoon (what the adapter needs). Output: a report of protocol facts for
  Q1–Q3 and Q7, the compose file, screenshots of Canopy's `frontend-demos`
  against our sim; Canopy left running.
- **C2, the GEECS directory adapter.** Raccoon's custom-adapter how-to, over
  `GeecsDb` (GEECS-Core), the native registry and the family files; tested
  with the connection double GEECS-Core's own DB tests use; mounted into the
  C1 instance by whatever mechanism Canopy offers. Output: mounted or not,
  the protocol's fit, the lines of code.
- **C3, the device-module question.** Read how modules register; decide
  in-process vs sidecar; prototype the smallest GEECS module: one route built
  from a family file (a device's roles with Hare-sourced values and a
  freshness verdict by the producing plugin's counter) and one put path that
  adds operator identity and a journal row over Hare's put, denying a role
  outside the device's allow list. Output: mounts or not, lines of code, the
  Canopy shapes it had to extend, each as a candidate upstream contribution.

C2 and C3 start when C1 reports Canopy up. Lane outputs live under
`Planning/geecs_canopy/<lane>/` on each lane's branch off master; spike code
reaches master only if the recommendation is "adopt". A lane that cannot make
Docker, Postgres or the image work in reasonable time says so.

## 6. Out of scope

The front-end choice and the composed-panel or screen format (SLAC's Screen
Builder is not public); migration of any device; gated runs; the scanner,
portal and logbook (they sit beside Canopy either way); touching `geecs-web`
or `feat/web-estate`; contacting SLAC (a two-question note is drafted in §7
for Sam to send).

## 7. Write-up and recommendation (2026-10-26)

The three lanes ran against Canopy `0a5ee23` (digest in `geecs_canopy/c1/`),
on the sim, with no patch to Canopy. Reports: `geecs_canopy/c1/report.md`
(protocol facts), `c2/report.md` (directory adapter, 299 lines, mounted),
`c3/report.md` (device module, 399 lines, mounted in-process). The instance
is still running at `http://127.0.0.1:8000` (stop: `cd
~/.local/state/geecs-canopy-exercise && docker compose down`).

### 7.1 Canopy against the checklist

| Steps | Canopy as-is | The GEECS module must add | Impossible today |
|---|---|---|---|
| 01 home | alarms per PV over Hare; snapshot list | roster by family (Raccoon adapter), journal, "active" = snapshot-vs-live compare (Squirrel compares snapshot to snapshot only) | — |
| 02 camera live | scalars and counters at 100 ms, latest-wins | the viewer stays on the IOC's MJPEG path | image frames over Hare: NTNDArray is monitored at full rate and emitted as `null` (reserved `image` kind) |
| 03, 04 | Hare put; Hare | front end | — |
| 05 tweak + operator | put-callback 200 at motion end | operator and journal: Canopy's only identity is the key and it logs no write (built: C3) | — |
| 06 hazard | — | family `confirm` + a confirm field on the module's put | — |
| 07 refusals | IOC refusal = 502 `pv_write_failed`; device NAK = 200 + INVALID (rule 11 holds) | a bounds gate: a put beyond DRVH is a silent 200, clamped; unset limits read `0.0`, so fail closed | — |
| 08 denied | — | per-device allow list, 403 before Hare (built) | — |
| 09 stale | `connected:false` in one diff ≈70 ms after the IOC dies; reconnect automatic | freshness by the producing counter and the `Compute*` flag (built, polled; streaming version = an internal broker client) | — |
| 10–14 snapshots | registry with device/setpoint/readback, save, restore a subset with put-callback, tolerances, snapshot-vs-snapshot compare | per-device scope (save = every registered PV), live compare, restore ordering and confirm, unreachable rows (saved as `null`, job "completed") | — |
| 15–16 scanner | beside Canopy | — | — |
| 17 three tabs | one WebSocket per page; 4 clients, 10 monitors, no starvation | the front end multiplexes per page | — |

Protocol facts that bind any client: `tsNs` is the IOC timestamp and monitors
are value-change only, so age never judges staleness; `diff` may precede
`initial`, and only `initial` carries `meta`; there is no resume (a new
socket re-subscribes); `wait:false` returns the same body as a confirmed put;
5000 PVs per subscribe, 20 000 per client. Hare and Squirrel otherwise behave
as their docs say; C1 §6 lists nine small doc/behaviour mismatches.

### 7.2 Deploy implications on the services box

New runtime: Postgres 16 (one schema per module, `alembic upgrade head`
before every image change; an older image cannot run on a newer schema),
Redis 7 and the arq worker (snapshots run as jobs: a dead worker means no
snapshots), the api, API keys (two env keys, the rest minted into Postgres by
`scripts/create_key.py`, cached, revocation not instant). Image: amd64 only
(the services box is x86; the Mac ran it emulated), 614 MB, api idle 150 MiB;
production compose is host networking, which the lab box supports natively.
Front door: `/canopy` prefix with `--root-path`; the WebSocket at
`/canopy/hare/v1/ws/pvs`; CORS is `*` hard-coded. Pin policy for a pre-1.0
service: pin the digest (tags are short SHAs of main; no releases exist),
record it in `site.env`, rehearse each bump on the sim first, and never
`latest`. `CANOPY_EPICS__BACKEND` defaults to `simulation`, so the heartbeat's
`backend` is a deploy check. Our code mounts as a bind-mounted pure-Python
directory on `PYTHONPATH` or a derived image; GEECS-Core's Python ceiling
(`<3.12`) must be relaxed to run inside Canopy's 3.12.

### 7.3 The licence, in plain words

BSD-3 (Stanford/SLAC) plus one clause: we owe SLAC nothing, but any
"Enhancement" (fix, patch, upgrade) we make available publicly or give to
SLAC without a separate written licence is thereby licensed to everyone,
royalty-free, forever, to use, modify, redistribute and sublicense. Since
GEECS-Plugins is public, anything in a `GeecsCanopy` package that is a
change to Canopy is in effect BSD-licensed; our adapter and module, which
only import Canopy's protocols, are ours under our own licence, and the
upstream contributions below are meant to be given anyway. Nothing restricts
private modification. Compatibility with GEECS-Plugins' own licence is Sam's
check, not an agent's.

### 7.4 Candidate upstream contributions (from the lanes; none filed)

1. A module registry by entry point, and key scopes derived from it (today
   an extra module needs `@*` keys). 2. `hare.gateway.put` and `subscribe`
   as declared surface (the module uses `get_broker()` internals). 3. A
   write audit: optional `operator` on `PUT /hare/v1/pvs` and a put-observer
   hook with key name, value, outcome; Squirrel's `createdBy` from the key.
   4. NTNDArray as the reserved `image` kind, and no full-rate monitor until
   then. 5. A put-callback timeout distinct from the 5 s connect timeout.
   6. Open error codes; per-module Alembic trees; named env keys. 7. Raccoon:
   a structured device → PVs call, the conformance suite shipped in the
   package, an adapter-declared pattern grammar. 8. Snapshot-vs-live compare
   and per-scope save in Squirrel; unreachable rows marked. 9. An arm64 image.

### 7.5 Questions only SLAC can answer (the note for Sam to send)

Subject: Canopy at BELLA, two questions before we build on it. We run GEECS
at LBNL's BELLA Center, moving devices to native EPICS IOCs and operator GUIs
to the browser. We have Canopy (0a5ee23) running against our simulated IOCs
and are evaluating it as our display backend, with a Raccoon adapter over our
device database and a small facility device module (device grouping from
per-record-template role files, freshness by the producing plugin's counter,
per-device write allow lists, operator identity and a journal over Hare's
put), mounted in-process by a wrapper app. (1) Is a facility module welcome
in-process: would you take an entry-point module registry with key scopes
derived from it, and a declared `hare.gateway.put`, or do you intend site
logic to live in sidecars over the HTTP/WS APIs? (2) When do you expect the
Screen Builder front end to be public, and is NTNDArray over Hare (the
reserved `image` kind) on the roadmap? We would contribute what we build.

### 7.6 Recommendation

**Adopt Canopy, with a `GeecsCanopy` package in GEECS-Plugins on the
GeecsArchiver pattern**: a deploy recipe (digest-pinned image, Postgres,
Redis, worker, keys, the `/canopy` prefix) plus `geecs_canopy/` holding the
Raccoon adapter and the device module, run in-process through the wrapper
app, seeded from the C2 and C3 spikes. Why: Canopy gives, unpatched, what the
parked bridge spent most of its 2.4k lines on (live PVs, puts with
put-callback, metadata, keys, caps, metrics) plus the three things we never
built (snapshots with a worker, an archiver read API that already speaks our
stock Appliance, a PV directory), and our part came to ~700 spike lines, all
data-driven by the family files. The maintainability bet is that SLAC's PyDM
team keeps the generic layer and we keep only GEECS semantics.

Conditions and guards: (a) one file in `geecs_canopy` lists every
Canopy-internal import (`canopy.main:app`, `tag_module`, the security
dependencies, `get_broker().put`), pinned to the image digest and checked by
a startup self-test, so a Canopy bump fails loudly; (b) `routes.py` keeps no
Canopy import (as C3 built it), so if SLAC answers "sidecar", the switch is
`app.py` alone; (c) Sam sends the §7.5 note before the package is deployed
anywhere; (d) the first package PR carries the deploy recipe and the adapter,
the device module follows with the bounds gate, hazard confirm and run lock
from Appendix A, each pinned by a test; (e) a GEECS-Plugins Python-ceiling
relaxation for GEECS-Core. Not chosen: *sidecar only* keeps one hop and a
duplicated key model for no gain (module scoping is lost either way until
upstream 1); *do not adopt* means rebuilding the parked bridge's generic
layer plus snapshots, keys and an archiver read API ourselves, the cost the
previous push already showed. Front end, panel format and migration remain
out of scope; whatever front end comes speaks Hare's WebSocket and the
module's routes.

### 7.7 Review dispositions

Pending: one fresh-context adversarial review (architecture and operations
lenses; severity floor: findings that change the recommendation).
