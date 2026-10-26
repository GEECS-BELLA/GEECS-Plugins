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
