# GEECS → EPICS CA Gateway — Design Note

*Status: proof of concept, verified end-to-end against real hardware (2026-07-01).*
*Home: this note lives with the package; the strategic framing belongs alongside
the bluesky planning docs (`geecs-plugins-bluesky/…/Planning/`) — copy it there
when that tree is the working context.*

## Problem

GEECS has no standard, network-addressable **access layer**. Every consumer —
the ophyd-async/Bluesky backend, continuous archiving, any GUI — opens its own
TCP/UDP connections and re-solves liveness, reconnection, and type mapping. Each
new capability is a *new bespoke bridge*. The recurring pain (archiver bridge,
bespoke ophyd Signal backend, hand-rolled reconnection in the legacy SVE tool)
are all symptoms of this one gap. EPICS (CA/pvAccess), NI-PSP/SVE, OPC UA, and
Tango each *are* that layer off the shelf.

## The fork

- **Path A — bespoke ophyd backend per consumer.** Where we started. Every new
  need (archiver, GUI, alarms) is another bespoke bridge onto GEECS TCP/UDP.
  N integrations, N reconnection stories, N maintenance burdens.
- **Path B — one standard gateway.** A single service holds connections to GEECS
  devices and re-publishes them over a standard protocol. Bluesky, the archiver,
  and GUIs all become *off-the-shelf clients*. The bespoke surface collapses from
  N integrations to one well-defined bridge.

Something must always speak GEECS TCP/UDP — that shim is irreducible. The only
question is whether it is **one gateway or N integrations**. Path B concentrates
all the bespoke code in one testable place.

## Decision: Path B via caproto (EPICS Channel Access)

EPICS is the highest-leverage standard here because ophyd-async's primary,
best-supported backend *is* EPICS — a GEECS→EPICS gateway puts Bluesky on the
community golden path **and** makes the EPICS Archiver Appliance + Phoebus/CSS
work for free.

The usual blocker — "we have no EPICS engineers" — does not apply, because we are
**not** doing classic IOC engineering (`.db`/asyn/sequencer/C build). **caproto**
lets us write a soft IOC as pure-Python classes over the GEECS Python client. We
write Python we fully own; we inherit the EPICS ecosystem. That is the sweet spot
for a solo, spare-time maintainer.

## Architecture

The GeecsBluesky transport core is already ophyd-free (`transport/` imports no
ophyd). The gateway is a **sibling presentation** over that same core, parallel
to the ophyd `SignalBackend`:

```
                 geecs_bluesky.transport (async core, ophyd-free)
                   GeecsUdpClient · GeecsTcpSubscriber
                    /                              \
   backends/geecs_signal_backend.py         GeecsCAGateway (this package)
        (ophyd presentation → Bluesky)      (caproto PV presentation → CA)

GEECS device  --TCP 5 Hz stream-->  readback PV   (caget / camonitor)
GEECS device  <--UDP set---------  setpoint PV   (caput)
```

- **Reads** come from the subscription stream into a per-PV cache; `caget` serves
  the cache, `camonitor` pushes per acquisition. The blocking "wait for new
  acquisition" is the *event source*, already inverted to a callback by
  `GeecsTcpSubscriber` — no polling.
- **Timestamps** are pulled from GEECS, not gateway-receive time. Each frame is
  stamped via a ladder (`DeviceSpec.timestamp_vars`): `systimestamp` (universal,
  LabVIEW 1904 epoch → Unix by subtracting `2_082_844_800`) by default, with
  `acq_timestamp` prependable for triggered devices (same VI/epoch, confirmed).
  Verified on real hardware.
- **Whole-experiment config** comes from `GatewayConfig.from_geecs_experiment`,
  which enumerates the experiment's `enabled` devices from the DB and builds a
  spec for each — three batched queries total (endpoints, variable metadata,
  get-list), so startup config is seconds, not the ~80 s the original
  2-queries-per-device implementation took for Undulator's 114 devices.
- **Types are DB-driven too.** GEECS `variabletype` maps to the PV type
  (`numeric`→float, `string`/`path`→string, `choice`→enum with options from the
  `choice` table); `image`/`1darray` are skipped. So the declarative overlay is
  now needed *only* for optional curation (hiding a variable, renaming), not for
  types — types come free from the DB.
- **The served set is DB-driven too** (`subscribed_only`, default on). Rather
  than every device-type variable, expose each device's `get='yes'` variables
  from `expt_device_variable` — the experiment's per-shot monitoring subset
  (~377 vs ~8600 for Undulator). That table's `set`/start/end fields are scan
  orchestration and are *not* used here; PV writability still comes from
  `devicetype_variable`. This is the sensible-default down-select; the overlay
  remains for finer curation only.
- **Writes**: a `:SP` setpoint PV whose `ChannelData.write` override forwards the
  value to GEECS over UDP before storing it. A failed set raises, so CA put
  failure semantics are correct.
- **Config is DB-driven**: `DeviceSpec.from_geecs_db(name)` pulls host/port +
  per-variable units → EGU, min/max → CA control limits, and the settable flag
  straight from `GeecsDb`. No manual per-variable binding (the SVE tool's tax).

### Deliberate scope boundaries

- **Scalars/controls only. Images stay off CA** — keep camera data on its
  existing path.
- **Shot correlation is not the gateway's job.** EPICS PVs are independent by
  design; per-shot bundling is what Bluesky's `trigger`→`read` is for, and
  archiving does not need it.
- **Non-float types**: the GEECS DB carries no scalar type, so variables default
  to `float`; enums/addresses must be named via `dtypes=` or excluded via
  `include=`.

## Fitness for our regime (~1 Hz, ~100 devices)

The axes where a real distributed EPICS facility earns its complexity — hard
real-time determinism, 10k–1M PV scale, 24/7 fault isolation, hardware-sourced
timestamps — are precisely the ones a 1 Hz / 100-device campaign experiment does
**not** stress. 100 updates/s through asyncio is trivial; a central process is
appropriate. CA name resolution is location-transparent, so we can start central
and later shard by subsystem (for fault isolation) **without touching any
client**. The gap to a real facility is widest where we don't care and narrowest
where we do (clean scalar control/readback/archiving at human timescales). This
is a right-sized bar, not a lowered one.

The honest caveat: the gateway is a **proxy**, so it inherits proxy consistency
edges (staleness, write races with GEECS-native clients). At 1 Hz, with GEECS
remaining the operational surface and the gateway serving *tooling*, those edges
are narrow and livable.

## What is proven (2026-07-01)

Against real device `U_S1H` (steering-magnet PS, beam off):

1. `GeecsDb` → host/port + variable metadata; `from_geecs_db` builds the spec.
2. Read path: live readback of `Current`/`Voltage` from the 5 Hz stream.
3. Write path: `caput Current:SP 1.0` → readback tracked to 0.9999 A, `Voltage`
   rose 0.0034→0.634 V (coil energized — real physics), then clean restore.
4. Real CA wire exercised with caproto's `caproto-get`/`caproto-put` CLI tools;
   15 offline tests against the in-process `FakeGeecsServer`.

## Foundational decisions (settled 2026-07-01)

Recorded so the next build phase doesn't relitigate them.

- **Goal is ecosystem modernization, not just archiving.** The archiver was one
  motivating example; the target is a modern controls/data ecosystem
  (ophyd/Bluesky, Phoebus, alarms, save/restore, archiver). This raises the value
  of correct semantics (naming, timestamps, alarm/validity, types), since every
  tool consumes them.

- **Protocol: Channel Access now; pvAccess later, per-device-class.** CA is the
  stable, universally-supported substrate. PVA/structured data
  (NTScalar/NTTable/NTNDArray) is additive, adopted where it pays — starting with
  images. `channels.py` is the only caproto-typed layer, kept swappable; `p4p` is
  the mature PVA-server route if/when needed. (`p4p` is the PVA Python library,
  not "the new pyepics"; pyepics stays CA-only and current.)

- **Images are a separate, distributed, PVA workstream — not this gateway.** A
  central gateway funneling ~100 cameras is a bandwidth bottleneck. Images belong
  on distributed per-camera IOCs (areaDetector-style, PVA/NTNDArray) where data
  stays at the edge. CA name resolution lets the central scalar gateway and
  per-camera IOCs coexist transparently.

- **Naming policy (retrofit-expensive — locked first):**
  - Namespace `[Experiment:]Device:Variable`, e.g. `Undulator:U_S1H:Current`.
    The experiment prefix future-proofs against cross-experiment collisions
    (experiments are already separated in the DB).
  - Character mapping: within a component allow only `[A-Za-z0-9_]`; `:` is the
    reserved separator; every other char (space, `.`, `-`, `()`, …) → `_`, runs
    collapsed, ends stripped. The **dot is critical** — EPICS reads `.` as the
    record/field separator, so `Trigger.Source` MUST become `Trigger_Source`.
  - Device *type* stays OUT of the name — a name is stable identity; type is
    derivable metadata (bake it in and a hardware swap renames the PV). Expose
    type as metadata if wanted, not as an address component.
  - The map is **lossy at the string level, and that's fine**: the gateway holds
    the authoritative bidirectional map (`geecs_var` ↔ PV) and publishes a
    **manifest** (PV → device/variable/kind). Don't reverse-engineer GEECS names
    from PV strings; use the manifest.
  - **Collisions** (two GEECS names → one PV) are a hard error, never silent.

- **Authorization/write-safety is NOT foundational here.** GEECS enforces the DB
  value limits server-side and returns an error the setter propagates (→ CA put
  fails correctly), so the gateway inherits range safety; the DB-derived CA
  control limits are a UX hint with GEECS as backstop. A client commanding an
  in-range value is "how things are" today — not a regression. Residual: serve CA
  only on the intended subnet (`EPICS_CAS_INTF_ADDR_LIST`); optional read-only
  mode. *Operational-envelope interlocks* are a genuine future EPICS/Bluesky
  benefit (calc/alarm records, suspenders, golden save/restore), not something we
  have today — deferred, nothing lost.

- **Config source of truth = the DB, not a static file.** The DB is edited as
  devices are added/removed from experiments (the master LabVIEW GUI already
  triggers a reboot on such changes). `from_geecs_experiment(name)` pulls the
  device/variable set **live from the DB**; a thin optional **overlay file** holds
  only curation/policy (naming overrides, dtype/enum choices, exclusions,
  write-enable). On a DB change the gateway re-pulls (a restart fits the existing
  reboot pattern; hot-reload is a later nicety). The served PV set stays in sync
  with the DB rather than going stale.

## Honest gaps / next steps

- **Reconnect supervisor** — `GeecsTcpSubscriber._listen_loop` exits on a dropped
  connection; surviving a device power-cycle needs a supervising retry loop. This
  is the piece that earns "as robust as the legacy SVE tool."
- **Fuller metadata contract** — alarm limits (HIHI/…) and archive deadbands,
  beyond the units/precision/control-limits already wired.
- **Archive-rate control belongs in an archive event mask, not the value
  deadband.** Through 0.5.0 the readback monitor deadband inherited the DB
  `tolerance` to pre-limit future Archiver Appliance volume — but that
  suppressed real sub-tolerance motion from every value monitor, i.e. from
  recorded scan rows and s-files (observed live on a magnet PSU; fixed in
  0.5.1 by defaulting the deadband to 0, with exact-repeat suppression
  keeping static channels silent). The archiving concern remains valid and
  has a standard EPICS answer: the MDEL/ADEL split. When the Archiver
  Appliance lands, either (a) rely on its per-PV sampling policies +
  retention decimation (zero source changes, the common practice), or
  (b) post `DBE_LOG` events gated by the DB tolerance while `DBE_VALUE`
  events carry every change — one PV, two audiences, both truthful for
  their purpose.
- **Sharding + systemd** for production fault isolation.
- **A `GatewayConfig.from_geecs_experiment(name)`** that enumerates a whole
  experiment's devices from the DB dict.
- **Extract `transport/` + `db/`** into a shared base package that both the ophyd
  backend and this gateway depend on (currently a `path`-dep on GeecsBluesky).
```
