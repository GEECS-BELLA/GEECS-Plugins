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

## Honest gaps / next steps

- **Reconnect supervisor** — `GeecsTcpSubscriber._listen_loop` exits on a dropped
  connection; surviving a device power-cycle needs a supervising retry loop. This
  is the piece that earns "as robust as the legacy SVE tool."
- **Fuller metadata contract** — alarm limits (HIHI/…) and archive deadbands,
  beyond the units/precision/control-limits already wired.
- **Sharding + systemd** for production fault isolation.
- **A `GatewayConfig.from_geecs_experiment(name)`** that enumerates a whole
  experiment's devices from the DB dict.
- **Extract `transport/` + `db/`** into a shared base package that both the ophyd
  backend and this gateway depend on (currently a `path`-dep on GeecsBluesky).
```
