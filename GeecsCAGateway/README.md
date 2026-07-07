# GeecsCAGateway

An EPICS **Channel Access soft-IOC gateway** that exposes GEECS devices as PVs,
so the standard EPICS ecosystem — ophyd-async/Bluesky, the Archiver Appliance,
Phoebus — can talk to GEECS without a bespoke bridge per consumer.

```
GEECS device  --TCP stream-->  readback PV   (caget / camonitor)
GEECS device  <--UDP set-----  setpoint PV   (caput)
```

## What it is (and isn't)

It is a **façade** over GEECS, not a native EPICS control system. GEECS remains
authoritative; this gateway mirrors device variables as PVs and forwards puts.
That proxy nature is deliberate and appropriate for ~1 Hz / ~100-device
experiments, where the axes real EPICS distribution buys (hard real-time, huge
PV counts, 24/7 fault isolation) are not stressed.

This package is the self-contained GEECS access layer: the UDP/TCP wire
protocol (`geecs_ca_gateway.transport`), the experiment database
(`geecs_ca_gateway.db`), the PV naming contract, and the CA server. The
Bluesky side consumes the PVs as a service (stock ophyd-async EPICS signals)
and imports only the library parts (`GeecsDb`, `pv_naming`).

## Documentation

- **[`PV_CONTRACT.md`](PV_CONTRACT.md)** — the normative contract every CA
  client relies on: PV naming, readback/setpoint semantics, timestamps and
  the frame-ordering guarantee, type mapping, alarm policy, collision and
  failure semantics. Every claim is pinned by a test.
- **[`DEPLOYMENT.md`](DEPLOYMENT.md)** — launching the gateway (CLI, config
  resolution), the current dev deployment and its risks, the client-side
  recipe (including Windows), smoke tests, and the target systemd shape.
- **[`DESIGN.md`](DESIGN.md)** — why the gateway exists (one standard access
  layer vs N bespoke bridges), the caproto decision, foundational decisions,
  and the honest gap list.

## Try it (offline, no hardware or lab network)

```bash
poetry install
poetry run python -m geecs_ca_gateway.demo
```

The demo runs an in-process `FakeGeecsServer`, self-checks both data paths, then
serves PVs you can poke with real CA tools (`caget`/`camonitor`/`caput`).

```bash
poetry run pytest        # offline tests (fake server)
```

## Status — in production use

Working and load-bearing: DB-driven whole-experiment config
(`python -m geecs_ca_gateway --experiment NAME`), stream→readback with GEECS
timestamps, caput→blocking UDP set, per-device reconnect supervisors with
INVALID/`CONNECTED` liveness, units/precision/limits metadata, offline
fake-server tests. Bluesky GUI scans and Phoebus displays consume it live.

Deliberately deferred (the honest gap list — details in `DESIGN.md`):

- **Full alarm metadata** — no value-based alarms (HIHI/…) yet; severity means
  liveness only (`PV_CONTRACT.md` §5).
- **Archive-rate control** — MDEL/ADEL split when the Archiver Appliance
  lands; the value deadband stays 0.0.
- **Sharding + systemd** — one central process today; target shape sketched in
  `DEPLOYMENT.md` §5.
- **Images stay off CA** — scalars/controls only, by design.
