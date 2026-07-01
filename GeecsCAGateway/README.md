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

Built on the GEECS transport core (`geecs_bluesky.transport`): `GeecsUdpClient`
for sets, `GeecsTcpSubscriber` for the readback stream. The ophyd-async device
layer is **not** used — the caproto PV layer is a sibling presentation over the
same transport, parallel to the ophyd `SignalBackend`.

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

## Status — proof of concept

Working: dynamic PVs from a `GatewayConfig`, stream→readback, caput→UDP set,
units/precision metadata, offline fake-server tests.

Deliberately deferred (the honest gap list):

- **Reconnect supervisor** — `GeecsTcpSubscriber` exits on a dropped connection;
  surviving a device power-cycle needs a supervising retry loop. This is the one
  piece that earns the "robust as the legacy SVE tool" claim.
- **DB-driven config** — specs are hand-built; the next step is generating them
  from the GEECS database dict + attributes DB (units/precision/limits).
- **Full metadata contract** — alarm limits, deadbands beyond units/precision.
- **Sharding + systemd** — one central process today; shard by subsystem for
  fault isolation in production (CA name resolution makes this invisible to
  clients).
- **Images stay off CA** — scalars/controls only, by design.
