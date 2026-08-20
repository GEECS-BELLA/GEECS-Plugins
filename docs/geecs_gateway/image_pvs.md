# Camera images over PVA — client orientation

GEECS camera images are served live as EPICS **pvAccess (PVA)** PVs of the
standard **NTNDArray** type by **GeecsPvaGateway** — a distributed service
running on each Windows camera server, serving only that host's cameras.
Any PVA-speaking consumer — a stock Phoebus *Image* widget, three lines of
`p4p`, an ophyd-async signal — gets typed pixel arrays with normalized
timestamps, with zero knowledge of GEECS internals.

```
LabVIEW camera device --loopback TCP--> per-server gateway --NTNDArray--> Phoebus / p4p / ophyd-async
```

The central [GEECS CA gateway](client_overview.md) (scalars/controls) and the
per-server image gateways are **peers in one flat namespace**: CA/PVA name
search finds whichever server owns a PV, and nothing proxies pixels through a
central box — image bandwidth stays at the edge, by design.

This page is a client-facing orientation. The authoritative detail lives in
the package alongside the code — `GeecsPvaGateway/DEPLOYMENT.md` (fleet
runbook, addressing) and `GeecsPvaGateway/CLAUDE.md` (architecture); treat
those as the source of truth if anything disagrees.

## Why not just read the camera's TCP stream yourself?

You could — the GEECS wire format is decodable, and for a one-off diagnostic
script running *on the camera server itself* that remains legitimate. For
everything else, the gateway exists so that the hard parts are solved exactly
once:

- **The wire format is a hazard.** Values that can't be comma-tokenized,
  latin-1 byte-mangling, and IMAQ image wrappers with multiple structural
  variants that have silently drifted before. Every bespoke decoder
  re-inherits all of it; the gateway's decoder is fixed in one place for
  everyone.
- **Fan-out economics.** GEECS TCP push is per-connection — N direct
  subscribers make LabVIEW flatten and send every frame N times. The gateway
  subscribes once per image variable and PVA fans out to any number of
  clients. Subscriptions are **gated**: a camera nobody is watching costs the
  device *nothing at all*.
- **The ecosystem is free.** Phoebus renders these PVs with a stock widget;
  ophyd-async speaks PVA natively (the door to live images in scans); a
  "bespoke image GUI" starts at one `p4p` line instead of at a socket.

## PV naming

Image PVs follow the same shared naming contract as the scalar gateway
(lowercase components joined by `:` — see
[PV naming](client_overview.md#pv-naming) for the normalization rules):

```
[experiment:]device:variable            e.g. undulator:uc_tubein:image
```

A camera device typically serves several image-typed variables
(`image`, `processed_image`, …); each is its own PV, gated independently —
watching `image` costs nothing for `processed_image`.

Each gateway instance also serves three **instance PVs** for fleet health:

```
[experiment:]pvagateway:<host_token>:version     installed package version
[experiment:]pvagateway:<host_token>:heartbeat   counter, +1 per 5 s
[experiment:]pvagateway:<host_token>:restart     write 1 → clean relaunch
```

`<host_token>` is the server's IP with dots as underscores
(`192.168.6.100` → `192_168_6_100`). A Phoebus fleet screen reading these
ships in the package (`GeecsPvaGateway/deploy/fleet_status.bob`).

## Reading images

**Phoebus**: add an *Image* widget and set its PV to
`pva://undulator:<camera>:image`. That's the whole recipe.

**Python (p4p)** — note `Context("pva")`, not `"ca"`:

```python
import time

from p4p.client.thread import Context

ctx = Context("pva")
sub = ctx.monitor("undulator:uc_tubein:image", lambda v: print(v.shape, v.dtype))
time.sleep(5)  # hold the gate open — the first update is the current cached
               # value; the first real frame lands ~1-2 s later (1 Hz camera)
sub.close()
```

!!! tip "Use a held `monitor`, not a bare `get`"

    Subscriptions are gated on client interest: the gateway only subscribes
    to the camera while at least one client channel is open, and the first
    *fresh* frame arrives one gating round-trip after connecting (subscribe +
    next device push, ~1–2 s at 1 Hz). A bare `get` returns immediately with
    whatever is cached — the `(1, 1)` startup placeholder on a fresh
    instance, or a **stale last frame** from a previous watch — and never
    waits for that round-trip. Hold a `monitor` open for at least one push
    interval.

**ophyd-async** consumes the same PVs as standard PVA signals — live camera
frames become readable device signals with no GEECS-specific code.

## Addressing — who needs a PV address list

PVA name search is UDP broadcast, and broadcast is **subnet-local**. The
camera-server fleet spans several lab subnets, so:

- A client on the *same subnet* as a camera server finds it with zero config.
- **Any client that wants cameras across the whole fleet** — control-room
  machines included, and all VPN/routed clients — needs the full fleet
  address list for unicast search:
    - Python / p4p / ophyd-async: `EPICS_PVA_ADDR_LIST="<fleet list>"` plus
      `EPICS_PVA_AUTO_ADDR_LIST=NO`
    - Phoebus: `org.phoebus.pv.pva/epics_pva_addr_list=<fleet list>` in the
      settings file (environment variables don't reach a macOS
      `open`-launched app)

The roster of record for the fleet list is `HOSTS` in
`GeecsPvaGateway/deploy/gen_fleet_status.py`; the current expansion is kept
in `GeecsPvaGateway/DEPLOYMENT.md` §Client access. The CA variables
(`EPICS_CA_*`) belong to the scalar gateway and are unaffected — a client
using both gateways sets both families.

## Semantics worth knowing

- **Latest-wins, never backlogged**: a slow consumer gets the newest frame
  and skips stale ones; nothing queues. The live stream is for *watching* —
  shot-complete data acquisition lives in the GEECS file path and scan
  system, not here.
- **Timestamps are normalized**: each frame carries the device's own
  acquisition timestamp (GEECS's LabVIEW-epoch clock converted to Unix) when
  available, falling back to receive time — so NTNDArray timestamps line up
  with the scalar gateway's convention.
- **Uptime is unattended**: instances run as auto-start Windows services
  that survive reboots and crashes, reinstall themselves from the lab's
  shared clone on every restart, and report liveness via the
  heartbeat/version PVs above.

## Reading more

- [GEECS Gateway client orientation](client_overview.md) — the scalar/control
  side of the same namespace: naming rules, readbacks vs setpoints, alarms.
- `GeecsPvaGateway/DEPLOYMENT.md` (in the source tree) — the fleet runbook:
  per-box onboarding, rollout via restart PVs, addressing detail.
- `GeecsPvaGateway/CLAUDE.md` — architecture: gating, supervision, the
  frame path.
