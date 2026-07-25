# GeecsPvaGateway — Developer Context for Claude

The **PVA peer of GeecsCAGateway**: a pvAccess server exposing GEECS camera
images as NTNDArray PVs. Same access-layer doctrine as the CA gateway (DB is
the source of truth, `pv_naming` is the one naming policy, GEECS wire protocol
via the gateway's transport), but deployed **distributed** — one instance per
Windows camera server, serving only that host's cameras — because a central
process relaying ~100 cameras is a bandwidth bottleneck (`GeecsCAGateway/
DESIGN.md`: "images stay off CA; data stays at the edge").

```
LabVIEW camera device --loopback TCP push--> this gateway --NTNDArray--> Phoebus / ophyd-async / p4p
```

The central CA gateway and these instances are peers in one flat namespace:
CA/PVA search finds whichever server owns a PV; nothing proxies pixels.

## Package Layout

```
geecs_pva_gateway/
  __main__.py   # geecs-pva-gateway --experiment NAME [--host IP] [--devices A,B] [--list]
  config.py     # CameraSpec / PvaGatewayConfig; DB-scoped served set
                #   (enabled devices on this host's IP with image-typed vars)
  server.py     # GeecsPvaGateway + per-camera worker: gated + supervised
                #   subscription, decode off-loop, latest-wins posting,
                #   version/heartbeat instance PVs
tests/
  test_config.py  # scoping/naming units (fake DB rows, no network)
  test_server.py  # end-to-end over a binary wire-format fake camera +
                  #   isolate=True PVA server
```

## Architecture (one asyncio loop)

- **Per-camera worker, per-variable subscriptions**: the worker owns the
  device's image `SharedPV`s; each image variable gets its own
  `GeecsTcpSubscriber` (loopback in production), **gated per variable** — p4p
  `onFirstConnect`/`onLastDisconnect` refcount client channels; zero clients
  on a variable ⇒ no subscription, no flatten/send in LabVIEW, no decode
  here. Watching `image` never costs anything for `processed image`.
- **Collision guard**: PV naming is lossy (normalization), so `run()` refuses
  to start if two (device, variable) sources land on one PV name — same
  doctrine as the CA gateway's manifest guard.
- **Supervision**: while gated on, a supervisor loop reconnects with
  exponential backoff (0.5→30 s) whenever `wait_disconnected()` returns —
  actual socket drops only; silence is not a drop (same doctrine as the CA
  gateway's device supervisors).
- **Frame path**: push frame → timestamp ladder (`acq_timestamp` →
  `systimestamp`, LabVIEW→Unix, else receive time) → **latest-wins slot** per
  variable → decode (`decode_imaq_image_string`) in the default executor, off
  the event loop → `pv.post(image, timestamp=...)`. A stalled consumer drops
  stale frames; nothing ever backlogs. Completeness lives in the GEECS file
  path, not this stream.
- **Identity PVs**: `{experiment}:pvagateway:{host_token}:version|heartbeat`
  per instance — the fleet screen reads these (version skew, liveness).

## Ground rules

- **Naming**: only via `geecs_ca_gateway.pv_naming`. No local copies.
- **Transport**: only `geecs_ca_gateway.transport`. Never GEECS-PythonAPI
  (deprecated, slated for deletion).
- **Images stay off the CA gateway; scalars stay off this one.** This package
  serves image-typed variables only. If PVA scalars ever happen, that is a
  deliberate design step (per-device-class PVA adoption, DESIGN.md), not a
  drive-by addition here.
- **Text variables**: image variables must always be subscribed as
  `text_variables` — numeric coercion destroys binary payloads.
- The wire format is binary-hostile in known ways — decode quirks live in
  `geecs_data_utils.io.images` (name-repeat vs tail-anchored wrappers), and
  the latin-1 byte↔str convention comes from the gateway transport (0.16.1).
  Do not re-derive either here.
- Repo-wide conventions apply (root `CLAUDE.md`): Pydantic v2, NumPy
  docstrings, `poetry version` + `CHANGELOG.md` on every code-changing PR.

## Testing

```bash
cd GeecsPvaGateway
poetry install
poetry run pytest tests -q   # offline; fake binary push server + isolated PVA
```

The fake camera in `test_server.py` is local to the tests deliberately: the
shared `FakeGeecsServer` is ASCII-only, and these tests need binary image
payloads on the wire.

## Deployment

`DEPLOYMENT.md` is the Windows camera server runbook. The two hard-won rules:
services (session 0) cannot see per-user mapped drives, so the GEECS config
chain must resolve through a **local** `user data\Configurations.INI`; and
Windows never kills orphaned processes, so lifecycle belongs to the service
manager (NSSM), not to whoever launched the process.
