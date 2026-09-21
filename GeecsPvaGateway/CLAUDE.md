# GeecsPvaGateway — Developer Context for Claude

The **PVA peer of GeecsCAGateway**: a pvAccess server exposing GEECS camera
images — and, since 0.12.0, the devices' array variables (lineouts, scope
traces) — as NTNDArray PVs. Same access-layer doctrine as the CA gateway (DB is
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

This is **load-bearing production infrastructure, not a prototype**: deployed
fleet-wide for Undulator since 2026-08 (one NSSM instance per active camera
server; Phoebus camera screens and any p4p/ophyd-async client consume it live).
Treat its externally observable behavior — PV names, NTNDArray shape,
instance-PV semantics — as a contract.

## Package Layout

```
geecs_pva_gateway/
  __main__.py   # geecs-pva-gateway --experiment NAME [--host IP] [--devices A,B] [--list]
                #   + `geecs-pva-gateway fleet` (read-only fleet probe, fleet.py)
  config.py     # DeviceSpec / PvaGatewayConfig; DB-scoped served set
                #   (enabled devices on this host's IP with a stream variable:
                #   image-typed, or 1darray-typed and not excluded by
                #   geecs_core.db.device_streams — exclusions + padding
                #   ceilings are GEECS-Core's per-devicetype declaration)
  streams.py    # per-variable decode + shape: IMAQ for an image; the three
                #   array wire shapes (geecs_data_utils.io.arrays, sniffed)
                #   → float64 in physical units, NaN-padded along axis 0 to
                #   the devicetype ceiling (longer = ArrayTooLongError,
                #   dropped + counted, never truncated); a waveform's
                #   x0/dx/samples ride as NTNDArray attributes
  fleet.py      # fleet roster: camera servers from the DB per experiment,
                #   each marked deployed by config.ini [pva] addr_list
                #   (absent = not deployed: no instance, not an outage);
                #   probe_fleet/fleet_main = the `fleet` subcommand that
                #   scripts/fleet_status.sh calls (lines + one role= record)
  server.py     # GeecsPvaGateway + per-camera worker: gated + supervised
                #   subscription (+ :connected state per variable, DB endpoint
                #   re-resolve at the backoff ceiling, #854), decode off-loop,
                #   latest-wins posting (the last decoded frame is kept for
                #   the plugin's arm, #894), version/heartbeat/restart
                #   instance PVs (restart -> exit 86)
  file_plugin.py # HdfFilePlugin (#806): one per stream variable, the
                #   areaDetector NDFileHDF5 PV set (+ Rewind, WriteStatus,
                #   WriteMessage) over a single writer thread; lossless
                #   intake before the latest-wins slot; NDFileHDF5 layout
  diff.py       # `geecs-pva-gateway diff`: plugin stacks vs native PNGs
                #   per scan (the rollout's parity check)
deploy/
  bootstrap.ps1   # one-time per-box setup (venv, firewall, NSSM service with
                  #   USERPROFILE override -> service-owned profile; installs
                  #   Python 3.11 itself when `py -3.11` is missing)
  launch.bat      # pull-on-restart launcher (fleet pins offline from the
                  #   share's wheel cache, then reinstall from the shared
                  #   GEECS-Plugins clone; its checked-out commit = fleet pin)
  requirements-fleet.txt # exact pins of external deps added after bootstrap
                  #   (the closure: --no-deps on both sides); h5py first
  stage_wheels.sh # downloads the pins' win_amd64/cp311 wheels into
                  #   <Active Version>/pva-wheels beside the share clone
  gen_fleet_status.py # --experiment X -> fleet_status_<x>.bob from
                  #   fleet.py's roster (DB + [pva] addr_list); rerun + commit
                  #   when the fleet changes, never hand-edit
  fleet_status_undulator.bob # GENERATED Phoebus fleet screen (HTU, the
                  #   reference deployment): version/heartbeat/restart per
                  #   deployed host, a "not deployed" label otherwise
tests/
  test_config.py  # scoping/naming units (fake DB rows, no network)
  test_fleet.py   # roster derivation, the [pva] addr_list deployed mark,
                  #   the generated screen's rows, the fleet probe's lines +
                  #   record and the `fleet` dispatch (fake DB/getter, no network)
  test_server.py  # end-to-end over a binary wire-format fake camera +
                  #   isolate=True PVA server
  test_entrypoint.py # CLI exit codes (incl. the restart code 86 contract)
  test_file_plugin.py # the PV contract pinned by the stock ophyd-async
                  #   ADHDFDataLogic over a real NDFileHDF5IO on pva://;
                  #   session semantics (arming, dedupe, stale, rewind,
                  #   counters, no directory creation, no empty file)
  test_diff.py    # the parity tool
  test_deploy_files.py # the launcher's package list + wheel step, the pins'
                  #   consistency with pyproject (name and specifier)
```

## Architecture (one asyncio loop)

- **Per-device worker, per-variable subscriptions**: the worker owns the
  device's stream `SharedPV`s (images and arrays alike; an array PV is a
  1-D or `(n, 2)` float64 NTNDArray); each stream variable gets its own
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
  gateway's device supervisors; a box ARMED through a long move pushes
  nothing for tens of seconds, #894). Each variable's `<image PV>:connected`
  (NTEnum `Idle` / `Disconnected` / `Connected`, MAJOR alarm while down)
  shows that subscription's state — `Idle` is "gated off, nothing known",
  so a client wanting the verdict holds a monitor on the image PV for one
  gating round-trip. Once the backoff sits at its ceiling the endpoint is
  re-asked of the DB (`endpoint_resolver`, the CA gateway's idiom) and a
  moved port redialed; a move off this host is logged, never adopted
  (#854).
- **Frame path**: push frame → timestamp ladder (`acq_timestamp` →
  `systimestamp`, LabVIEW→Unix, else receive time) → **latest-wins slot** per
  variable → decode (`decode_imaq_image_string`) in the default executor, off
  the event loop → `pv.post(image, timestamp=...)`. A stalled consumer drops
  stale frames; nothing ever backlogs. Completeness lives in the GEECS file
  path, not this stream.
- **File plugin** (`file_plugin.py`, #806): a second consumer of
  the push frame with the *opposite* delivery contract — lossless within
  a capture session — branching off in `_on_frame` **before** the
  latest-wins slot. Per image variable: the `NDFileHDF5IO` PV set under
  `<image PV>:hdf1:` (prefix minted by `geecs_core.pv_naming.hdf_plugin_prefix`),
  one writer thread owning all session state and the file handle (puts
  and frames only enqueue). `Capture=1` zeroes the session readbacks
  (`NumCaptured_RBV` first — the stock logic baselines on it, #853),
  retains the variable's subscription like a client and completes at
  once on the last frame the worker decoded for the variable (that is
  where the geometry the worker describes the stream with comes from;
  the held frame is never written); only a never-decoded variable waits
  for its first push, `ARM_TIMEOUT_S` at most (#894 — waiting for a push
  on a box ARMED through a long first move failed the run's first
  prepare; never-decoded is every camera after each gateway restart until
  its first session gets a push — an image monitor held for one gating
  round-trip in STANDBY seeds it); frames are deduped on `acq_timestamp` and
  stale-filtered against a watermark set at `Capture=1` and moved by
  `Rewind` (the refire guard: truncate to N, drop older-stamped
  arrivals); `NumCaptured_RBV` posts after each frame is on disk;
  `Capture=0` stamps the reconciliation counters and closes. Beside the
  two frame stamps the plugin writes the device's **subscribed scalars**
  per frame (`CameraSpec.scalar_variables`, from
  `geecs_core.db.scalar_policy.GeecsDbScalarPolicy` filtered by
  `geecs_core.db.variable_types.scalar_attribute_variables` — the same
  rule the worker builds a device's row from; numeric types only, the
  timestamp ladder excluded) as `<device>-hdf-<variable>-<scalar>`
  `DOUBLE` attributes, declared in `NDAttributesFile` so the stock data
  logic describes them as stream columns; their values come from the
  frame's own TCP push (the one subscription is widened by the list,
  `_CameraWorker.subscription_variables`), `NaN` when absent. Frames **and those attribute
  datasets** are written **compressed by default** (`Compression`
  defaults to `zlib` → shuffle + gzip level 1, built-in HDF5 filters,
  self-describing: no reader learns anything) — one switch governs both.
  For the attributes it is not a nicety: their chunk is
  `ATTRIBUTE_CHUNK` (16384) f8 slots that a scan fills a few dozen of,
  HDF5 commits the whole chunk on first write and rewrites every dirty
  chunk on the per-frame flush, so uncompressed they cost ~1.7 MB of
  SMB write traffic **per frame** to carry ~100 bytes of numbers.
  The chunk shape stays 16384 either way: ophyd-async hard-codes
  `chunk_shape=(16384,)` in the stream resource it hands Tiled
  (`epics/adcore/_data_logic.py` — "NDAttributes appear to always be
  configured with this chunk size"), so that is the chunking Tiled
  registers; the file has to match the declaration, not the other way
  round. Nothing *enforces* it — Tiled's consolidator validation is
  opt-in via a `_validate` parameter ophyd-async does not set, and runs
  `fix_errors=True` when it does run — so a mismatch corrupts the
  registered layout silently rather than failing. Compression is on
  because no client ever puts that PV — the
  stock `ADHDFDataLogic` does not; a client wanting raw frames puts
  `None` before `Capture=1`. Any of the eight areaDetector choices is
  *accepted* by the put; only those two survive the arm, and a third
  fails `Capture=1` with `WriteStatus=Write Error`.
  Never
  creates a directory (`CreateDirectory` is ignored); never HDF5 SWMR
  across SMB (`SWMRMode` accepted and ignored; flush per frame, file
  locking off). Served only where `h5py` imports (`file_plugin.available`).
- **Identity/control PVs**: `{experiment}:pvagateway:{host_token}:version|
  heartbeat|restart` per instance — the fleet screen reads the first two
  (version skew, liveness); writing `:restart` exits 86 for the service
  manager to relaunch (rollout mechanism, `deploy/`).

## Ground rules

- **Naming**: only via `geecs_core.pv_naming`. No local copies.
- **Transport**: only `geecs_core.transport`. Never GEECS-PythonAPI
  (deprecated, slated for deletion).
- **Non-scalars stay off the CA gateway; scalars stay off this one.** This
  package serves image-typed and `1darray`-typed variables only (the file
  plugin's control PVs are areaDetector's, per stream variable — not GEECS
  scalars). If PVA scalars ever happen, that is a deliberate design step
  (per-device-class PVA adoption, DESIGN.md), not a drive-by addition here.
- **Which arrays are served is the DB minus GEECS-Core's exclusions**
  (`geecs_core.db.device_streams`): never a per-host list, never an image.
  A devicetype's array shape policy (the padding ceiling) lives there too.
  An instance whose host has no stream device idles on its identity PVs
  rather than exiting.
- **Text variables**: image and array variables must always be subscribed
  as `text_variables` — numeric coercion destroys binary payloads (and would
  turn a CSV lineout into its first number).
- The wire format is binary-hostile in known ways — decode quirks live in
  `geecs_data_utils.io.images` (name-repeat vs tail-anchored wrappers) and
  `geecs_data_utils.io.arrays` (the three array shapes, byte-exact), and
  the latin-1 byte↔str convention comes from the gateway transport (0.16.1).
  Do not re-derive any of them here.
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
