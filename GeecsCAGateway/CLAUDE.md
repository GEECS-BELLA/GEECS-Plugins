# GeecsCAGateway — Developer Context for Claude

The caproto **Channel Access soft-IOC** that mirrors GEECS devices as EPICS
PVs (readback + `:SP` setpoints), built on the GEECS access library
**GEECS-Core** (`geecs_core`: UDP/TCP transport, experiment DB, PV naming,
exceptions — extracted from this package 2026-08-20). Every EPICS-ecosystem consumer — GeecsBluesky's ophyd-async
devices, Phoebus displays, the planned Archiver Appliance — talks to GEECS
through these PVs instead of growing its own bespoke bridge.

```
GEECS device  --TCP push stream-->  readback PV   (caget / camonitor)
GEECS device  <--blocking UDP set--  setpoint PV  (caput …:SP)
```

This is **load-bearing infrastructure**, not a prototype: Bluesky GUI scans on
the Windows control machines and Phoebus displays consume it live. Treat its
externally observable behavior as a contract — the three deep references are:

- **`PV_CONTRACT.md`** — the normative API contract for every CA client
  (naming, types, timestamps + the frame-ordering guarantee, alarm policy,
  collision and failure semantics). Behavior changes that touch it must update
  it and its pinned tests in the same PR.
- **`DEPLOYMENT.md`** — launch/CLI, config resolution, the current dev
  deployment on 192.168.6.14 and its risks, the client-side recipe (Windows
  included), smoke tests, target systemd shape.
- **`DESIGN.md`** — why this exists (Path A vs B), the caproto decision,
  regime fitness, foundational decisions, honest gaps.

## Package Layout

```
geecs_ca_gateway/
  __main__.py        # python -m geecs_ca_gateway --experiment NAME (CLI; console
                     #   script `geecs-ca-gateway`) — DB-driven config, then run()
  gateway.py         # GeecsCaGateway — pvdb build, manifest/collision guard,
                     #   per-device subscription supervisors (reconnect+backoff),
                     #   push-frame fan-out (timestamps last), INVALID marking,
                     #   CONNECTED + CAGateway:* status PVs (incl. the writable
                     #   RESTART control → exit 86 → systemd relaunch = DB resync),
                     #   UDP setter closures
  channels.py        # caproto channel construction: readback (client-read-only)
                     #   vs setpoint (write forwards to GEECS first), enum
                     #   index/label resolution, path long-string channels,
                     #   cast/unwrap helpers
  config.py          # Pydantic models: GatewayConfig / DeviceSpec / VariableSpec;
                     #   DB→dtype mapping incl. choice-descriptor quirks;
                     #   from_db_metadata (pure) / from_geecs_db / from_geecs_experiment
  audit.py           # read-only DB-hygiene audit (#496): reports every get='yes'
                     #   variable the gateway won't serve + FK-orphans. Pure
                     #   classifier audit_subscribed_variables() (no DB) + CLI
                     #   (python -m geecs_ca_gateway.audit / geecs-ca-gateway-audit)
  demo.py            # offline self-checking demo over the fake server
```

The access-layer modules this package used to own — `transport/`
(GeecsUdpClient, GeecsTcpSubscriber), `db/` (GeecsDb + the alarms model),
`pv_naming`, `exceptions`, `testing/fake_device_server.py` — live in
**`GEECS-Core/geecs_core/`** (see its `DESIGN.md`); this package imports them
from `geecs_core.*` and layers the CA server on top.

Dependency direction: **nothing imports this package except GeecsPvaGateway**
(config helpers, e.g. `effective_vartype`). GeecsBluesky and GEECS-Console
take the library parts (`GeecsDb`, `pv_naming`, exceptions) from `geecs-core`
directly and consume this gateway purely as a *service* (the PVs, via stock
ophyd-async EPICS signals). The gateway also imports `geecs-schemas` for
schema-validated derived-channel overlay files. The gateway env is deliberately
slim: caproto + pydantic + geecs-schemas + geecs-core + PyYAML — no
ophyd/bluesky/pandas.

## Architecture

One asyncio event loop runs everything:

- **`GeecsCaGateway` (the supervisor)** builds the `pvdb` at construction time
  (no network), then `run()` = `connect()` (UDP clients) → `subscribe()` (one
  supervising task per device + a 5 s status loop) → `serve()` (caproto CA
  server). The **manifest** (`PV → (device, geecs_var, kind)`) is both the
  authoritative reverse map (PV naming is lossy) and the startup collision
  guard: exact DB duplicates are tolerated, genuine collisions raise.
- **Per-device UDP client** (`GeecsUdpClient`) handles sets/gets: send command,
  await ACK on the cmd socket, await exe reply on cmd_port+1. One in-flight
  exchange per device (lock). Setpoint puts get a 30 s budget (CaMotor's move
  contract — GEECS sets block until convergence); gets keep 10 s.
- **Per-device TCP subscriber** (`GeecsTcpSubscriber`) sends `Wait>>var1,var2`
  (4-byte big-endian length framing) and receives ~1–5 Hz push frames. Each
  device's subscription runs under a **supervisor** that reconnects with
  exponential backoff (0.5→30 s) on an *actual socket drop* — silence is not a
  drop (GEECS devices legitimately idle for seconds). While down: readbacks go
  INVALID/COMM, `CONNECTED` goes MAJOR; recovery is automatic and the
  change-suppression cache is cleared so the first frame always posts.
- **Fan-out** (`_make_callback`): each frame is stamped from the timestamp
  ladder (`acq_timestamp` then `systimestamp`, LabVIEW→Unix), values are
  cast/enum-resolved per spec, exact repeats suppressed (deadband default 0.0),
  and **timestamp-ladder variables are posted last** so a client triggering on
  `acq_timestamp` observes the completed frame (PV_CONTRACT.md §3).
- **channels/naming**: `channels.py` is the only caproto-typed layer (kept
  swappable per DESIGN.md's per-device-class PVA plan — images already run
  on PVA via `GeecsPvaGateway/`; scalars stay CA). Readback channels deny client
  writes (access rights READ); setpoint channels forward to GEECS *before*
  storing, so a failed set fails the caput. `geecs_core.pv_naming` is the one
  shared naming module; full names are assembled by `DeviceSpec.pv_name_for`.
- **db (geecs_core.db)**: `GeecsDb` is class-method-only, credentials cached
  from the standard GEECS config chain. `config.py` maps DB rows → specs; the network-free core
  is `from_db_metadata` (unit-tested without MySQL);
  `from_geecs_experiment` builds a whole experiment (get='yes' subset +
  control surface by default — settable-only devices keep their `:SP` PVs)
  from three batched DB queries.

## Wire-protocol quirks that bit us (do not relearn these live)

- **UDP exe replies must be correlated.** A slow device blows the exe timeout;
  the next command's future would then be resolved by the *stale* reply —
  values desync from variables (shipped bug, fixed 0.5.1). Exe replies name
  their variable in field 2, either bare (`Var`) or as the hardware's command
  echo (`getVar`/`setVar`) — accept both, drop everything else with a warning.
  The bare `accepted`/`ok` **ACK carries no identifying field and cannot be
  correlated** — accepted limitation, documented in PV_CONTRACT.md §7.
- **Push frames cannot be comma-tokenized.** Values contain commas (save
  paths!), `>>`, even newlines. The parser anchors on the *subscribed variable
  names* + ` nval,`/` nvar` tokens, longest names first. Residual ambiguity (a
  value containing a full pair-boundary lookalike) is inherent to the format
  and documented — do not "fix" it into a stricter parser that drops path
  values.
- **Numeric coercion is lossy for text.** `'007'` → 7 → `"7"`. String/path/enum
  variables must be passed to the subscriber as `text_variables` so the exact
  wire text survives (enum labels like `"1.0"` need this too).
- **Exe status is `"no error,"`, not the documented `"ok,"`** — match the
  prefix. Real hardware also differs from protocol docs in the echo field
  (above); when in doubt trust the fake server (and the legacy
  GEECS-PythonAPI parser, now in git history), which encode observed behavior.
- **Numeric enum labels resolve by value, never index** (`"2.000000"` is the
  label `"2"`, not index 2 — DG645 configs have labels `["1","2","5"]`).
  Index-interpretation is a fallback reserved for fully non-numeric label sets
  (PR #452; PV_CONTRACT.md §4).
- **DB `tolerance` is a set-convergence criterion, not a monitor deadband.**
  Wiring it as a deadband froze sub-tolerance motion out of readbacks, scan
  rows, and s-files (shipped bug, fixed 0.5.1). Deadband stays 0.0.
- **Local-IP detection at connect time** — binding UDP to `""` raises
  `EADDRNOTAVAIL` on macOS PPP/VPN links; the client probes the OS route and
  binds explicitly. One device's bind failure must not abort the gateway
  (per-device skip, 0.5.1).

## Testing infrastructure

- **`FakeGeecsServer` / `FakeGeecsDevice`** (`geecs_core.testing`) — an
  in-process asyncio UDP+TCP server speaking the real wire protocol (5 Hz
  pushes, ACK/exe replies). The whole suite is offline: no hardware, no lab
  network, no CA client needed (tests drive channels directly via
  `channel.write`). GeecsBluesky no longer imports it — its hermetic tests
  use ophyd-async mock backends instead.
- `pytest` runs `asyncio_mode = "auto"` and deselects `integration` (lab-DB)
  tests by default; `fake_server` marks tests that open localhost sockets.
- Test files map to layers: `test_naming` (DeviceSpec PV assembly) /
  `test_channels` / `test_config_from_db` (pure units), `test_gateway`
  (fake-server end-to-end), `test_pv_contract` (thin example-tests pinning
  PV_CONTRACT.md claims), `test_entrypoint` (CLI). The wire-protocol and DB
  suites (`test_transport`, `test_udp_reply_correlation`, `test_geecs_db`,
  `test_coerce`, `test_pv_naming`) moved to `GEECS-Core/tests/` with the
  code they pin.
- **When you change externally observable behavior**: update `PV_CONTRACT.md`,
  and add/adjust its pinned test in the same PR. The contract's test map is the
  index.

```bash
cd GeecsCAGateway
poetry install
poetry run pytest tests -q          # offline, green with no lab access
poetry run python -m geecs_ca_gateway.demo   # serve fake PVs, poke with caget
```

## Config resolution

No package-owned config file. `~/.config/geecs_python_api/config.ini`
(`[Paths] geecs_data`) → `{geecs_data}/Configurations.INI` (`[Database]`) →
the GEECS DB as the live source of truth for the served set. `EPICS_CAS_*`
env vars scope the server subnet; client-side addressing (`[epics]
ca_addr_list`, applied by geecs_bluesky at import) is documented in
`DEPLOYMENT.md` §3. Restarting the gateway *is* the DB-resync mechanism.

## Ground rules

- **Never let producer and consumer naming drift** — the policy lives only in
  `geecs_core.pv_naming`; GeecsBluesky imports the same module. No copies.
- **Scalars/controls only; images stay off CA** (DESIGN.md). Don't add image
  PVs here — they are served by `GeecsPvaGateway/`, the distributed PVA peer
  running on the camera servers (it imports geecs-core's transport, DB, and
  `pv_naming`, plus this package's config helpers; same one-namespace
  coexistence as GeecsBluesky).
- Follow the repo-wide conventions (root `CLAUDE.md`): Pydantic v2, NumPy
  docstrings, type hints, `poetry version` + `CHANGELOG.md` on every
  code-changing PR.

## DB-hygiene audit (`audit.py`, #496)

`expt_device_variable` accumulates `get='yes'` rows that map to nothing real
(the LabVIEW DB-editing GUIs don't cascade deletes), and each one makes the
gateway/scanner try to connect a PV that never exists → per-device stalls. The
audit reports them; run it on-network with `python -m geecs_ca_gateway.audit
--experiment Undulator` (add `--full` for the per-device listing, `--sql` to
*print* review-only DELETEs). It flags: **GHOST** (`get='yes'` name isn't a real
device variable), **SKIP:<type>** (`get='yes'` var whose *effective* type is
`image`/`1darray`, in `_SKIP_VARTYPES`, so no scalar PV), and **FK-orphan**
(`expt_device_id` with no `expt_device`). Effective types come from the shared
pure helper `config.effective_vartype()` — the same choice-descriptor rules the
config builder applies (`variabletype='choice'` + `choices='image'` is an image;
#512) — so the audit flags exactly what the gateway skips. The classifier
`audit_subscribed_variables()` is a pure function over plain dicts (no DB —
unit-tested in `test_audit.py`); DB access is lazy.
**Read-only: it never DELETEs or modifies** — `--sql` only prints.
