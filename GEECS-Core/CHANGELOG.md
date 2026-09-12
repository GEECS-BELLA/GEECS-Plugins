# Changelog

All notable changes to `geecs-core` are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.


## [0.6.0] - 2026-09-12

### Added

- `db.scalar_policy`: `GeecsDbScalarPolicy` and the `ScalarPolicyProvider`
  protocol — the subscribed (`get='yes'`) scalars rule, moved here from
  `geecs_bluesky.db_runtime` beside `variable_types`.  The PVA gateway's
  file plugin now writes a camera's subscribed scalars as per-frame
  attributes (`Planning/native_bluesky/08_gated_batch.md` §4.4) and
  depends on GEECS-Core alone; one home for the rule keeps a gated row's
  columns and a strict row's columns the same by construction.  Semantics
  unchanged: one batched query per kind, cached; a DB failure degrades to
  empty policy with one warning.

## [0.5.1] - 2026-09-11

### Added

- `pv_naming.hdf_plugin_prefix(experiment, device, variable)` and
  `HDF_PLUGIN_SUFFIX` (`:hdf1:`): the file plugin's PV prefix (#806),
  minted here so the PVA gateway and the worker's `GeecsHdfIO` cannot
  drift.
- `db.variable_types.image_variables(rows)`: the camera test (image-typed
  DB variables), the one home for the PVA gateway's served set and the
  worker's plugin-backed rule.

## [0.5.0] - 2026-09-09

### Added

- `geecs_core.db.variable_types`: the one DB-type rule — `effective_vartype`,
  `VARTYPE_TO_DTYPE`, `SKIP_VARTYPES`, `CHOICE_TYPE_DESCRIPTORS` moved
  unchanged from `geecs_ca_gateway.config`, plus a small `is_scalar_vartype`
  helper — so the CA gateway, the PVA gateway and GeecsBluesky share it
  without importing a gateway's config module (GEECS-Plugins#807 phase 1).
  No behaviour change. The canonical DB source is
  `devicetype_variable.choice_id` → the `choice` table (ids 1–4 base types,
  5+ enum lists); `variabletype` is the secondary annotation. Known DB
  defect recorded in the module docstring: 18 Undulator rows with
  `variabletype='numeric'` and an option list should be `choice` (DB sweep).

## [0.4.0] - 2026-08-27

### Added

- **`GeecsDb.get_experiment_device_types(experiment, *, enabled_only=True)`**
  — batch `{device: devicetype}` for an experiment in one connection, the
  batch counterpart of `get_device_type` (mirrors `get_experiment_devices`).
  First consumer: the capture daemon's devicetype-keyed camera discovery
  (GeecsBluesky `geecs_bluesky.capture`).

## [0.3.0] - 2026-08-24

### Added

- **TCP keepalive on `GeecsTcpSubscriber`, on by default** (issue #611 —
  the half-open-socket blind spot; live incident 2026-08-17, u_s1h): the
  subscription is read-only after the `Wait>>` command, so an ungracefully
  dead peer (host crash, power cycle, a partition that eats the FIN/RST)
  used to leave the listener waiting forever on a half-open socket, with
  readbacks frozen at valid-stale values and no alarm.  With keepalive the
  OS probes the idle peer and a dead one resets the connection, ending the
  listener — the supervised-reconnect path.  Probes are answered by a live
  peer's TCP stack even when its application is silent, so the "silence is
  not a drop" doctrine is preserved.  New constructor knobs `keepalive`
  (default True), `keepalive_idle_s` (30), `keepalive_interval_s` (10),
  `keepalive_count` (3) — dead-peer detection in roughly a minute where
  the platform exposes the tuning (Linux/macOS; a bare `SO_KEEPALIVE`
  fallback with OS defaults elsewhere).  Tuning failures are logged and
  never fail the connect.  Pinned by `test_keepalive_enabled_by_default` /
  `test_keepalive_can_be_disabled`.

## [0.2.0] - 2026-08-20

### Added

- **`GeecsDevice`** (`geecs_core.client`) — the entry-level synchronous
  client for one GEECS device, succeeding the legacy GEECS-PythonAPI object:
  `get`/`set` (blocking, typed exe-response values, errors raise — never
  `None`-on-failure), `subscribe` (push frames into `state` with
  `"shot number"` and `"connected"` reserved keys, optional `on_update`
  callback, auto-reconnect supervisor with the gateways' 0.5→30 s backoff,
  opt-out via `reconnect=False`), `close` (idempotent, releases both UDP
  sockets — the legacy port-leak bug class is pinned by test), context
  manager support. Construction resolves the endpoint via
  `GeecsDb.find_device`, or takes explicit `host`/`port` (tests,
  off-network). All I/O rides one shared background asyncio loop
  (`client/_loop.py` — the package's single sync/async bridge per
  DESIGN.md rule 2); no per-device threads, no cross-device command lock.
- **Live-lab test tiers** (`tests/test_live_lab.py`, both deselected by
  default and self-skipping off-network): `integration` (real MySQL:
  endpoint + variable-metadata shape) and `hardware` (real device get +
  subscribe via `GEECS_HW_DEVICE`/`GEECS_HW_VAR`, default U_S1H/Current;
  a set-back test additionally gated on `GEECS_HW_ALLOW_SET=1`).
- 17 fake-server client tests, including the rapid open/get/close socket
  pin and a supervisor server-restart reconnect test.



### Added

- **Package created** by mechanical extraction from `GeecsCAGateway` — no
  behavior changes. Moved in verbatim (last changed in geecs-ca-gateway
  0.18.0): `transport/` (`GeecsUdpClient`, `GeecsTcpSubscriber`, `_coerce`),
  `db/` (`GeecsDb`; plus `alarms.py`, relocated to `db/alarms.py` since it
  models the `ca_alarm_limits` table `GeecsDb` reads), `pv_naming`,
  `exceptions`, and `testing/fake_device_server.py` — together with their
  test suites (`test_transport`, `test_udp_reply_correlation`,
  `test_geecs_db`, `test_coerce`, and the policy half of the old
  `test_naming` as `test_pv_naming`).
- `DESIGN.md` — the layering doctrine (one-way dependencies, a single
  sync/async bridge point, the admission rule for new code).
- Lazy public face: `from geecs_core import GeecsDb` works, while
  `import geecs_core.transport` stays stdlib-only.

### Changed

- Consumers (`GeecsCAGateway`, `GeecsPvaGateway`, `GeecsBluesky`,
  `GEECS-Console`) now import these modules from `geecs_core.*`; the
  gateway's `naming.py` re-export shim was retired in favor of
  `geecs_core.pv_naming` directly.
