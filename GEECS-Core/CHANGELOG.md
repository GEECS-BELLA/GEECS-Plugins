# Changelog

All notable changes to `geecs-core` are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.


## [0.8.3] - 2026-09-16

### Changed

- Strip the `Planning/` provenance citations from docstrings and comments: a
  docstring now states the rule itself, and the derivation stays in git
  history. Part of the `Planning/` prune (#931); no behaviour change.
- `CLAUDE.md`: state the capability inheritance rule this package owns — the
  `variable` row replaces the `devicetype_variable` row **wholesale**, with no
  field-level fallback, so a type-level default under an instance row needs a
  deliberate per-field coalesce and is a departure from the rule. What the
  gateways serve from those rows (`.DESC`, its 40-character EPICS limit,
  control limits) stays the gateway's contract and is pointed at, not copied.

## [0.8.2] - 2026-09-16

### Changed

- Publish the shared LabVIEW-to-Unix epoch offset for PVA timestamps and native optimizer frame joins.

## [0.8.1] - 2026-09-14

### Changed

- Merge of `master` (21f46821) into `feature/native-bluesky-plans`: the
  two lines were released in parallel and are listed below in version
  order; the block marked *(master line, parallel release)* is master's
  0.4.1, whose one change — `transport._coerce.format_float`, the
  shortest round-trip float formatting for every set command (#819,
  PR #897) — is now on this line too. No branch-side code changed.

## [0.8.0] - 2026-09-14

### Added

- `pv_naming.CONNECTED_SUFFIX` / `connected_pv(experiment, device,
  variable)` — the PVA gateway's per-image-variable subscription-state PV
  (`<image PV>:connected`, GeecsPvaGateway 0.10.0, GEECS-Plugins#854),
  minted beside `HDF_PLUGIN_SUFFIX` for the same reason: the server and
  a worker-side reader (the submit preflight's liveness gate) both take
  it from here, so the two sides cannot drift.

## [0.7.0] - 2026-09-14

### Added

- `db.settables`: `numeric_settables(rows_by_device)` → `NumericSettable`
  rows — the one filter (`settable` and `effective_vartype == numeric`)
  and the one order (aliased first, alphabetical by alias, then the rest
  by canonical `Device:Variable`) every movable picker shows. Pure logic
  over `get_experiment_device_variables` rows, placed beside
  `variable_types` so the web scanner, the scan MCP and later pickers
  import one list instead of each sorting their own.

- `GeecsDb.get_device_variables` / `get_experiment_device_variables` rows
  carry **`alias`** — the per-instance `variable.alias` the DB curates as
  the operator-facing short name (`""` when none; the type table's column
  rides along but is unpopulated in practice). First reader: the web
  scanner's movable panel, which lists every numeric settable alias-first
  and shows the alias beside the canonical `Device:Variable`, never
  instead of it (the request stores the canonical name, so a rename in
  the DB breaks nothing).

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
- `db.variable_types.scalar_attribute_variables(rows, subscribed,
  normalize=…)` and `TIMESTAMP_LADDER`: the per-frame scalar filter beside
  `image_variables` — the subscribed **numeric** variables of a device in
  DB order, minus the timestamp ladder, matched to the metadata rows
  case-insensitively (the namespace's rule; the row's spelling is kept),
  a second name normalizing onto an earlier one's dataset dropped with a
  warning.  Enums are excluded on
  purpose: their wire value is the text label on both gateways.  The PVA
  gateway builds its roster from it; the worker recovers a stack's columns
  through it.

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

## [0.4.1] - 2026-09-14 *(master line, parallel release)*

### Fixed

- **Float sets transmit the caller's digits, not a `%.12f` expansion**
  (issue #819). `GeecsUdpClient.set` formatted every float with `%.12f`,
  which at 12 decimals exposes the binary representation — 86% of
  five-decimal values in `U_CompAeroTech`'s working range grew such a
  tail, `40854.24625` going out as `40854.246249999997` — and LabVIEW's
  `Is Value a number.vi` rejected that string as "not a number"
  (reproduced on hardware; not every tail is rejected, but the fix
  removes them all). Every non-integral real (Python floats, numpy float
  scalars, ...) now renders via the new
  `transport._coerce.format_float`: the shortest round-trip decimal
  (`40854.24625`, `40966.0`, `0.00001`), never truncated to a fixed
  number of decimals (a `%.1f` would zero the `0.001` tolerances and
  `1e-05` minima in the DB), and never exponent notation (expanded to a
  plain decimal, which LabVIEW parses). One formatter for every float
  set to every device — the CA gateway and `GeecsDevice` forward values
  unformatted.

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
