# Changelog

All notable changes to `geecs-core` are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [0.4.1] - 2026-09-14

### Fixed

- **Float sets transmit the caller's digits, not a `%.12f` expansion**
  (issue #819). `GeecsUdpClient.set` formatted every float with `%.12f`,
  which at 12 decimals exposes the binary representation — `40854.24625`
  went out as `40854.246249999997` — and LabVIEW devices rejected the
  string as "not a number" (86% of five-decimal values in
  `U_CompAeroTech`'s working range). Floats now render via the new
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
