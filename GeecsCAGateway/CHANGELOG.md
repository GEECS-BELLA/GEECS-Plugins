# Changelog

All notable changes to `geecs-ca-gateway` are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [0.1.0] - 2026-07-01

### Added

- Initial proof of concept: EPICS Channel Access soft-IOC gateway exposing GEECS
  devices as PVs.
- `GatewayConfig` / `DeviceSpec` / `VariableSpec` Pydantic v2 config models.
- `GeecsCaGateway`: builds a caproto `pvdb` dynamically from config; readback PVs
  driven by the `GeecsTcpSubscriber` stream; settable variables get a `:SP`
  setpoint PV whose CA puts forward to the device over `GeecsUdpClient`.
- Naming policy: `[Experiment:]Device:Variable` namespace (experiment prefix via
  `DeviceSpec.experiment` / `pv_name_for`); strict component mapping to
  `[A-Za-z0-9_]` (the dot is critical — `Trigger.Source` → `Trigger_Source`);
  build-time collision detection; a `manifest` (PV → device/variable/kind) as the
  authoritative bidirectional map.
- Reconnect supervisor: each device's TCP subscription runs under a supervising
  task that reconnects with exponential backoff on a dropped connection.
- Stall watchdog: since GEECS pushes at ~5 Hz, no frame for `stall_timeout_s`
  (default 2 s) is treated as a drop — catches a silently-vanished device
  (powered off with no TCP FIN), which socket-close detection alone misses.
- PV timestamps from GEECS, not gateway-receive time: each frame is stamped via
  a timestamp ladder (`DeviceSpec.timestamp_vars`, default `["systimestamp"]`;
  prepend `acq_timestamp` for triggered devices to prefer true shot time). GEECS
  timestamps are LabVIEW epoch (1904) — converted to Unix by subtracting
  2_082_844_800. Verified on real hardware (PV timestamp tracks wall-clock).
- Validity: while a device is down its readback PVs are marked `INVALID` (alarm
  severity) so clients can tell live from stale; live frames clear it
  automatically.
- PV name mapping (`deviceName:variable` → CA-safe PV; whitespace normalized).
- `DeviceSpec.from_geecs_db(name)` / `from_db_metadata(...)` — build a device
  spec straight from `GeecsDb` (units → EGU, min/max → CA control limits,
  `set` → settable). The network-free `from_db_metadata` core is unit-tested.
  Duplicate variables (the GEECS DB can list one twice) are deduped.
- `GatewayConfig.from_geecs_experiment(name)` — build a whole-experiment config
  live from the DB, skipping devices not `enabled` in `expt_device` and any that
  fail to resolve. (Verified on Undulator: 145 devices → 114 enabled.)
- **`subscribed_only` (default on)** down-selects each device to its `get='yes'`
  variables from `expt_device_variable` — the per-shot monitoring subset — via
  `GeecsDb.get_subscribed_variables`. Turns the every-variable firehose (~8600
  Undulator variables) into a sensible set (~377). `subscribed_only=False`
  restores the full set.
- **Variable types from the DB.** `variabletype` maps to the PV type
  automatically: `numeric`→float, `string`/`path`→string, `choice`→**enum**
  (`ChannelEnum` with options from the `choice` table). `image`/`1darray` are
  skipped (not scalar CA data). Enum readback maps the GEECS option string to the
  CA index; enum `caput` maps the index back to the GEECS string.
- CA control/display limits (`lo`/`hi` on `VariableSpec`) wired onto channels.
- Offline demo (`python -m geecs_ca_gateway.demo`) and tests against the
  in-process `FakeGeecsServer` — no hardware or lab network required.
- `DESIGN.md` — design note (Path A vs B, caproto rationale, regime fitness,
  what's proven on real hardware, honest gaps).

### Verified

- End-to-end against real device `U_S1H`: DB-driven config, live readback, and a
  `caput` that drove the magnet current and tracked back to baseline.
- Real CA wire via caproto CLI tools; 15 offline tests.
