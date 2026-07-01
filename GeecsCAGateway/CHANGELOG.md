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
- PV name mapping (`deviceName:variable` → CA-safe PV; whitespace normalized).
- `DeviceSpec.from_geecs_db(name)` / `from_db_metadata(...)` — build a device
  spec straight from `GeecsDb` (units → EGU, min/max → CA control limits,
  `set` → settable). The network-free `from_db_metadata` core is unit-tested.
- CA control/display limits (`lo`/`hi` on `VariableSpec`) wired onto channels.
- Offline demo (`python -m geecs_ca_gateway.demo`) and tests against the
  in-process `FakeGeecsServer` — no hardware or lab network required.
- `DESIGN.md` — design note (Path A vs B, caproto rationale, regime fitness,
  what's proven on real hardware, honest gaps).

### Verified

- End-to-end against real device `U_S1H`: DB-driven config, live readback, and a
  `caput` that drove the magnet current and tracked back to baseline.
- Real CA wire via caproto CLI tools; 15 offline tests.
