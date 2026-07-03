# Changelog

All notable changes to `geecs-ca-gateway` are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [0.1.1] - 2026-07-03

### Changed

- `naming.normalize_pv_component` now delegates to the shared
  `geecs_bluesky.pv_naming.normalize_component`, so the gateway (PV producer) and
  the CA-backed ophyd-async devices (PV consumers) share one naming policy and
  can never drift. No behavior change.

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
  task that reconnects with exponential backoff on an **actual disconnect** (the
  socket closing). A device merely going quiet is NOT treated as a drop — GEECS
  devices are legitimately silent for seconds (waiting on triggers, slow online
  analysis, toggled), so silence just ages the PV timestamp rather than forcing a
  pointless reconnect. (A hard power-off with the socket left open is a known gap
  best closed later with TCP keepalive, not app-level silence-guessing.)
- PV timestamps from GEECS, not gateway-receive time: each frame is stamped via
  a timestamp ladder (`DeviceSpec.timestamp_vars`, default
  `["acq_timestamp", "systimestamp"]` — both subscribed on every device;
  `acq_timestamp` (triggered devices, true shot time) preferred, `systimestamp`
  (universal) fallback). GEECS timestamps are LabVIEW epoch (1904) — converted to
  Unix by subtracting 2_082_844_800. Verified on real hardware.
- The transport's "missing variable(s)" notices are quiet by default in the
  serve entry point (subscribed-but-idle vars are normal for monitoring); pass
  `--show-missing` to keep them.
- The intrinsic timestamp variables (`systimestamp`, `acq_timestamp`) — which are
  not in the DB — are now also exposed as float readback PVs per device, carrying
  the **raw** LabVIEW-epoch value (what's stamped on saved external assets like
  images, so it matches for synchronicity). A per-device acquisition/liveness
  signal, in addition to stamping each data PV's timestamp.
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

- **Serve entry point** — `python -m geecs_ca_gateway --experiment NAME` (also
  the `geecs-ca-gateway` console script) builds the config live from the DB,
  connects, and serves the PVs over CA until interrupted. `--all-variables` /
  `--include-disabled` widen the set; EPICS `EPICS_CAS_*` env vars scope the
  subnet. This is the library→service step.
- **Monitor deadband** — readback PVs only re-post when the value actually
  changes (floats beyond a per-variable deadband sourced from the DB
  `tolerance`), so a static device produces no CA/archiver traffic. Keeps
  archive storage proportional to real changes, not the 5 Hz push rate.
  (`GeecsDb.get_device_variables` now returns `tolerance`; geecs-bluesky 0.13.6.)

### Fixed

- The `choices` field is authoritative for type: `choices='image'`/`'1darray'`
  means a non-scalar (skip) even when `variabletype='choice'` is explicitly set.
  Fixes image/scope-trace variables being built as bogus one-option enums and
  then choking on raw image bytes under `--all-variables`.
- Type inference when the DB `variabletype` column is blank: some rows encode
  the type only via `choice_id` (e.g. `U_VisaPlungers DigitalOutput.Channel 0–3`
  have `variabletype=NULL` but `choices='on,off'`). These were defaulting to
  float and then failing on string values; now a real option list infers `enum`
  and a bare descriptor (`numeric`/`string`/`path`/`image`/`1darray`) infers that
  type.
- A value that can't be coerced to its PV type now warns **once** per variable
  (concise, no traceback) instead of every ~5 Hz frame.
- Readbacks use **display** limits (informational), not **control** limits, from
  the DB min/max. caproto enforces control limits on write and was rejecting
  faithful-but-out-of-range readbacks — notably `NaN` from a failed analysis.
  Readbacks now report reality (incl. NaN); GEECS remains the authority on valid
  values. Static NaN is deadband-suppressed too.
- Reconnect logging is now **state-change** based: one concise warning when a
  device goes down/unreachable, one info when it reconnects — no per-attempt
  tracebacks for devices that are simply off.

### Verified

- End-to-end against real device `U_S1H`: DB-driven config, live readback, and a
  `caput` that drove the magnet current and tracked back to baseline.
- Real CA wire via caproto CLI tools; 15 offline tests.
