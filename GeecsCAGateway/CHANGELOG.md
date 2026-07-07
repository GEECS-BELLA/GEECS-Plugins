# Changelog

All notable changes to `geecs-ca-gateway` are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [0.5.2] - 2026-07-06

### Fixed

- **Push-frame fan-out posts timestamp variables last** — the subscription
  callback wrote PVs in device-payload order with an `await` per write, so a
  frame that happened to list `acq_timestamp` first could complete a strict
  Bluesky `CaTriggerable.trigger()` on the new shot id while the data PVs
  still held the previous frame's values — pairing shot N's id with shot
  N-1's data. The fan-out now guarantees all data variables of a frame are
  posted before its timestamp variable(s) (stable sort, so device payload
  order is preserved among the data variables), so a client triggering on
  `acq_timestamp` always observes the completed frame.
- Empty-string readback values for numeric/enum variables skip at DEBUG
  instead of warning "DB variabletype mismatch" — devices push `''` for
  values they haven't computed yet (camera analysis fields before the first
  acquisition, idle devices' whole frames), which produced dozens of
  misleading warnings at every gateway start against live hardware.
  String/path dtypes still pass `''` through (a cleared save path is a real
  value).
- `enum_geecs_value` (the **setpoint** direction) gets the same
  numeric-label treatment: a caput of text `"2.0"` against labels
  `["1","2","5"]` now sends `"2"` to GEECS instead of `choices[2] == "5"`.
  Genuine `int`/`float` puts remain CA enum indices (standard CA
  semantics). Found while writing PV_CONTRACT.md — the readback fix below
  had left its mirror image behind.
- Removed `tests/fake_server_helpers.py` — dead since the transport move
  out of GeecsBluesky (referenced by no test; imported non-dependencies).
- **Numeric enum labels resolve by value, not index** — `enum_index` fell
  back to interpreting any non-matching numeric wire text as an option
  *index*, so a device streaming `"2.000000"` for the label `"2"`
  (DG645-style trigger configs have labels like `["1", "2", "5"]`) resolved
  to index 2 and silently selected `"5"`. Numeric wire values are now
  matched by value against numeric labels; if numeric labels exist but
  nothing value-matches, the update is skipped (with a warning) rather than
  guessed. Choice sets with no numeric labels (e.g. `["Off", "On"]` with
  wire `"1"`) keep the index-interpretation fallback.

## [0.5.1] - 2026-07-05

### Fixed

- **Readback deadband no longer inherits the DB `tolerance`** — the monitor
  deadband was wired to each variable's database `tolerance` field, which is
  a *set convergence* criterion (0.05 A on magnet PSUs), so real
  sub-tolerance motion never posted to readback PVs — and therefore never
  reached recorded event rows or s-files (observed live on `U_S1H:Current`:
  the PSU moved, the readback stayed frozen). Deadband now defaults to 0.0:
  every changed stream frame posts; only exact repeats are suppressed.
- **UDP exe replies are correlated to the in-flight exchange** (PR #449
  review #3) — a reply must name the expected variable (bare or
  command-echo form) or it is logged and dropped, so a late reply from a
  timed-out exchange can no longer resolve the next command's future with
  the wrong value. The bare-token ACK carries no identifying field and
  cannot be correlated; that limitation is documented (a stale positive
  ACK is equivalent to the real one).
- TCP push frames are parsed by anchoring on the subscribed variable names
  (review #4), so values containing commas — e.g. paths — no longer drop
  variables or corrupt neighbours; the payload is taken after the second
  `>>` so values may contain the delimiter.
- String/path/enum variables reach PVs as verbatim wire text instead of a
  float round-trip (review #5): '007' stays '007', '1.10' stays '1.10'.
- Setpoint writes use a configurable 30 s exe timeout aligned with
  `CaMotor`'s move budget — "a slow axis is not a dead one" — while gets
  keep the standard 10 s (review #7).
- One device's UDP bind failure no longer aborts gateway startup (review
  #13): the device is skipped loudly with its transports closed (caputs to
  it raise a clean `GeecsConnectionError`) and the remaining devices start
  normally.

## [0.5.0] - 2026-07-04

### Fixed

- **Readback PVs now deny CA client writes** (CA access rights: READ). A
  mistaken `caput` to a readback used to *stick*: the deadband cache compared
  against the gateway's own last write, so the next unchanged hardware frame
  was suppressed and the PV showed the client's value until the hardware
  actually changed. Clients now get a proper `ECA_PUTFAIL` ("cannot write");
  setpoints live at `…:SP` as before. Verified over the CA wire.

### Added

- **Self-diagnostics PVs** (devIocStats-style):
  `[Experiment:]<Device>:CONNECTED` per device (enum, MAJOR severity while
  down — explicit liveness for Phoebus/alarm layers instead of inferring from
  INVALID data severity), and `[Experiment:]CAGateway:{UPTIME, HEARTBEAT,
  DEVICES_CONNECTED, VERSION}` updated by a 5 s status loop.

## [0.4.0] - 2026-07-04

### Changed

- **This package is now the self-contained GEECS access layer.** The wire
  protocol (`transport/`: `GeecsUdpClient`, `GeecsTcpSubscriber`), the
  experiment database (`db/GeecsDb`), the PV naming contract (`pv_naming`),
  the transport-level exceptions, and the `FakeGeecsServer` test double moved
  here from GeecsBluesky, and the package dependency flipped: GeecsBluesky now
  depends on this package (library: `GeecsDb`, `pv_naming`, exceptions;
  service: the PVs) — never the other way around. Deploys slim down
  accordingly: the gateway environment needs only caproto + pydantic +
  mysql-connector (no ophyd/bluesky/pandas). On an existing deployment,
  `git pull` + `poetry install` inside `GeecsCAGateway/` completes the switch.

## [0.3.0] - 2026-07-03

### Changed

- **Path variables are served as long-string (char-array) PVs.** EPICS
  `DBR_STRING` caps at 40 characters; GEECS save paths routinely exceed that,
  so path-typed variables (DB descriptor `path`) were silently truncating their
  readbacks at 40 chars and rejecting >40-char setpoint puts outright
  (`CAException 186` — found live when the CA-backed scanner tried to configure
  camera image saving). They now use `ChannelChar` (capacity 512) per the
  standard EPICS long-string convention (areaDetector `FilePath` does the
  same); ophyd-async reads/writes them as `str` natively. Plain `string`
  variables keep the native 40-char string PV for display friendliness.

## [0.2.0] - 2026-07-03

### Added

- **Settable variables are exposed in subscribed mode** (`include_settable`,
  default on; CLI `--no-settable` to opt out). The get-list is the *monitoring*
  subset, but settable variables are the device's *control surface* — a camera's
  `save` / `localsavingpath`, a magnet's setpoint — and CA clients need their
  readback + `:SP` PVs for writes regardless of what is monitored per shot.
  Without this, control-side consumers (the CA-backed Bluesky devices, Phoebus
  set panels) could only drive variables that happened to be in the get-list.

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
