# Changelog — geecs-pva-gateway

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-07-25

### Added

- **Initial package: the PVA peer of GeecsCAGateway, serving GEECS camera
  images as NTNDArray PVs from the camera servers themselves.** Verified
  architecture from the 2026-07-25 live pilot (UC_Amp2_IR_input at 1–5 Hz over
  VPN).
  - `PvaGatewayConfig.from_geecs_experiment` — DB-scoped served set: enabled
    devices on this host's endpoint IP with image-typed variables (via the
    gateway's `effective_vartype`, including the choice-descriptor quirk).
  - `GeecsPvaGateway` — one process, N cameras: **per-variable** gated +
    supervised GEECS TCP subscriptions (start on a variable's first PVA
    client / stop on its last; reconnect with 0.5→30 s backoff on socket
    drops), IMAQ decode off the event loop, latest-wins frame slots, GEECS
    timestamp ladder (`acq_timestamp` → `systimestamp`, LabVIEW→Unix),
    startup collision guard on normalized PV names, and per-instance
    `version`/`heartbeat` identity PVs.
  - CLI `geecs-pva-gateway --experiment NAME [--host IP] [--devices A,B]
    [--list]`.
  - Hermetic test suite: binary wire-format push server + isolated PVA server
    (`isolate=True`), covering frame flow, timestamps, gating both directions,
    and reconnect-after-drop.
