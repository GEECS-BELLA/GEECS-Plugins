# Changelog

All notable changes to `geecs-control-room` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.


## [0.1.0] - 2026-10-28

### Added

- The package skeleton: the app factory on the kit, settings from
  `GEECS_CONTROL_ROOM_*` (the shared `GEECS_EXPERIMENT` and
  `GEECS_CONFIGS_ROOT` beneath), the module client of `/geecs/v1`, the
  Hare WebSocket client (`static/hare.js`: read key as `?api_key=`,
  reconnect and resubscribe), `/healthz`, the systemd unit template.
