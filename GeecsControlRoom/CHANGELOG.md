# Changelog

All notable changes to `geecs-control-room` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.


## [0.3.0] - 2026-11-04

The raw-frames arc: the camera page shows the camera's own 12-bit frame
beside the IOC's MJPEG stream.

### Added

- **The raw view** on the camera page: a Raw toggle polls
  `/api/devices/{name}/frame.raw` at 2 Hz with `If-None-Match`, decodes the
  camera's 16-bit frame, applies the camera's `TRANS1` transform so the
  crosshairs and the analysis ROI keep their coordinates, and renders it
  with a histogram, auto-levels and the page's colour maps and scales on the
  12-bit values; levels are kept per mode as fractions of each mode's full
  scale; the chip reads new frames per second from the frame tag (#1269)
- **`GET /api/devices/{name}/frame.raw`**: the raw frame read over Channel
  Access at the PVs the module names (`raw_frame`, `raw_width`,
  `raw_height`, `raw_counter`), one read in flight per device shared for
  0.5 s, the counter probed before any array read, served as uint16
  little-endian under gzip with `ETag`/304, 503 words for an unreadable
  counter or frame and for a geometry disagreement (#1267)
- The operator page's raw-view section and screenshots, the deploy notes
  (bandwidth, `EPICS_CA_MAX_ARRAY_BYTES`, the gzip pass-through a proxy must
  keep, the Canopy image rebuild a family change needs); the page's decode
  path tested through the real route over the fake CA (#1271)

### Changed

- `/api/devices/{name}/arrays` serves the on-request roles **of type
  `array`** alone, so a frame or an on-request scalar never lands in the
  JSON answer (#1263)
- The camera family dropped `frame_rate` (`cam1:FrameRate_RBV`): the record
  is the camera's configured rate, unavailable on a triggered Blackfly and
  an unused cap on a Basler; the delivered rate is `rate` (#1254)

## [0.2.0] - 2026-11-03

The first release since the skeleton: pushes 2 and 3 of the GeecsCanopy arc
and the device pages' first three slices, deployed on the services box.

### Added

- **The screens renderer**: one template draws every screen file in
  `<experiment>/screens/`, live over Hare, every write through the module's
  put route; group columns and packed screen columns for a dense panel; a
  bare PV read-only; a DB device bound by name and deferred to the module's
  word, its writes live through the same route (#1193, #1195, #1206)
- **The home**: the run strip, the Not live and In alarm lists sorted from the
  module's words (a list the module could not give reads "unknown", never
  "none"), the launcher from `home.yaml` (#1194, #1201)
- **Action buttons**: a screen button runs an action plan in the module and
  shows the run's state with Stop (#1208)
- **Presets**: Squirrel's snapshots through the module — save dialog, diff
  drawer, restore as a run on the row, the golden mark, the strip's baseline
  chip; a late diff for a superseded selection dropped, an empty pick
  restores nothing (#1213, #1215)
- **Device pages**: `/device/<kind>` picks a device of the kind,
  `/device/<kind>/<name>` draws every role with its verdict and value and
  writes through the module; the **camera page** over the areaDetector chain
  — the frame and the plots' arrays read over Channel Access by this server
  (aioca, a new runtime dependency), colour map, levels, zoom, the analysis
  ROI and crosshairs drawn and dragged, the camera ROI behind a confirm, a
  hazard confirm before a hazard role (#1226, #1230)
- The page rules as pure modules under node tests, one poll loop with
  back-off, and a contract test that drives the module's own routes through
  this app (#1203, #1205)
- The operator page `docs/web_services/control_room.md`, linked from the
  header (#1221, #1239)

### Changed

- The unit is rendered and installed by `deploy/bootstrap_host.sh` with the
  rest of the fleet (#1240)

### Fixed

- An undrawable screen names its cause first (the module's roster
  incomplete), folds the refused bindings behind a disclosure, and asks the
  server again every 30 s (#1243)

## [0.1.0] - 2026-10-28

### Added

- The package skeleton: the app factory on the kit, settings from
  `GEECS_CONTROL_ROOM_*` (the shared `GEECS_EXPERIMENT` and
  `GEECS_CONFIGS_ROOT` beneath), the module client of `/geecs/v1`, the
  Hare WebSocket client (`static/hare.js`: read key as `?api_key=`,
  reconnect and resubscribe), `/healthz`, the systemd unit template.
