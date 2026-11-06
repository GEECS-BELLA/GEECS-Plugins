# GeecsControlRoom — Developer Context for Claude

The operator surfaces over Canopy (`Planning/operator_surfaces.md`): the
experiment's home, the **Control Room**, and the **screens renderer** —
a control GUI is a screen file in the configs repo drawn by one renderer,
never code per screen. FastAPI + Jinja on the kit, plain JavaScript, no
build step (§7: the first slice decides whether that stays).

## Layout

```
geecs_control_room/
  settings.py       Settings.from_env: GEECS_CONTROL_ROOM_* over the shared site keys
  app.py            create_app: the kit at /theme, /static, the pages, /healthz
  module_client.py  ModuleClient: GeecsCanopy's /geecs/v1 routes over HTTP (read key)
  home.py           the home at /: the strip (/api/home/strip, the scanner's API; /api/home/baseline,
                    the golden presets' diff), the "Not live" and "In alarm" lists (/api/home/devices),
                    the launcher (home.yaml), the presets (/api/home/presets and the pass-throughs
                    /api/presets[/{id}/diff|golden|restore] to the module's preset routes)
  screens.py        load(): a screen file bound by resolve_screen; GET /screens/{name},
                    its image files, and the module pass-throughs (/api/devices/{name}[/put],
                    /api/actions, …/{plan}/run, …/runs/{id}[/stop])
  device_pages.py   KINDS (kind word → family kind → template); /device/{kind} the picker (the registry's
                    devices of the kind, /api/device/{kind} their roster states), /device/{kind}/{name} the
                    device resolved by one module read (denied / stale panes, never blank)
  images.py         a camera's frame (/api/devices/{name}/frame.jpg: the IOC's JPEG at the URL role `image`
                    names, rehosted to the IOC's CA address, one fetch in flight, shared, held on role
                    `image_counter` with ETag/304; § "The JPEG" below) and arrays
                    (/api/devices/{name}/arrays: the `on_request` roles of type `array`, at most every 0.5 s, shared)
                    and the raw frame (/api/devices/{name}/frame.raw: role `raw_frame`, uint16 LE row-major,
                    gzip framed, geometry + `raw_counter` tag in headers, ETag (counter + full scale)/304; `raw_counter` probed first so
                    an unmoved camera costs one scalar read; one read in flight, held RAW_HOLD_S; 503 words
                    for an unreadable counter or frame and a geometry disagreement; the contract is
                    § "The raw frame" below), over CA (aioca), read only; the module's route cached ROUTE_TTL_S
  static/hare.js    GeecsHare: Hare's WebSocket, reconnect + resubscribe
  static/home.js    the home's live half: polls the /api/home reads (the presets and the baseline
                    slowly), fills the device pickers, the presets region (save dialog, diff drawer,
                    restore confirm, the run on the row, the golden mark)
  static/screen.js  the renderer's live half: PVs and verdicts from the module, values from Hare, writes
  static/screen_rules.js, home_rules.js   the pages' rules, pure (verdict word, value shown,
                    bounds, writable, differs, outcome state; the home lists' words)
  static/device.js, device_rules.js   the device pages' live half (roles from the module, values
                    from Hare, read only) and its rules (picker options, a role row's word and units)
  static/device.css the device pages' layout
  static/camera.js, camera_controls.js, camera_rules.js, camera.css   the camera page (device_camera.html):
                    the viewer (frame loop, zoom/pan, colour map, levels), crosshairs and analysis ROI
                    drawn and dragged, the acquisition controls and camera ROI (a dialog first), the
                    lineouts and histogram; the Raw toggle (frame.raw at 2 Hz, rotated / flipped by role
                    `transform` as TRANS1 would, levels per mode as fractions of the full scale);
                    device.js's GeecsDevicePage gives it the roles, values, words
  traces.py         a trace (/api/devices/{name}/trace: the family's axis role (`axis_of`) and the array
                    it names, `[channels, samples]` by its `channels_by`, float32 LE gzip framed, held on
                    its `fresh_by` counter with ETag/304; § "The trace" below), over CA through Images' cache
  static/trace.js, trace_rules.js, trace.css   the trace page (device_trace.html): the rail and readings
                    from the family's `panel`, the plot (Plotly through the theme's plot.js: one trace per
                    channel, decimated to the plot's width, linear/log, the viewer's window = Plotly's x
                    range, and average); the rules reuse camera_rules.js's
  static/device_writes.js   a device page's write path (operator, hazard dialog, one list at a time,
                    sent once): the camera's and the trace page's
  static/poller.js  the pages' one poll loop: one request in flight per target, late answers
                    dropped, a timeout, back-off to 10 s, at most 8 requests at once
  templates/        base.html (every page's shell), screen.html (EVERY screen), device.html (every
                    device page until a kind gets its own, filling `kind_component`; device_camera.html the camera's, device_trace.html the trace's), the pages
examples/sim_bench.yaml   the worked example: the geecs-epics simulator bench's four devices
docs/             the example rendered live (the PR's screenshots)
deploy/
  geecs-control-room.service   unit TEMPLATE (rendered by deploy/render_units.sh)
```

## Rules that hold here

- **A client of Canopy, never a part of it.** Canopy and GeecsCanopy are
  services: Hare's WebSocket for live values, the module's routes for
  verdicts and writes. Never import `geecs_canopy`, GeecsBluesky, the
  portal or the logbook (import contract `control-room-edges`). The one
  exception to "through Canopy" is what Hare does not carry: a camera's
  frame and its arrays, and a trace, are read over Channel Access by this
  server (`images.py`, `traces.py`, aioca), **read only**, at the PVs the module's device route
  names (never derived here); CA addressing is the standard `EPICS_CA_*`
  environment from `site.env`, never `config.ini` or GEECS-Core. No write
  ever leaves this process but through the module's put route.
- **Read keys only (no operator key carries Canopy `write`).** The
  page carries the display's read key (on `<body>`, for Hare's socket);
  a write key never reaches this process. Every write is
  `POST /geecs/v1/devices/{name}/put` with the operator's name, sent once:
  the page shows the module's outcome word and code and never retries.
- **A screen binds, groups, confirms and calls; it never computes.** No
  expressions, scripts or conditions in a screen file, and no device logic
  in the renderer: verdicts are the module's (`freshness.py` there), alarms
  and connection are Hare's. A changing value is never evidence of
  freshness.
- **One template draws every screen.** `screen.html` + `screen.js`, from
  the screen document (GEECS-Schemas `Screen`, `resolve_screen`). A new
  screen is a file in `<experiment>/screens/`, never code. A widget's word
  is the module's verdict for a device role and Hare's connection and alarm
  for a bare PV, read only forever; "unknown" whenever either source is
  silent. PV names come from the module's device route, never derived
  here. A DB device (on the module's roster, in neither device file) binds
  `{device, role: <GEECS variable>}` and is **deferred**: the page judges it
  by the module's device route ("unknown role" when the module lists no
  such role, "wrong kind" when its type does not fit the kind, Hare's alarm
  rules where the module gives no verdict) and writes it through the same
  put route. A writing kind on a bare PV is drawn disabled.
- **A button may name an action plan** (`bind: {action: <plan>}`, checked
  against the experiment's `action_library/actions.yaml`). The module runs
  it, every `set` through its gates; the page asks once (the plan's
  description and its hazard steps) when it writes a hazard role, sends
  the run once, and shows the module's run (`running 2/5`, `failed · step
  3 · out_of_bounds`) with Stop while it runs. A sequence is never
  scripted in a screen.
- **Dense, like the panel it replaces.** When no group spans columns,
  each column packs its groups top to bottom in row order (a LabVIEW
  panel's layout; grid row tracks would leave gaps under short groups);
  otherwise the grid aligns rows. A group's `columns` lays its widgets
  side by side, row by row.
- **Presets are the module's.** The home lists Squirrel's snapshots as the
  module's `/v1/presets` gives them, saves with the header's operator and a
  scope of families and screens (the module resolves both), shows a diff
  drawer (device · role · saved · live · tolerance · word, the module's
  words), restores a device subset as a run the module synthesizes (the
  confirm dialog first; the run shown on the row as on a screen button),
  and marks golden by name. The strip's "Baseline" chip is the golden
  snapshots' differing rows, polled every 10 s, "not compared" (never
  "as baseline") while a row of theirs could not be compared; hidden
  without a golden.
- **The home sorts the module's words; it judges nothing.** "Not live" is
  every device whose state is not `live` (a DB device's state is its
  gateway `connected` PV, so a Disconnected LabVIEW device is listed; a live
  one is not read in detail: its roles carry no module verdict); "In alarm" every device whose
  state or any role's verdict is `error` (the module's word for severity
  major or invalid, and for a tripped fault bit). The strip's run word is
  the manager's `re_state`, as on the scanner's page. A list the module
  could not give reads "unknown", never "none". A row's cause is the module's evidence (heartbeat word,
  roles by verdict). The launcher is `<experiment>/home.yaml` (GEECS-Schemas
  `HomeLayout`); its link URLs are settings, never the file's.
- **The raw frame is the driver's frame; the page transforms it.** The
  camera family's `raw_frame` role is `ARR:ArrayData` (`NDStdArrays`,
  `USHORT`) on `CAM1.CAM`, the driver's own port: before `TRANS1`, before
  background subtraction (`PROC1`). The MJPEG stream, the crosshair store
  (`OVER1`) and the analysis ROI (`ROI1`) are after `TRANS1`, so in Raw the
  page applies the `transform` role's value (read from Hare like every
  value, never a header) to the decoded frame before drawing, and every
  overlay coordinate stays MJPEG mode's. A transform that is not one of
  `TRANS1:Type`'s eight is drawn untransformed with the overlays hidden and
  said. "Raw" means exactly that, and the page says so. Mono frames only.
  A Blackfly's embedded pixels (FlyCapture `FRAME_INFO`) are the camera's
  to turn off, never masked in the page. Never a second `ARR` after
  `TRANS1`, never a write; at most 2 Hz per camera, shared, for the lab
  network (no binned preview). Hare does not carry images; if it does
  later, the frame's *source* moves and the rendering stays. The rest is
  § "The raw frame" below.
- **The GEECS device name is the one identity key.** A binding names a
  device and a family role (and an axis), never a PV a second time; a bare
  PV is for a DB-served device only.
- **No facility value in code.** The experiment, the configs root and
  Canopy's URLs are settings; the home's title is "Control Room" unless
  `GEECS_CONTROL_ROOM_TITLE` says otherwise. The read key is a secret
  (`/etc/geecs/control-room/secrets.env`), never a `site.env` key.
- **The kit's rules** (`GeecsWebTheme/CLAUDE.md`): tokens, never a literal
  colour (this package's templates and static files are in
  `GeecsWebTheme/tests/test_no_literal_colours.py`'s `_SURFACES`); status
  is a dot and a word.

## The raw frame

`GET /api/devices/{name}/frame.raw` (`images.py`) and the page's Raw mode
(`camera.js`, `camera_rules.js`) are one contract, pinned end to end by
`tests/test_raw_page.py` (the route over the fake CA, its bytes through the
page's rules under node):

- **200**: uint16 little-endian, row-major, `raw_height` rows of
  `raw_width`; `Content-Type: application/octet-stream`,
  `Content-Encoding: gzip` from `zlib.compressobj(1, wbits=31)` (gzip
  framing, which `fetch` decodes; bare `zlib.compress` is zlib framing and
  a browser will not); `Content-Length` the compressed length;
  `Cache-Control: no-store`. Headers `X-Frame-Width`, `X-Frame-Height`
  (read in the frame's batch), `X-Frame-Full-Scale` (role `full_scale`,
  read once per device in the background, never in the frame's way;
  omitted while unreadable, and the page then assumes 4095 and says "full
  scale assumed"), `X-Frame-Tag` = the quoted decimal `raw_counter` (the
  page shows it), and `ETag` = the representation (a strong validator):
  `"<counter>"` while the full scale is unread, `"<counter>;<full scale>"`
  once read (`"7;4095"`). No transform header.
- **304** on an `If-None-Match` naming the held `ETag` (weak, a list or `*`
  too): no body, the same `ETag` and `X-Frame-Tag`. A full scale landing
  after the first frame changes the `ETag`, so the page's next conditional
  ask is a 200 with the held body and the new `X-Frame-Full-Scale` (no
  second array read), never a 304 that leaves "full scale assumed" up. The
  page treats 304 as "nothing new", never as an error (`rawAnswer`), draws
  an answer whose `ETag` differs from the one drawn last (`rawIsNew`: the
  same tag at a new scale is drawn again; `new frames/s` counts tags
  only), and drops its held `ETag` on a toggle, so the first raw ask after
  one is never a 304 with nothing drawn.
- **404** when the module names no `raw_frame` (or `raw_width`,
  `raw_height`, `raw_counter`) role: the PV is never derived. **503** with
  words: `the frame counter (<pv>) could not be read`, `the raw frame (<pv>)
  could not be read: <why>`, `geometry disagrees: <n> values for <w>×<h>`
  (the batch is not atomic across a camera-ROI change, so the length is
  checked against the geometry before compressing; a colour frame and an
  empty one fail here too). Never a frame that does not match its headers.
- **Reads**: one in flight per device, the result held `RAW_HOLD_S` (0.5 s)
  and shared. `raw_counter` alone first; equal to the held tag, the held
  frame is served (a 2 Hz poller on a 1 Hz camera costs one scalar read
  per 0.5 s); else the frame, `raw_width` and `raw_height` in one batch,
  then the counter again — on a mismatch the batch is read once more and
  tagged by the second counter. The frame is asked as `DBR_SHORT`
  (`datatype=np.int16`; a `USHORT` waveform's native CA type is
  `DBR_LONG`, int32, twice the bytes on the wire) and its sizes natively;
  a count above 32767 arrives negative and the cast, a `view` as uint16
  (little-endian), restores it exactly: never a clamp. This replaces the
  earlier "`astype`, never `view`" rule, which was for the int32 delivery
  (a view of int32 doubles the element count); a frame that arrives other
  than 16-bit is a 503 (`the raw frame arrived as <dtype>, not 16-bit`):
  a guard on this client's read, not a check of the IOC's `FTVL` (the IOC
  converts any `FTVL` to `DBR_SHORT`; every camera family's `ArrayData` is
  `USHORT`).
  The cast and the gzip run in `asyncio.to_thread` (≈ 15 MB peak per
  read of a 2592×2048 frame).
- **The page**: Raw off by default and never remembered; on, a 2 Hz poll
  with `If-None-Match`; the body is a `Uint16Array` (every platform the
  page runs on is little-endian); the histogram has `full scale + 1` bins,
  counted once per frame; auto-levels take its 0.1/99.9 percent points and
  the histogram plot draws it (`histogramPlot`: Raw's own histogram over
  0..full scale, STAT's `histogram_on` not consulted; MJPEG mode, and Raw
  before its first frame, draw STAT's over `hist_min`..`hist_max`; the
  Raw toggle redraws the plot at once), the LUT has `full scale +
  1` entries over the same maps and scales, and the black/white inputs and
  the cursor read counts. **Levels are per mode**, stored as fractions of
  the mode's full scale (255, or `X-Frame-Full-Scale`), so a white of 255
  from MJPEG never saturates Raw and a 12-bit black never zeroes the JPEG;
  `levels`, `lut` and `autoLevels` take the full scale. The colour map,
  scale and auto flag are shared. Raw's chip is `new frames/s` from tags
  advancing; the `image_counter` chip is hidden. Under auto-levels a lone
  hot pixel clips to the same white as the noise's top 0.1 percent, under
  any scale (the white level clips, not the scale); a white at full scale
  singles it out. The operator page documents this; the maintainer has not
  ruled on changing the rule, so it stays as it is until they do.
- **Rules with rows**: every rule Raw adds is a pure function in
  `camera_rules.js` with a row in `test_camera_rules_js.py` — the
  histogram at a scale, the percentile levels on it, the LUT at 4096,
  stored levels per mode, `transformName` and `transformFrame` for each of
  the eight names (TRANS1's mapping of a W×H frame's (x, y): Rot90
  (H-1-y, x), Rot180 (W-1-x, H-1-y), Rot270 (y, W-1-x), Mirror (W-1-x, y),
  Rot90Mirror (y, x), Rot180Mirror (x, H-1-y), Rot270Mirror (H-1-y,
  W-1-x)), `decodeRaw`, `rawAnswer`, `rawIsNew`, `histogramPlot`,
  `modeWords`, `imageSize` in raw; for the JPEG, `jpegIsNew` and
  `ifNoneMatch` (both loops).
  `camera.js` keeps the DOM work only.

## The JPEG

`GET /api/devices/{name}/frame.jpg` is held on role `image_counter`
(MJPG1's `ArrayCounter_RBV`) the way `frame.raw` is held on `raw_counter`:

- **Reads**: one in flight per device, shared, at most every
  `FRAME_EVERY_S` (0.1 s, the floor on the counter probe). The counter
  first; equal to the held JPEG's and the JPEG younger than `JPEG_REUSE_S`
  (1 s, timed from the request), the held bytes are served with no
  upstream fetch (a 10 Hz poller on a 1 Hz camera costs ten scalar reads
  and one fetch per second, not ten fetches); else the JPEG is fetched,
  **tagged by the counter read before the fetch** (the plugin stores its
  JPEG, then posts the counter, so the JPEG is at least that new and a
  later counter change always fetches again; no second probe, no
  refetch). These are GeecsPanels' `FrameSource` rules (`feed.py`), ported:
  the dependency graph keeps the two packages apart.
- **The reuse cap is the safety net**: should a counter ever be posted
  before its JPEG, the stale frame lasts `JPEG_REUSE_S`, never until the
  next trigger. A stopped camera therefore costs one fetch per second.
- **Headers**: `X-Frame-Tag` = the bytes' CRC-32 (unchanged: the page
  decodes new bytes only); `ETag` = the counter and the CRC
  (`"3;0a1b2c3d"`), so a capped fetch that brings other bytes under the
  same counter is a 200, never a 304 that hides them; **304** on a matching
  `If-None-Match` (`etag_matches`, as the raw route), no body, the same `ETag`
  and `X-Frame-Tag`.
- **No counter, no hold**: the module names no `image_counter`, or the
  probe fails (any CA failure, a NaN): the JPEG is fetched every floor as
  before the hold, served without an `ETag`, and never held. An unreadable
  counter is not asked again for `ROUTE_TTL_S` (a silent one would hold
  every frame for the CA timeout) and never fails the image.
- **The page**: the MJPEG loop asks with `If-None-Match` the last answer's
  `ETag` (`ifNoneMatch`), treats a 304 as "nothing new" (`rawAnswer`),
  decodes an answer whose `X-Frame-Tag` differs from the one drawn
  (`jpegIsNew`; `new frames/s` counts these only), and drops its held
  `ETag` on an error, a decode failure and a toggle, so the next ask after
  one is never a 304 with nothing drawn.

Deploy (bandwidth, the CA array size, the front door's gzip, the Canopy
image rebuild): `README.md` § Deploy.

## The trace

The trace page (`/device/trace/<name>`, family kind `array_detector`) binds
by data shape, never by model: one page for every array over a calibrated
axis (the PicoScope's `scope_contract`, `[channels, samples]` over its time
axis; the spectrometer's `spectrometer_contract`, one `[pixels]` spectrum
over `Wavelength_RBV`, no `channels_by`). What differs per hardware is the family file's: the
rail's groups and the readings strip are its `panel` (each rail role drawn
by its type: an enum a select of the PV's own `enumChoices`, a setpoint a
number with its readback, a writable bit a checkbox, a command a button,
anything else its text; a role the device omits is not drawn), the axis is
the role with `axis_of`, the y array the one it names (`traces.trace_roles`,
the one reading of that rule, for the route and the template). A group whose roles
are all `ch_<x>_…` is hidden while `ch_<x>_present` reads 0 (and its
readings with it); unknown keeps it shown. Controls are filled, enabled and
written exactly as the camera's (`camera_rules.control`, the module's put
route, sent once, a `confirm: hazard` role behind the dialog).

`GET /api/devices/{name}/trace` (`traces.py`) and `trace_rules.js` are one
contract, pinned by `tests/test_traces.py` and `tests/test_trace_rules_js.py`
(the route's own bytes decoded under node):

- **200**: float32 little-endian, gzip framed (`zlib.compressobj(1, wbits=31)`,
  as `frame.raw`): the axis (`samples` values), then the y array as the IOC
  lays it out, `channels × samples`, channels fastest (a disabled channel
  NaN); headers `X-Trace-Channels`, `X-Trace-Samples`, `X-Trace-Tag` = the
  quoted counter, `ETag` the same. **304** on a matching `If-None-Match`.
  **404** when the family names no axis or no `fresh_by` for its array, or
  the module names no such role; **503** with words: the counter or the
  array unreadable, `shape disagrees: <n> values for <c> channels`, the axis
  unreadable or `has <m> values for <s> samples`.
- **Reads**: one in flight per device, held `TRACE_HOLD_S` (0.5 s), shared.
  The counter first (equal to the held tag: the held trace, one scalar
  read); else the array asked `DBR_FLOAT` and the channel count in one
  batch, the counter again (a mismatch reads the batch once more, tagged by
  the second). The axis is read with the first trace and again only when
  its length is not the trace's, its first two values moved (a two-value
  probe per new capture: a new interval or pre-trigger moves them; a whole
  axis would be a third of a 2 × 100 k capture's bytes) or it is older than
  `AXIS_EVERY_S` (10 s). The counter is the y array's `fresh_by`, the
  producing plugin's own (`ARR:ArrayCounter_RBV` for the scope): the
  driver's posts before `ARR` publishes, so a trace tagged by it could hold
  the previous capture under a fresh tag.
- **The page**: polls at 2 Hz while visible, `If-None-Match` the last
  `ETag`, 304 is nothing new; the chip is the counter role's word; the
  scale, the window and the average (the last N captures of one shape and
  one axis, a tag counted once, finite values only) are the viewer's,
  written nowhere. What the route serves is unchanged by how the page
  draws: the plot is **Plotly through the theme's template**
  (`GeecsWebTheme/CLAUDE.md` § "Line plots are Plotly"; the bundle and
  `plot.js` from `/theme`), one trace per channel in channel order (an off
  channel one null point named `<x> off`, so trace *i* keeps
  `--trace-(i+1)`), each decimated to two points per pixel column of the
  plot (`plotData` over `decimate`; on a log scale a column's positive
  extrema only), `Plotly.react` on each new capture with `uirevision` (the
  viewer's zoom and legend toggles survive captures; x's revision is the
  capture's shape, y's the scale). The window is Plotly's x range: a box
  zoom or pan (`windowFromRelayout`) re-decimates inside it; autoscale,
  reset or a double-click is the whole trace. Axis titles carry the
  family's units; SI prefixes on ticks and in the hover for base SI units
  only (`axisFormat`: s, V, A, Hz, m, W), every other unit plain (the
  first cut prefixed every unit and a wavelength axis read "knm"). The
  cartesian bundle has no `scattergl`; decimation is why SVG `scatter`
  holds a 2 × 100 k capture at 2 Hz.

The benches are `scripts/scope_soft_ioc.py` (caproto through `uvx`, loopback,
every role of the scope contract, a synthetic `[3, samples]` capture at
1 Hz, the counters in the IOC's order; channel C disabled, D absent) and
`scripts/spectrometer_soft_ioc.py` (every role of the spectrometer
contract, a 2048-pixel spectrum over a wavelength polynomial at 1 Hz, the
chain's background path: `PROC:SaveBackground` holds the raw spectrum,
`PROC:EnableBackground` subtracts it, STAT1 over the result): geecs-epics
has no simulator for either.

## Settings

| Key | Meaning |
|---|---|
| `GEECS_CONTROL_ROOM_EXPERIMENT` | the experiment (unset: `GEECS_EXPERIMENT`) |
| `GEECS_CONTROL_ROOM_CONFIGS_ROOT` | the configs repo root (unset: `GEECS_CONFIGS_ROOT`) |
| `GEECS_CONTROL_ROOM_CANOPY_URL` | Canopy as this process reaches it |
| `GEECS_CONTROL_ROOM_CANOPY_BROWSER_URL` | Canopy as the browser reaches it (the front door's `/canopy`); unset: the above |
| `GEECS_CONTROL_ROOM_READ_KEY` | the display's Canopy read key (secret) |
| `GEECS_CONTROL_ROOM_BIND`, `_PORT` | where it listens (127.0.0.1, 8700) |
| `GEECS_CONTROL_ROOM_TITLE` | the page title (default "Control Room") |
| `GEECS_CONTROL_ROOM_SCANNER_URL` | the scanner as this process reaches it (the strip); unset: "not configured" |
| `GEECS_CONTROL_ROOM_SCANNER_BROWSER_URL` | the scanner link as the browser reaches it; unset: the above |
| `GEECS_CONTROL_ROOM_PORTAL_URL`, `_LOGBOOK_URL`, `_CONFIG_EDITOR_URL` | the launcher's links (browser URLs); unset: a disabled entry |

## Testing

`poetry run pytest tests -q`: the module is stubbed with httpx's
`MockTransport`; `test_images.py` drives the frame, raw frame and arrays
routes over a fake CA and a fake fetch (no CA library, no IOC): the raw
frame's bytes are checked on the wire (`aiter_raw`, then `gzip.decompress`),
since httpx would decode the gzip itself, and the fake CA's `counters` script
the counter probe and the mid-batch retry. The page scripts run under node (skipped without it; no
npm, no jsdom): `test_hare_js.py` with a fake socket, `test_poller_js.py`
with a fake fetch, and `test_screen_rules_js.py` / `test_home_rules_js.py` /
`test_device_rules_js.py` / `test_camera_rules_js.py` as tables, one row per rule (`tests/_node.py` is the harness);
`test_raw_page.py` drives the raw path end to end: a 600×600 dark floor with one hot pixel served
by `frame.raw` over `test_images.py`'s fake CA (as int16, as aioca delivers it asked `DBR_SHORT`), its bytes checked on the
wire, then decoded, transformed and levelled by `camera_rules.js` under node. A rule a
page needs goes in a `*_rules.js` file with a row here; the page script
keeps the DOM work only.

`test_module_contract.py` is the cross-service contract: GeecsCanopy's own
module (`geecs_canopy.testing.make_env`: its routes over a fake Hare) mounted
at `/geecs` behind `ModuleClient`, driven through this app's routes. It pins
the fields the pages read (a role's `role`, `axis`, `pv`, `read`, `writable`,
`verdict`, and for the camera `label` and `locked`; the image and arrays read
at the PVs the module names; the put envelope's `outcome`, `code`, `journalId`, `detail`; a
journal row's fields) and the home's lists and `unread`. geecs-canopy is a
dev-only dependency; `geecs_control_room` never imports it.
