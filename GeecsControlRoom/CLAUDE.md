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
                    names, rehosted to the IOC's CA address, one fetch in flight, shared) and arrays
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
  static/poller.js  the pages' one poll loop: one request in flight per target, late answers
                    dropped, a timeout, back-off to 10 s, at most 8 requests at once
  templates/        base.html (every page's shell), screen.html (EVERY screen), device.html (every
                    device page until a kind gets its own, filling `kind_component`; device_camera.html the camera's), the pages
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
  frame and its arrays are read over Channel Access by this server
  (`images.py`, aioca), **read only**, at the PVs the module's device route
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
  than 16-bit is a 503 (`the raw frame arrived as <dtype>, not 16-bit`).
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
  `modeWords`, `imageSize` in raw.
  `camera.js` keeps the DOM work only.

Deploy (bandwidth, the CA array size, the front door's gzip, the Canopy
image rebuild): `README.md` § Deploy.

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
