# Raw frames — the brief (integration branch `feat/raw-frames`)

Scaffolding per `Planning/README.md`: the promotion PR to `master` deletes this
file after moving anything load-bearing into the owning `CLAUDE.md` files. The
coordinator runs the slices, one PR each into this branch, with the `/land`
ritual, and merges a lane PR after its review is dispositioned (maintainer's
rule for integration branches). `master` is merged forward at least weekly.
Promotion to `master` is the maintainer's merge.

## Goal (the maintainer's rulings, 2026-11-03)

The camera page shows the camera's **raw 12-bit frame** without compression
artifacts, with black and white levels, a histogram and the page's colour maps
and scales applied on the 12-bit values, so an operator can see the dark
floor's noise ("snow") and a lone hot pixel as one pixel. The MJPEG stream
stays the default view; raw is a toggle on the page.

Why: the IOC's display chain (`PROC2` 12→8 bits, `MJPG1` JPEG q80, `MAXW`/`MAXH`
1600×1200) flattens a Blackfly's dark floor to 3–4 levels and smears an impulse
into an 8×8 tile. Measured 2026-11-03: UC_UndulatorRad2's 2592×2048 frame is
10.6 MB as 16-bit, reads over CA from the services box in ~0.2 s (aioca
delivers it as int32, 21 MB), compresses to ~3.8 MB with zlib level 1 in
~0.1 s (3.4 MB zstd-3; zlib level 6 takes 1.3 s and is out); a Blackfly's
600×600 ROI is 0.72 MB raw, 0.2 MB compressed, 20 ms to read. Everything runs
at 1 Hz under the beamline trigger.

Rulings this arc builds on:

- **Rate**: raw at most 2 Hz per camera, shared across viewers; MJPEG up to
  5 Hz. ("5 Hz? that's more than enough. could even bring it down.")
- **Scope**: the lab network only; no binned preview for the VPN.
- **Topology**: frames relay through the services box (one CA read per camera,
  compressed fan-out; 10 GbE bond). Budget for 20 streams: ~0.2 Gbit/s MJPEG,
  ~1.3 Gbit/s if every stream were raw at 2 Hz; the maintainer's ceiling is
  ~5 Gbit/s so analysis reads stay fast. This overturns the 2026-10-30
  "direct IOC→workstation" ruling, made when the box was ~1 Gb.
- **Colour maps**: the page's existing grey/viridis/inferno/jet and
  linear/sqrt/log scales with percentile auto-levels. No LabVIEW ports.
- **Embedded Blackfly pixels** (FlyCapture `FRAME_INFO`): fixed at the camera,
  never masked in the page.
- **Default**: MJPEG. The IOC-side caps and quality are a geecs-epics change
  (lane `mjpeg-caps`, geecs-epics#122), not this arc.
- **Not doing**: a PVA client in the browser (a page has no sockets), an
  areaDetector WebSocket plugin (ours to maintain, redundant if Canopy's Hare
  gains images — slaclab/canopy#88, open). If Hare carries images later, the
  page's frame *source* moves; the rendering stays.
- The Control Room reads frames and arrays over Channel Access itself, read
  only, at PVs the module's device route names (option A, 2026-11-01;
  `GeecsControlRoom/CLAUDE.md`). The raw frame is the third such read.

## Where the raw frame comes from (geecs-epics, the chain contract)

Every camera IOC's `ARR` plugin (`ADCore.NDStdArrays`, `FTVL: USHORT`,
`TYPE: Int16`, `EnableCallbacks 1` in the startup) sits on **`CAM1.CAM`**, the
driver's own port, beside `HDF1`, `PVA` and `STAT0` (geecs-epics
`services/undulator/cameras/<ioc>/config/ioc.yaml`, the chain comment; the
runbook's "Display and analysis chain"). Consequences the page must honour:

- The raw frame is the **driver's frame**: no background subtraction (`PROC1`
  is downstream of `TRANS1`), no transform. "Raw" means exactly that, and the
  page says so.
- The MJPEG stream, the crosshair store (`OVER1`) and the analysis ROI
  (`ROI1`) live **after `TRANS1`**. In raw mode the page applies the
  `transform` role's value (the `TRANS1:Type` enum: None, Rot90, Rot180,
  Rot270, Mirror and the three Rot+Mirror) to the decoded frame before
  drawing, so every overlay coordinate stays the one MJPEG mode uses. A
  pure rule (`camera_rules.js`) maps a transform name to the pixel mapping,
  pinned by rows; a frame whose transform is not one of the eight is drawn
  untransformed with the overlays hidden and a word.
- `ArrayData` is unsigned 16-bit (`USHORT`); aioca returns it as int32. The
  server casts with `astype(np.uint16)` — exact for every USHORT value;
  never `view()` (a reinterpretation of int32 doubles the element count),
  never a clamp. The cameras' data is 0–4095; the contract does not assume
  it. **Mono frames only**: `ArraySize0` is the width; a colour pixel format
  (`ArraySize0` = 3) is out of scope and fails closed on the geometry check.

## What changes

**GEECS-Schemas, the camera family** (`ad_camera_chain.json`) — landed in
slice 0:

| role | suffix | type | read | notes |
|---|---|---|---|---|
| `raw_frame` | `ARR:ArrayData` | `array` | `never` | the frame; the bridge never reads it, the frame route does, by this name (as `image` and `image_counter` are the page's well-known names); `array` is never writable and never archived; `read: never` keeps it out of `/arrays` |
| `raw_width`, `raw_height` | `ARR:ArraySize0_RBV`, `ArraySize1_RBV` | `readback`, expert | `on_request` | the frame's geometry as `ARR` serves it (the driver's, before `TRANS1`); read with the frame, carried in its headers |
| `raw_counter` | `ARR:ArrayCounter_RBV` | `counter`, expert | `on_request` | the frame's identity and the cheap probe; read with the frame |

None of the four is monitored, judged, subscribed on Hare, or archived: the
module's `_judged` is `read_mode == "monitor"`, so the device's state does not
depend on `ARR` (an `ARR` with callbacks off cannot make a live camera read
stale), and Hare's subscription count per camera is unchanged. Pinned by the
family test, the Control Room contract test, and `images.arrays()`, which now
serves `on_request` roles **of type `array`** alone (slice 0). The one other
family with on-request scalars is `scope_contract` (`arr_channels`,
`arr_samples`, beside its `data` trace): nothing reads them through `/arrays`
today, and the scope page's own route reads them with the traces (noted in
`Planning/device_pages.md`). A family change
rebuilds Canopy's image at deploy (the module's roles come from the image's
copy of the schemas).

**GeecsControlRoom, the server** (`images.py`): `GET /api/devices/{name}/frame.raw`.

The contract, pinned here so R1 and R2 can run in parallel against one
fixture; both lanes implement it from this text alone:

- **200**: the frame as **uint16, little-endian, row-major, `raw_height` rows
  of `raw_width`**, `Content-Type: application/octet-stream`,
  `Content-Encoding: gzip` — gzip framing, produced with
  `zlib.compressobj(1, wbits=31)` (never bare `zlib.compress`, which is
  zlib framing and a browser will not decode); the browser's `fetch`
  decodes it transparently and `arrayBuffer()` is the uint16 bytes.
  `Content-Length` is the compressed length. `Cache-Control: no-store`.
- Headers: `X-Frame-Width`, `X-Frame-Height` (from `raw_width`/`raw_height`
  read in the same batch), `X-Frame-Full-Scale` (from `full_scale`, read
  once per device; omitted when unreadable, and the page then assumes 4095
  and says "full scale assumed"), `X-Frame-Tag` and `ETag` both set to the
  quoted decimal `raw_counter` value at the read (`"12345"`; a strong
  validator). `X-Frame-Transform` is **not** sent: the page reads the
  `transform` role from Hare as it does every value.
- **304** when the request carries `If-None-Match` equal to the held frame's
  `ETag`: no body, the same `ETag` and `X-Frame-Tag`. The page's `fetch`
  sees `status === 304` with `ok === false`; the page treats 304 as "no
  new frame", never as an error (today's JPEG loop treats non-ok as an
  error; the raw loop does not share that branch).
- **404** when the module names no `raw_frame` role for the device (the
  frame is never derived). **503** with words (the kit's vocabulary) when a
  source is silent: `the frame counter (<pv>) could not be read`, `the raw
  frame (<pv>) could not be read: <why>`, `geometry disagrees: <n> values
  for <w>×<h>` (the batch is not atomic across a camera-ROI change; the
  server checks `len == width*height` before compressing and never serves
  a frame that does not match its headers).
- **Reads**: one in flight per device, the result held for `RAW_HOLD_S =
  0.5` and shared (`Shared`, as the JPEG). A read is: `raw_counter` alone
  first (a scalar); if it equals the held tag, the held frame is served
  again (so a 2 Hz poller on a 1 Hz camera costs one scalar read per
  0.5 s, not a 10 MB array read); else the frame, `raw_width`, `raw_height`
  in one batch `caget`, then `raw_counter` again — on a mismatch the batch
  is retried once, then served with the second counter as the tag (the
  frame may be one newer than its headers say; the geometry check is what
  must hold). The cast and the compression run off the event loop
  (`asyncio.to_thread`; numpy's cast and zlib release the GIL). Peak memory
  per camera read ≈ 36 MB (21 MB int32 + 11 MB uint16 + 4 MB gzip); the
  int32 array is dropped as soon as the cast is done.
- Never a write; CA addressing is the host's `EPICS_CA_*`. The client-side
  `EPICS_CA_MAX_ARRAY_BYTES` was unnecessary in the Canopy container (aioca
  sizes its buffer); R1 confirms in the Control Room's own environment and
  writes the fact into `DEPLOYMENT.md` either way.

**GeecsControlRoom, the page** (`camera.js`, `camera_controls.js`,
`camera_rules.js`, `device_camera.html`, `camera.css`):

- A **Raw** toggle beside the image header. Off: today's JPEG loop. On: a
  poll at 2 Hz of `frame.raw` with `If-None-Match`, 304 = nothing new; the
  body a `Uint16Array` (little-endian: `new Uint16Array(buffer)` on every
  platform the page runs on is little-endian; a `DataView` path is not
  needed and not written); the `transform` role applied (above); `lum`
  becomes 16-bit; the histogram has 4096 bins (or `full scale + 1`);
  auto-levels take the 0.1/99.9 percentiles on 0–full scale; `lut` becomes
  a `full scale + 1` entry table over the same maps and scales; the
  black/white inputs read in counts of the full scale.
- **Levels are per mode**, stored as fractions of the mode's full scale
  (255 or `X-Frame-Full-Scale`), so a stored white of 255 from MJPEG mode
  never saturates the raw view and a 12-bit black never zeroes the JPEG;
  `levels` and `autoLevels` take the scale as an argument. The inputs'
  `max`, Reset and the stored values follow the mode.
- Zoom, pan, cursor readout (in counts), crosshairs and the analysis ROI
  work unchanged on the transformed raw geometry; `imageSize` reports true
  size = served size in raw mode (no stretching). Toggling mid-zoom keeps
  the view (both modes share the post-transform geometry).
- The freshness chip in raw mode is "new frames/s" from tags advancing (as
  the JPEG's checksum rate today), plus the module's `image_counter` word is
  no longer relevant and is hidden; the hint "The 8-bit stream, stretched
  in this browser" reads "The camera's 16-bit frame, levels applied in
  this browser; nothing is written to the camera" in raw mode.
- The rules go in `camera_rules.js` with a node-test row each: the
  histogram at a given scale, the percentile levels on it, the LUT at
  4096, the transform mapping for each of the eight names, the decode of
  a fixture frame (geometry, a known pixel), the toggle's state words, the
  304 handling.
- No change to writes, the module, or Hare.

## Slices

0. **Slice 0, the ground** (the coordinator, landed with this brief): the
   four family roles with their read modes and the tests above;
   `images.arrays()` narrowed to `type == "array"` with a test; the
   contract test pinning the raw roles' `read` and `verdict: null`; this
   contract.
1. **R1, the server** (one lane): `frame.raw` in `images.py` per the
   contract, with tests over the fake CA and a fake module route (a
   2592×2048 and a 600×600 fixture; the counter probe serving the held
   frame; the 304; the 404 and the three 503 words; the geometry
   disagreement; one read in flight under concurrent requests; the gzip
   framing checked by decompressing with `gzip`); the `MAX_ARRAY_BYTES`
   fact in `DEPLOYMENT.md`; `GeecsControlRoom/CLAUDE.md`'s `images.py` line.
2. **R2, the page** (one lane, in parallel with R1, developing against a
   fixture served by a test-only route that honours the contract above,
   including 304 and gzip framing): the toggle, the transform, the 12-bit
   rendering, the per-mode levels, the rules and their rows, the chip and
   hint, the CSS. Bench proof on the Mac sim stack is **plumbing only**:
   ADSimDetector serves 8-bit frames, so the 12-bit claims rest on the node
   fixture rows and on the hardware proof below. Screenshots light and
   dark on `screenshots/rf-s2-raw` (the fixture frame: a dark floor with
   noise and one hot pixel, so the screenshots show the point).
3. **R3, docs and promotion** (the coordinator): the operator page's camera
   section (what raw shows, the toggle, the rate, "raw means the driver's
   frame"), `DEPLOYMENT.md`'s bandwidth note, the front-door note (a proxy
   in front of the Control Room must pass `Content-Encoding: gzip` through
   unchanged), the promotion PR to `master` with the `/land` record, then
   the box deploy (Canopy image rebuild + Control Room restart) and the
   hardware proof.

R1 and R2 meet on this branch; the integration check is R2's page against
R1's endpoint on the sim stack before the promotion PR.

## Proof of done

- On the services box, read-only, by the coordinator after the deploy:
  UC_UndulatorRad2 and one Blackfly in raw mode: the dark floor's noise
  visible under the log scale with auto-levels; a lone hot pixel one pixel
  wide at 1:1; the histogram spanning the 12-bit values; the tag advancing
  at 1 Hz; overlays coinciding with MJPEG mode's on a camera whose
  `transform` is not None (if one exists; else the rule's rows are the
  proof); `frame.raw` for Rad2 ≤ 4 MB per frame and ≤ 0.3 s from request to
  body.
- Twenty raw viewers on one camera (a scripted fetch loop) cost one scalar
  CA read per 0.5 s and one array read per new frame on the box, not twenty.
- The MJPEG view unchanged; the module, Hare and every write untouched;
  the archiver's registry pass unchanged for the camera (no new PV archived:
  pinned in slice 0).

## Out of scope

The VPN (binned preview, ROI-only raw), the PVA path (p4p), the IOC-side caps
and quality (geecs-epics#122), the Hare image kind (upstream), saving frames to
disk, any colour map not already on the page, a second `ARR` after `TRANS1`
(the page transforms instead), GeecsPanels.

## Size and cost rules

R1 ≈ 450 lines of code plus tests, R2 ≈ 550 plus node rows; either lane past
1.5× its budget stops and reports. Each lane PR carries its LOC by concern,
its review dispositions, and its hardware OWED line. Bench proof is the Mac
sim stack or the lane's own containers, never a production camera; the
production proof is the coordinator's after the deploy, read-only.

## Status

### 2026-11-03 — the brief, red-teamed; slice 0

Written from the exploration of 2026-11-03 and the maintainer's rulings the
same evening. A fresh-context red team read it against the code before any
lane started; its findings and their dispositions:

| # | Finding | Disposition |
|---|---|---|
| 1 | Blocker: the byte contract was ambiguous — "int16, clamp negatives" (wrong for `USHORT`), "gzip (zlib level 1)" (zlib framing is not gzip), no tag when the counter is unreadable, `If-None-Match` quoting and the 304 under `fetch`, no check that body length matches the headers, `ARR`'s chain position asserted not cited | Fixed in the contract: uint16 by reinterpretation, `compressobj(1, wbits=31)`, `ETag` quoted strong, 503 words for the counter and for geometry, 304 as "nothing new"; `ARR` on `CAM1.CAM` cited from the IOC config, with the transform consequence (a new section) |
| 2 | Blocker: a `frame` role type with `read_mode on_request` would land the 10 MB frame in `/arrays` as JSON | No new type: `raw_frame` is `array` + `read: never`; `images.arrays()` narrowed to `type == "array"`; both pinned (slice 0) |
| 3 | Should-fix: "no change to the module or Hare" was false — monitored geometry/counter roles add three Hare subscriptions per camera and couple the device's state to `ARR` | All three scalars `read: on_request`: none monitored, judged or subscribed; the contract test pins `verdict: null`; `raw_counter` is read with the frame and is the tag |
| 4 | Should-fix: the 304 saves bandwidth, not the box — a 2 Hz poller forces a 10 MB read every 0.5 s regardless | The counter probe first, the frame only on a new counter, re-read after and retry once; memory stated (≈36 MB per read) |
| 5 | Should-fix: the page's stored levels and `levels()` are 8-bit; a stored white of 255 saturates raw | Levels per mode as fractions of the mode's full scale; `levels`/`autoLevels` take the scale |
| 6 | Should-fix: the sim cannot prove the goal (8-bit frames, no snow, no hot pixel) | Said in R2 and § Proof: sim = plumbing; the claims rest on fixture rows and the hardware proof |
| 7 | Nits: the 8-bit hint text; `ETag` beside the custom header; a front-door proxy may re-encode gzip | All three in the contract, the page section and R3 |

The slice-0 code review (fresh context, on the PR) added: the `/arrays` narrowing also drops `scope_contract`'s two on-request scalars (no consumer today; the scope route reads them — noted here and in `device_pages.md`); `astype`, never `view`, for the cast; mono frames only; three stale docstrings fixed.

Verdict before fixes: "not yet; land a slice 0 first". Slice 0 is this
commit; R1 and R2 start after it.
