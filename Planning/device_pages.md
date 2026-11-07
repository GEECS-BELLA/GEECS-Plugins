# Device pages — the brief (integration branch `feat/device-pages`)

Scaffolding per `Planning/README.md`: the promotion PR to `master` deletes this
file after moving
anything load-bearing into the owning `CLAUDE.md` files. The coordinator runs
the slices in order, one PR each into this branch, with the `/land` ritual, and
merges a lane PR after its review is dispositioned (maintainer's rule for
integration branches). `master` is merged forward at least weekly. Promotion
to `master` is the maintainer's merge; the GeecsPanels deletion was its own PR
under the kill rule, after the maintainer had looked at the rendered pages
(slice 6, done 2026-11-06: § Status).

## Goal (the maintainer's acceptance criterion, 2026-10-31)

Each kind of hardware gets ONE general front panel that can connect to ANY
device of that kind, configure it, and show its readbacks, traces and images:
first the **camera** (the areaDetector chain) and the **oscilloscope** (the
PicoScope IOC's scope contract), then the motor and the power supply from the
family files. The device is chosen on the page from a dropdown of the
registry's devices of that kind (`native_devices.yaml` + `panel_devices.yaml`
through GEECS-Schemas), with a free-text device name as the fallback. Adding
a device is a registry row, never code. These pages live in
**GeecsControlRoom** as its device pages (`/device/<kind>/<name>`, the
drill-down from the home and from a screen's device), and **GeecsPanels is
deleted** when they are proven: one operator surface, no bench tool.

## What moves and what changes

- **Scalars and writes go through Canopy**: live fields over Hare, every write
  through the GEECS module's put route with the operator's name, verdicts from
  the module's device route, exactly like a screen. GeecsPanels' own CA client
  (aioca), its per-kind allow-lists and its bench file retire: the module's
  policy, allow, hazard and run-lock gates are the one write gate.
- **Images and arrays do NOT go through Hare** (it carries scalars). The
  components that make GeecsPanels' camera and scope pages work MOVE into
  GeecsControlRoom as components with their own small routes: the frame proxy
  that pulls the IOC's JPEG one frame at a time, the client-side colour map,
  levels and ROI, the freshness judgement by the plugin's `ArrayCounter_RBV`,
  and the scope traces streamed over SSE. Copied, adapted to the kit, and the
  originals deleted with the package in slice 6 (done 2026-11-06).
- **The camera page's feature list** is `Planning/operator_surfaces.md` §6
  (ROI, exposure, gain, trigger mode and delay, rate, last update, max counts
  against saturation, crosshairs, centroid/sigma/max/mean/total, lineouts and
  histogram, colour table and autoscale, the save toggle, device errors). The
  three recorded gaps (spatial low-pass, perspective correction, FWHM and 45°
  moments) are NOT built here; the page shows what the IOC chain serves.
- **Rules that bind**: `GeecsWebTheme/CLAUDE.md` (kit, tokens, never a literal
  colour, the overlay ladder); `GeecsControlRoom/CLAUDE.md` (a screen binds,
  groups, confirms and calls, never computes; plain JS, no build step; the
  rules modules under node tests; the contract test over
  `geecs_canopy.testing`); the import graph in root `CLAUDE.md` (GeecsCanopy a
  dev-only dep, nothing from GeecsBluesky, the portal); facility
  values in config only; no IOC address in code (the geecs-epics chain
  contract names the IOC).

## Slices

1. **Inventory (no code).** Per kind, what GeecsPanels renders and from which
   PVs; what Hare and the module already cover; what the page needs that
   neither carries (image, traces, freshness). The device-selection source and
   the kind → template mapping. Write it into § Status.
2. **Device picker and routing.** `/device/<kind>` lists the registry's devices
   of that kind in a kit `select` with the free-text fallback; `/device/<kind>/<name>`
   renders the kind's page with the device resolved against the module's
   roster (not live → the kit's `stale`/`denied` states, never a blank).
   Rules module + node tests; contract test through the module's routes.
3. **Camera page.** Scalars and writes through Hare and the module; the
   moved frame proxy + colour map/levels/ROI component; freshness chip;
   screenshots light and dark on a `screenshots/*` branch (the maintainer
   reviews by looking).
4. **Trace page** (was "scope page"; § Status 2026-11-05). One page for every
   array over a calibrated axis (one channel, or `[channels, samples]`) (a scope's capture over its time axis, a
   spectrometer's spectrum over its wavelengths): the rail and the readings
   strip come from the family's `panel`, the y array is the `axis_of` target
   of the axis role (`[channels, samples]` by its `channels_by`); the trace
   polled at 2 Hz, held on the y array's `fresh_by` counter (not SSE);
   screenshots.
5. **Motor and supply pages** from the family files' roles (small; the
   inspector's rich view and the drill-down).
6. **GeecsPanels deletion PR** (separate, under the kill rule): the package,
   its unit template, its row in root `CLAUDE.md`, `.importlinter`,
   `scripts/check.sh`, `scripts/ci_select.py`, the theme's template walk, the
   docs page, the CHANGELOG list. **Done 2026-11-06**, one PR to `master`
   after the maintainer's look (§ Status).
7. **Docs + promotion body.** `GeecsControlRoom/CLAUDE.md` gains the device
   pages and the image/array rule in one paragraph; `operator_surfaces.md` §6
   is folded into it and the section deleted; the promotion body lists lane
   PRs, screenshots and the open decisions.

## Proof of done

On the Mac sim stack (geecs-epics `hosts/sim-undulator`: the areaDetector sim
camera; it has no scope simulator, so the scope page is proven on a bench-only
soft IOC serving the scope contract on loopback, § Status 2026-10-31; isolation
rule respected) the camera page connects to
any listed camera, shows a live frame with colour map, levels and ROI, and
writes exposure through the module with the write journaled; the scope page
shows live traces and sets a channel range the same way; both pages render in
light and dark with no literal colour (the theme's tests walk the templates).
Live proof on the IOC host's prototype camera is the maintainer's call.

## Out of scope

The three camera gaps; an image path inside Canopy (an upstream question);
composed screens (they exist); any new Canopy module route beyond what the
pages need; Phoebus or any non-web GUI.

## Size and cost rules

A slice ≤ ~600 changed lines (slice 3 may reach ~900 with the moved
component; say so in the body). Reviews bounded; fresh verifier for fixes. No
GitHub issues, no comments outside the lane PRs, no hosts or addresses in any
committed file or PR body.

## Status

(The coordinator appends dated entries here.)

### 2026-10-31 — slice 1, the inventory (no code)

**Kind → family → page.** The URL's kind word maps to a family file's
`kind` (GEECS-Schemas `load_families()`), and from there to one template:

| `/device/<kind>` | family `kind` | families today | page |
|---|---|---|---|
| `camera` | `aravis_camera` | `ad_camera_chain` (82 roles) | `device_camera.html` |
| `scope` | `array_detector` | `scope_contract` (71 roles) | `device_scope.html` |
| `motor` | `epics_motor` | `motor_record` (29 roles, per axis) | `device_motor.html` |
| `supply` | `setpoint_device` | `easydriver` (8), `fastps` (10) | `device_supply.html` |

A new family of an existing kind gets the kind's page with no code; a new
kind is a template and a row in that table.

**Device selection.** The picker lists the registry: `native_devices.yaml` +
`panel_devices.yaml` read through GEECS-Schemas (`NativeDevices`,
`PanelDevices`, joined by device name as `screens.load` already reads them),
a device's kind from its panel entry's family, else from its native
`kind` (the same words: `aravis_camera`, `epics_motor`). Each row carries the module's roster
state (`GET /geecs/v1/devices`) when the module answers, `unknown` when it
does not. The free-text name is the fallback for a device the files do not
list (a bench device on the module's roster, a typo): it is resolved against
the module like any other.

**Resolution.** `/device/<kind>/<name>` reads the module's device route; the
page never derives a PV. Not on the roster (404) → the kit's `denied` pane
("not on the module's roster"); a family of another kind → `denied` ("wrong
kind: `<family>` is a `<kind>`"); a roster device whose family is no family
file (a DB device, `source: gateway`: its family is the DB device type) →
`denied` ("not a native device"); the module silent → `stale`/`unknown`,
never a blank page. The page draws the device's **resolved** roles (`PanelDevice.resolve`: a
device may omit a family role or relabel it; the module's device route serves
the same list), taking `level` from the family (the operator set open,
`expert` folded in a `details`), and the script fills each from the device route (`pv`, `writePv`, `writable`,
`locked`, `verdict`) and Hare (value, alarm, connection).

**What each GeecsPanels page renders, and who carries it now.**

| Kind | GeecsPanels renders (from) | Hare + the module carry | Neither carries (moves into GeecsControlRoom) |
|---|---|---|---|
| camera | state, rate, exposure, pixel format, centroid/sigma/max/mean/total, raw max (`cam1`, `STAT`, `STAT0`); counters of `cam1`/`STAT`/`STAT0`/`MJPG1`; crosshairs (`OVER1:1/2`), analysis ROI (`ROI1`), cursor, background + transform (`PROC1`, `TRANS1`), run lock, capture, the driver's exposure/period/gain/trigger mode/acquire/camera ROI | every scalar above is an `ad_camera_chain` role with its verdict (freshness by the producing plugin's counter, `off` by the `Compute*` flag: the module's `freshness.py`); writes by role through `put_role` (policy, allow, hazard, the run lock as `locked` roles and a 423) | the **frame** (role `image`, `read: once`, the `JPG_URL_RBV` string: the frame proxy, one frame in flight, rehosted to the IOC's CA address); the **arrays** (`profile_x/y`, `cursor_profile_x/y`, `histogram`, `read: on_request`: read on request, shared, at most every 0.5 s, only while a page has the plots open); colour map, levels, zoom/pan, the ROI and crosshair layer (client side) |
| scope | identity, state, acquire, image mode, timebase (interval, samples, pre-trigger), trigger (mode, source, level, edge, auto ms), per channel present/enable/range/coupling/offset/overrange and `STAT_<CH>` stats; enum choices from the PVs | every one a `scope_contract` role; enum choices are Hare's `enumChoices` meta; writes through `put_role` | the **traces** (`data` = `ARR:ArrayData` [channels, samples], `time_axis`): read once per new capture (`arr_uid`), split per present channel, min/max-decimated to the screen width, streamed over SSE |
| motor | per axis `.RBV/.VAL/.DMOV/.HLM/.LLM`, switches, `.MSTA` by name, `.VELO/.VMAX`, tweak, stop, home | every field a `motor_record` role per axis; the family's actions (`move`, `tweak`, `stop`, `home`, `velocity`) and a device's `allow` gate the writes in the module | nothing |
| supply | the roles (current sp/readback, voltage, output sp/readback, reset, loop mode) and the fault bits, from the panels' bench file | the `easydriver`/`fastps` roles; the drive limits (`DRVL`/`DRVH`, or the role's `limits`) as the module's bounds (422) | nothing on the page; one guard is lost (below) |

**What does not move (cheap to veto).** Each is a computation the page would
own, so each is the module's to give or is dropped:

- the camera's **saturation chip** (raw max against the pixel format's full
  scale) and its **history plot** (centroid/mean/max kept in the service):
  OWED to the module (a verdict) and to Vole (trends); the page shows raw max
  and the pixel format;
- **"crosshair to centroid"** (converts STAT's ROI pixels to image pixels on
  the server): OWED, a module action; crosshairs are set by value;
- the **run-lock override** (writes `RUN:Lock` 0): the module's `lock` role is
  read only, and its 423 says why a driver write is refused;
- the motor's **`.MSTA` decoded by name** and the supply's **state word**: the
  page shows the module's verdicts and the raw word; a decoded word is the
  module's to add;
- the supply's **HIHI/LOLO pre-refusal** (GeecsPanels refuses a setpoint at or
  past `HIHI`/`LOLO` when that severity is INVALID; the module's bounds are the
  drive limits only): OWED to the module's gates (`gates._bounds`), where every
  page and action plan gets it; until then the IOC's `IVOA` is the refusal;
- GeecsPanels' bench file (`allow`, `note`, bench supplies), `--scope
  NAME=PREFIX`, `--read-only` and its own CA writes: the module's policy and
  `panel_devices.yaml` replace them.

**The image/array path (a decision, cheap to veto).** The Control Room's
server reads the frame URL and the arrays over Channel Access itself (aioca),
at the PVs the module's device route names, behind its own small routes:
`/api/devices/{name}/frame.jpg`, `/api/devices/{name}/arrays` and
`/api/devices/{name}/traces` (SSE). Read only; no write ever leaves the page
but through the module's put route. This makes GeecsControlRoom a CA client
as well as a Canopy client, so the slice that adds it (3) changes, in the same
PR, the root `CLAUDE.md` row and graph entry, `GeecsControlRoom/CLAUDE.md`'s
"a client of Canopy" rule and the `control-room-edges` note: a new runtime
dependency (aioca), and the host's reach to the IOCs' CA and MJPEG ports (the
frame URL is rehosted to the IOC's CA address). The CA addressing has one
home: `site.env`'s `EPICS_CA_ADDR_LIST`/`EPICS_CA_AUTO_ADDR_LIST`, which the
Control Room's unit already loads (`EnvironmentFile=@SITE_ENV@`); never
`config.ini`, never GEECS-Core's `epics_env`. Not chosen: a module route in GeecsCanopy
reading the arrays through Hare (the brief keeps images and arrays off Hare),
and a CA reader in the module beside Hare (the brief places these components
in the Control Room, and it keeps the frame and array traffic, megabytes a
second at a camera's rate, out of Canopy's api process, which serves every
screen's scalars).

**The bench.** The Mac sim stack serves the camera (`UC_VisaEBeam6`), the
motor (`SimMotor1`) and two supplies (`eztest1`, `fpstest1`). **It has no
scope simulator** (geecs-epics' `hosts/sim-undulator` lists none; the
PicoScope IOC needs its SDK and hardware): slice 4 proves the scope page on
a bench-only caproto soft IOC serving the scope contract's names on
loopback, kept out of the repo, and says so. A scope sim in geecs-epics is
the durable fix (its owner's call). Each lane runs its own Canopy api
container (the shared derived image, the lane's packages mounted, a configs
copy with a bench policy) and its own Control Room; the shared api and the
sims are left as found.

### 2026-10-31 — slice 2, the picker and routing

**Landed.** `geecs_control_room/device_pages.py`: `KINDS` (the table above,
every kind on `device.html` for now), the registry listing per kind
(`screens.registry` is now the one reader of the two device files),
`/device/<kind>` (a kit select, a table of the kind's devices with the
roster word beside each, polled from `/api/device/<kind>`, and a free-text
name; the form works without script, a 303 to the page),
`/device/<kind>/<name>` (one module read; `denied` for not on the roster,
wrong kind with a link to the right kind, not a native device; `stale`
with a 503 for a silent module, the page polling until it answers; an
unknown kind a 404 pane). `device.html`: header (name, family, kind, the
module's state as a chip, heartbeat), the switcher, the role table from
`PanelDevice.resolve` (else the family's roles), operator roles open and
expert in a `details.disc`, one table per axis, and an empty
`kind_component` block for slices 3 and 4. `device.js` / `device_rules.js`
(reusing `screen_rules.js`'s judge, show and view). The home's pickers and
its two lists open `/device/<kind>/<name>` when the family has a kind (a DB
device still opens the module's raw answer).

**Judgment calls.** An array role (`read: on_request`) is never subscribed
on Hare and reads "on request" (the kind's component reads it); a row's
units are the family's, else the record's EGU for a readback or setpoint
only (the motor record serves its EGU on every field, bits included). The
picker's select names are drawn server-side and its states filled by the
page, so the select works without script and no word is decided twice.

**Review round 1.** The role rows are now the module's device route's
roles (resolved there by `panel.resolve`), `level` and units from the
family by role name, so rows and PVs share one source; the picker keeps
the files and the module's join rule: a device the two files give
different families is listed under no kind, named in the picker's note.
A 401/403 from the module is a `denied` pane and any other 4xx an `error`
pane, neither polling; only an unreachable module or a 5xx is `stale` and
asks again, and the page reloads only on a 200 or a 404.

**OWED.** The native/panel join (one family per device, roles resolved)
is written twice, in GeecsCanopy's `native.py` and in the picker's
listing; its home is GEECS-Schemas, a slice of its own. The sim camera reads `error` (`DetectorState` Disconnected, as
the bench notes found it); the reset is a sim-side write this lane did not
make. The "not a native device" pane has no bench device (proved in tests).

### 2026-10-31 — slice 3, the camera page

**Landed.** `images.py`: `/api/devices/{name}/frame.jpg` (the JPEG at the
URL role `image` names, read over CA as a char string and rehosted to the
IOC's CA address; one fetch in flight per device, a frame shared within
0.1 s; `X-Frame-Tag` the bytes' checksum, so a frozen frame is never "new")
and `/api/devices/{name}/arrays` (every `on_request` role of type `array` by name, at most
every 0.5 s, shared; the raw-frames arc narrowed it, so a scope's on-request scalars `arr_channels`/`arr_samples` are the scope route's to read with its traces, as the frame route reads `raw_width`/`raw_height`), both read only, the module's device route cached 10 s;
404 when the module names no such role (never a derived PV), 503 with words
when a source is silent. aioca is a runtime dependency; CA addressing is
`EPICS_CA_*` alone. `device_camera.html` + `camera.js` (viewer, overlay,
readings), `camera_controls.js` (acquisition, camera ROI behind a dialog,
crosshairs and analysis ROI by value, lineouts and histogram, polled only
while open), `camera_rules.js` (pure: colour maps, levels, the view, the
ROI's pixels, formatting, which control is enabled). `device.js` exposes
`GeecsDevicePage` (the view, each role's entry, frame, meta and word) to a
kind's component; the role table stays, folded.

**Judgment calls.** Arrays are polled JSON, not SSE (a plot is a read on
request; SSE stays the scope's, slice 4). A run-locked role is not disabled
on the page: the module refuses it while a run holds the camera and its
outcome says so (the page would otherwise decide "held" from the lock's
enum label, the module's rule). Units are the family's (exposure in s, not
GeecsPanels' ms). Not built, cheap to add: background grab/subtract and
rotate/flip (their roles show in the table), crosshair colour, shape and
size edits. Not built by slice 1's rule: saturation chip, history plots,
to-centroid, run-lock override.

**OWED.** A live frame: the sim camera is in `error` (Disconnected, frames
frozen); the page was proven on its frozen frame, and the reset is the
maintainer's (a sim-side write) or the IOC host's camera.

### 2026-11-05 — slice 4 becomes the trace page

**Decision (maintainer and coordinator).** Slice 4 is built as a **trace
page**, not a scope page: one page for every 1-D array over a calibrated
axis, bound by data shape, so the PicoScope (family `scope_contract`) and
the spectrometer (an areaDetector `ADSpecfirm` driver, its family file to
come) render through one template. What differs per hardware (its controls,
its readings, its array's shape and counter) is the family file's, never
page code. Polled binary at 2 Hz, held on the y array's `fresh_by` counter
as `frame.raw` is held on `raw_counter`, not SSE. The camera rail's move into a `panel` block is a
later tidy, not this slice.

**Three schema additions** (GEECS-Schemas `epics/family.py`, PR 4a):

- `Role.axis_of`: an array role names the array it is the x axis of (both
  arrays, one axis per array, an axis no array's trace, validated on the
  family file). The scope's `time_axis` is `axis_of: data`. Beside it,
  `Role.channels_by`: an array of several channels names the readback that
  counts them (`[channels, samples]`, channels fastest); the scope's `data`
  is `channels_by: arr_channels`, `fresh_by: count` (the hold counter).
- `FamilyFile.panel`: `rail` (titled groups of roles, in order) and
  `readings` (the readings strip). Every named role must exist; a rail role
  sits in one group; a reading is a monitored role (it carries the module's
  word). No rail role type is constrained: the page draws each by its type;
  `level` stays the role table's. The scope's rail (every writable role):
  Acquisition, Trigger, Channel A..D; its readings: state, captures, per
  channel total/mean/sigma/over range, last error.

**Then.** PR 4b, the page in GeecsControlRoom (`/device/trace/<name>`,
`/api/devices/{name}/trace`); PR 4c, the spectrometer's family file, waits
on its IOC image.

### 2026-11-05 — slice 4b, the trace page

**Landed.** `/device/trace/<name>` (the kind word `scope` renamed `trace`:
one family kind per word), `device_trace.html` (rail and readings from the
family's `panel`, each rail role drawn by its type), `trace.js` /
`trace_rules.js` (one series per channel, SI ticks, cursor, linear/log, the
viewer's drag window and average), `device_writes.js` (the write path, now
the camera's and the trace page's), and
`GET /api/devices/{name}/trace` (`traces.py`: float32 axis then y, gzip,
held on the y array's `fresh_by`, the scope's `ARR:ArrayCounter_RBV`
(role `arr_counter`, added in review: the driver's `count` posts before
`ARR` publishes); `GeecsControlRoom/CLAUDE.md` § "The trace"). The bench scope `picotest1` is registered in the configs repo's
`panel_devices.yaml` with its bench prefix.

**Proof.** On a loopback caproto soft IOC serving the scope contract
(`GeecsControlRoom/scripts/scope_soft_ioc.py`; geecs-epics has no scope
simulator), through the real trace route over CA, a stub module and a fake
Hare fed from the same IOC: screenshots light and dark in
`docs/web_services/control_room.md`. **OWED:** live proof on `picotest1`
from the services box after the merge and deploy (Canopy's image rebuilt
for the family change; the Control Room restarted).

### 2026-11-05 — slice 4c, the spectrometer's family file

**Landed.** `spectrometer_contract.json` (GEECS-Schemas), the family of
geecs-epics' `docs/spectrometer-contract.md` (the Thorlabs CCT on
`ADSpecfirm` first): 57 roles after review, the spectrum (`ARR:ArrayData`, `[pixels]`,
one channel, held on `arr_counter`) over `wavelength` (`Wavelength_RBV`,
`axis_of: spectrum`); the rail Acquisition (acquire, image mode, spectra
per acquire, integration time, averages, period, light/raw), Trigger
(mode, slope) and Background (`PROC:SaveBackground`, `PROC:EnableBackground`:
the chain's dark path, never the instrument's dark command); the readings
state, count, peak count and its share of full scale, saturated, background
held, the processed spectrum's integral, peak, centroid and width (STAT1
over PROC, in index units, fresh by STAT1's own counter), temperature, last
error. The plot is the raw spectrum (`ARR`, before PROC): saturation and
exposure are judged on it; the statistics are of the processed one, the
background-subtracted spectrum when subtraction is on. Nothing autosaved
(the IOC writes nothing at boot). `Saturated_RBV` is read as the record's
own word, not a family fault bit; its alarm severity is the IOC's (`OSV`
MAJOR on the CCT), and the module turns MAJOR into the device's `error`
state, so a saturated spectrum reads as the device in error, the CCT's
full-scale first frame after an idle period included (review finding; the
severity is a geecs-epics ruling, not this repo's). The two `array_detector` families share
their vocabulary where the thing is the same (pinned). The bench unit
`ccttest1` is registered in the configs repo's `panel_devices.yaml` with
its bench prefix. The page needed no change: the spectrometer renders
through `device_trace.html` as planned.

**Proof (2026-11-05, evening).** Live on `ccttest1` (the Thorlabs CCT on
ioc-spectrometer 2026.11.1b2, started by the maintainer on the camera
server) from the services box after the merge (#1291) and the deploy
(Canopy rebuilt, the Control Room restarted): the roster lists it, the
device route resolves all 57 roles, the heartbeat and every scalar read
`live`, and `/api/devices/ccttest1/trace` serves one channel of 2048
samples over 183.9 to 1021.1 nm (a dark input: 30 to 280 counts), drawn on
the page over the wavelength axis; the spectrum-bound roles read `stale`
while `Acquire` is Done, the counters still, which is the right word. Two
facility values stood in the way and were fixed in their homes on the box,
by the maintainer's hand: the configs clone on the data share is written
only as the service account (the CIFS mount's uid), and the box's
`EPICS_CA_ADDR_LIST` in `site.env` did not carry the bench IOCs' ports on
the camera server, so every role of both bench devices read `unreachable`
until it did (geecs-epics' IOC host runbook has the rule). Hare's
WebSocket was proven from the box, not from a browser over the VPN.

**OWED.** The maintainer's look at the page in their own browser (Hare live
there); the same proof on `picotest1`, which stays `error` until the
PicoScope is plugged back in (its IOC reports no unit on USB); an
operator-policy grant for `spectrometer_contract` (and
`scope_contract`) if writes from the page are wanted, the maintainer's; the
`Saturated_RBV` severity ruling and the wavelength axis's last value
(1021.1 nm against the fit's 1006.7 at pixel 2047), both the spectrometer
port's in geecs-epics.

### 2026-11-05 — slice 4d, the maintainer's findings from the live spectrometer

The maintainer drove the deployed spectrometer page (`ccttest1`) and ruled a
centre-wide plotting rule: **line plots are Plotly**, through one layout
template the theme derives from its tokens and vendors beside its
stylesheets; images stay each surface's own (the camera page's canvas viewer
stays; the portal is not bound on images). Four findings, in two PRs:

**PR 1, the rule and the trace plot on Plotly** (GeecsWebTheme +
GeecsControlRoom; the portal's bundle moved). The rule is
`GeecsWebTheme/CLAUDE.md` § "Line plots are Plotly"; `plot.js` is the
template (tokens read at run time, trace *i* in `--trace-(i+1)` through the
colorway, the modebar, the repaint on `geecs:theme`); the one vendored
bundle is the portal's `plotly-cartesian-3.1.1.min.js`, moved into the
theme (the portal now loads it from its `/theme` mount; its empty `/static`
mount is gone). The trace page draws with it: one trace per channel,
decimated to the plot's width, the window as Plotly's x range (the drag and
"Full window" are gone), hover for the readout, `uirevision` across
captures. The unit bug (a wavelength axis read "knm") is fixed: the unit in
the axis title, SI prefixes for base SI units only. The bench gained
`scripts/spectrometer_soft_ioc.py`. Found while proving it: Plotly 3.1.1
hangs in `Plotly.react` when the config names `modeBarButtons` (its config
comparison never returns on the second react), so the template removes the
selection tools by `modeBarButtonsToRemove` instead, pinned.

**Next slice's first item: the camera page's lineouts and histogram onto
`plot.js`** (`camera_controls.js`'s `lines()` canvas drawing,
`camera_rules.js`'s plot rules). Left out of PR 1 to keep it reviewable.
When it lands, `decimate` (still in `trace_rules.js`) moves to `plot.js` if
the camera needs it, so the theme holds every shared plot rule;
`axisFormat` already moved there in PR 1's review.

PR 1 merged as #1293 and is deployed on the services box (the live
spectrometer page loads `plot.js` and the bundle, its axis in nm).

**PR 2: the processed spectrum and Start/Stop** (GEECS-Schemas +
GeecsControlRoom). The plot showed the raw `ARR` only, so Save/Subtract
background changed nothing on it, and `acquire` drew as a select of the
driver's words. Now `Role.axis_of` takes a list (a string is read as a
list of one), so `wavelength` draws `spectrum` (`ARR`) and a new
`processed` (`PROC:ARR:ArrayData`, held on its own `proc_counter`); the
route takes `?y=` (default: the first; an array the axis does not draw is
a 404 with words); the page's Show toggle picks, following the panel's new
`show_while` (`processed` while `background_enable` reads 1) until the
viewer picks. `Panel.buttons` draws a two-state rail role as a pair writing
0 and 1: the scope's and the spectrometer's `acquire` as Stop / Start, the
current state pressed. `show_while` is this PR's own addition to the
brief: the default needed a home that names the roles, and the family's
panel is where the page's layout already lives (no role names in the
page's code). On the hardware the processed path worked
(`PROC:ValidBackground_RBV` 1, `PROC:EnableBackground_RBV` 1, STAT1 mean
about 0 after a save), so the default is what the operator expects.

What PR 2's review changed: the page re-checks the array after every await
of a request, so a switch mid-body never draws the last array's capture
under the new label (or keeps its ETag); "one read in flight per device"
now says "per device and array"; `pressed`, a control rule, moved to
`camera_rules.js` beside `control`. Left for the camera slice: the camera
page draws its own Start | Stop pair (no pressed state, the opposite order
to `Panel.buttons`' 0-then-1); when the camera page moves onto the panel's
hints it takes `buttons` and `pressed` and the two pages agree. The
maintainer's GPT round found one more: after a switch the last array's
plot stayed drawn (under the new toggle and chip) until the new array's
first capture, indefinitely if its reads kept failing; the plot is now
hidden until then, the window kept, pinned by a harness running the real
`trace.js`.

### 2026-11-06 — slice: the device directory

**Planned** (the maintainer, after driving the spectrometer page: the home
offered no path to the device pages but a Cameras picker he added to
`home.yaml` himself). A registry-derived directory on the home, uncurated:
every device of `native_devices.yaml` + `panel_devices.yaml`, grouped by
kind in `KINDS` order, each row the name (linking its page), the family,
the module's word and its IOC's heartbeat; a client-side filter; the kind
headings link the pickers; a device whose family no page draws listed last
("no page yet", unlinked); and a way back from every device page.

**Landed.** `device_pages.catalogue` is the one registry join (every
device with its kind word or `None`); `registered` (the pickers) filters
it. `home.directory` groups it for `home.html`, server-drawn (the rows and
links work without script, as the picker's table does); the words ride the
home's existing 2 s `/api/home/devices` poll (`words`: the roster's state,
the heartbeat from the device detail that poll already reads for the two
lists), so the directory costs no module read of its own. Polling the four
`/api/device/<kind>` rosters instead was rejected: they carry no heartbeat,
and would read the roster four more times per tick for what the lists'
read already has. `directory_rules.js` holds the filter (every term a
substring of the name or the family), a row's words and the chip; a silent
module leaves the rows with `unknown`. The trail is a `crumbs` block in
`base.html` after the brand (the home's link and the trail's first step),
filled by `device.html` (kind's picker › the device), so the camera and
trace templates inherit it untouched.

**What review changed** (one adversarial round, four low findings, all
fixed): a device off the roster read heartbeat "no IOC" beside the IOC its
panel entry names, now `unknown`; a registry whose devices all conflict
between the two files read "cannot be read", now an `empty` pane naming the
conflicts (`Catalogue.unreadable` keeps `error` for files that do not
parse), and the empty panes no longer carry a "0/0 live" chip; a
`[hidden]` rule the kit already holds (`!important`) was deleted; the roster
state word is one rule for the pickers and the directory
(`device_pages.state_word`). Kept: the directory's rows are fixed at page
load (the pickers re-read theirs each poll); a registry change shows on
reload, which the operator page says.

**Left as it is.** `home.yaml`'s `device_pages` pickers now overlap the
directory (every device a picker lists is a directory row); they stay the
curated layer (pinned families), and whether to drop them from the
experiment's `home.yaml` is the maintainer's (the configs repo is his). The
directory lists the registry, not the roster: a DB (LabVIEW) device has no
device page and stays out; a screen draws it. Grouping by area would need
an area per device, which the registry does not carry; by kind is what it
knows.

### 2026-11-06 — the detector's health beside its Start button

**Landed.** From the spectrometer port's hand-off (geecs-epics, the ccttest1
session): a fault on the instrument is the driver's decision, not a crash.
It stops, sets `Acquire` to Done and says why in `StatusMessage_RBV`;
`DetectorState_RBV` reads Error; a new Start recovers from Error. A line
that stops answering reads Disconnected on `Connected_RBV` (MAJOR) and the
driver retries on its own. So the page must show, beside Start, the state
word, the line and the driver's reason. Both `array_detector` panels'
readings strips now open with `state`, `connected`, `status`, then the
counter (the roles existed; only the layout changes). Deploy: Canopy's
image carries the family files, so a rebuild and the Control Room's
restart.

**Noted, not built.** Every IOC serves `<IOC>:SysReset` (a put of any
value exits the IOC; a widget writing 0 then 1 would restart it twice); a
production container restarts on its own, a bench container stops and is
started by name on its host, and the put is subject to the IOC's access
file. A restart control on the device drill-down (operator_surfaces § 2)
is therefore a production-only affordance with a confirm, never offered
for a bench instance; the contract page is geecs-epics
`docs/architecture.md`, "Operating an IOC".

### 2026-11-06 — slice 5, the settables page

**Ruled** (the maintainer): one type-driven settables page for every kind
without a data component, serving the native families (motor, supply) and
the GEECS DB devices the module serves through the gateway; GeecsPanels'
deletion waits until this page is proven.

**PR 1 (#1298, merged).** GEECS-Schemas `Panel.tweak` (a setpoint, its step
setpoint and two step commands, one rail group, compared by place, not
heading: the maintainer's GPT finding), `PanelGroup.layout: grid` (short
controls only: bits, enums, two-state whole numbers, the button pair's
rule), and panels for `motor_record`, `easydriver`, `fastps`. Per-axis
families need no field: the axes are instance data the module serves.

**PR 2, the page.** `device.html` draws the rail and readings from a panel
by default (`_panel.html`, the macros the trace page now shares;
`device_rail.js`, the one rail script); a per-axis family is one pane per
axis behind an axis selector, each write carrying `axis`. A family without
a panel, and every DB device, is laid out automatically
(`device_pages.auto_panel`: settings in role order, switches in a grid of
their own from four, every monitored read-only role a reading). Kind word
`settables` for the DB devices: the picker, the roster route and the
home's directory list them from the module's roster. Proven on the
geecs-epics sim stack (simmotor1, fpstest1, and trigger-box-sim standing in
for a DB device) through GeecsCanopy's own module in process over an aioca
Hare: a tweak on axis m2 accepted and the motor moved; the DB device's
amplitude accepted; the FAST-PS current refused by the IOC's access
security, the refusal shown (the sims restored after). The automatic layout
is Python, not a node rule: one drawing path, server-side, from a panel
authored or automatic.

**What review changed** (two adversarial rounds and the maintainer's GPT
round): a number or text field on the rail is a draft written on Enter (Esc
drops it; kept on blur, as a screen keeps it), never on `change`, so an
arrow key on a motor's target sends no move (the trace page gets it too),
and a tweak's − / + sends a draft step first (round 2's finding: a typed
step was dropped on the button's mousedown and the motor stepped by the old
one); a gateway device is classified by its `source` before any family
lookup (the GPT finding: a DB type spelled `motor_record` was sent to the
motor page); the module's device route now serves each role's `text` and
`units` (GeecsCanopy `freshness.view`), so a DB device's text setting is a
text field from the first draw and its units show (the page's runtime type
flip is deleted); the home's Not live and In alarm rows link a DB device to
its settables page; the alarm line is one rule (`camera_rules.alarms`,
filtered by axis) and the camera's write value is `device_rules.writeValue`.
Left for the merge slice: the rules still in `camera_rules.js`, the camera's
own rail, and the rail-and-pane grid's home (`GeecsControlRoom/CLAUDE.md`
names each piece on both sides).

**OWED.** The deployed pages on the services box (the coordinator's): a
real PLC (many switches, the grid) and a filter wheel from the DB; Canopy
rebuilt for the family files and the route's new fields, the Control Room
restarted.

### 2026-11-06 — slice 6, GeecsPanels deleted

**Ruled** (the maintainer, after looking at the deployed camera, trace and
settables pages): the prototype is deleted under the kill rule. One PR to
`master`, off `master` (#1300).

**Deleted.** `GeecsPanels/` whole (0.1.0): the FastAPI app on port 8500,
its own CA client (aioca), the per-kind pages (camera, scope, motor,
supply), its bench file and `allow` lists, its tests. It ran from a laptop
over the VPN only; nothing on the services box changes. The final state is
kept at the tag `geecs-panels-v0.1.0-final`, cut by the maintainer on
master's last commit before the deletion (the PR body has the lines).

**Rewritten** (every live reference): the root `CLAUDE.md` (the map row,
the "GEECS side" sentence, the dependency-graph entry and two "never
imports" lines, the CHANGELOG list, a paragraph in "Deleted legacy
packages"); `README.md` (the table row, the mermaid node and edge);
`CONTRIBUTING.md` § Tests; `.importlinter` (the package in `root_packages`
and the layers, every forbidden list, the `panels-edges` contract, the
Control Room contract's name); `scripts/check.sh` and `scripts/ci_select.py`
(the leg); `scripts/doc_audit.py` (`GeecsPanels` joins `STALE_TERMS`);
GeecsWebTheme's literal-colour walk and its surface table; the docstrings
and notes that named it as a consumer (`epics_env`, `pv_naming`,
`panel_devices`, `family`, GEECS-Core's `DESIGN.md`, GEECS-Schemas'
`CLAUDE.md`, GeecsBluesky's `NATIVE_DEVICES.md`); GeecsControlRoom's
`CLAUDE.md` and `camera.js` header; this brief's live instructions and
`operator_surfaces.md` §6 (now pointing at the Control Room's pages).
There was no unit template under `deploy/`, no mkdocs page and no skill
probe of port 8500 to remove.

**Stays.** The configs repo's `panel_devices.yaml` and GEECS-Schemas'
`PanelDevices`: no longer GeecsPanels' file, read by GeecsCanopy,
GeecsControlRoom and the archiver, untouched. History stays history: the CHANGELOGs, this brief's
dated entries, `CONTRIBUTING.md`'s Planning audits, the "Panels" paragraph
of `NATIVE_DEVICES.md` (marked as the record).

### 2026-11-06 — vocabulary merge, PR 1: the setpoint control

**Ruled** (the maintainer, 2026-11-06): the rail becomes the one control set,
the screen renderer adopts it (PR 2), flipper and annotation stay
screen-only; four setpoint behaviours from his operators' request to set a
value and see it read back in the same interface (a field that tracks the
setpoint, both values beside it, `differs` after a settle with a tolerance,
the draft kept until accepted).

**Built** (the rail only: the trace and settables pages; the camera's
hand-coded rail moves in PR 3). The setpoint is the role's write record, the
module's `writePv`, read over Hare: no family declares a setpoint record of
its own, because the route already names it for every device, native and
DB, and declaring one would change what a role means to the archiver,
presets and Squirrel. A DB device's field tracks its value, not its `:SP`
(the gateway's last forwarded put, 0 until the first). GEECS-Schemas'
`Role` gains `tolerance` and `tolerance_by`; `motor_record`'s `val` is
judged within `rdbd`. The rules and their reasons are
`GeecsControlRoom/CLAUDE.md` § "The setpoint control".

**Proved**: the rules in the module contract tests (tracking sources over
the module's own entries, the module's answers to a write accepted, refused
and denied, a settled mismatch, a motor by done-moving, a DB device's
tolerance reaching the field); the page's DOM half (tracking, the draft,
arming) on the spectrometer and scope soft IOCs through
GeecsCanopy's real module (the bench's driver clamps integration time to
10 s: a write of 20 reads `set 20 · now 10 · differs`). The bench found one
defect, fixed: the field is disabled while its write runs and so lost focus,
leaving Esc on a kept draft with nowhere to go.

**Review** (round 1, five findings) changed: the focus guard is back (a
DB device's field tracks its measured value, which would land under the
caret and be written); a DB device's `differs` is judged within its DB
`tolerance` (GeecsCanopy's gateway source puts it on the role, the module's
route serves `tolerance`; the gateway's PREC is 3 for every numeric, so the
fallback was 0.001); a kept draft no longer takes focus back from the
operator's name; `device_writes.js` stops on `device_rules.accepted`, the
rule the tests pin. Waived: Squirrel's registry tolerance (presets' own
concern), a disarm on one agreeing sample (the tolerance is the noise
control). The maintainer's GPT round (one P2): a failed device poll
dropped the route's entry, which read as "nothing to judge" and disarmed the
field, so the mark never came back with the route; the module's silence now
keeps the arming (`settleMark`, `disarms`), pinned by running the real rail
script over a fake page (the case failed before the fix).

### 2026-11-06 — vocabulary merge, PR 2: the screen renderer draws the rail's controls

**Built.** A screen's setpoint, toggle, flipper, select, button and readout
are `_panel.html`'s `control`, filled and written by `device_rail.js` through
`device_writes.js` (one per device; the widget's own outcome line and
`confirm` note); a toggle is the rail's two-state pair. `screen.js` keeps the
polling, the words, a flipper's lamps and the action runs; its own field,
toggle and write path, and `screen_rules.writable` / `hazard` / `differs`,
are deleted. One device reading for both kinds of page:
`device_source.js` (the device page's `GeecsDevicePage` and each device a
screen binds). The rail takes a page and writes per element (`for(el)`).

**Found on the way.** The configs repo's Control Center uses `select` and a
toggle's `values` (#1241), merged into `feat/control-room-push4` and never
promoted: master could not draw it. This PR carries both (the schema half
as #1241 had it, the page half on the rail), so master draws the screen
unchanged; an integration test validates and draws every screen in the
configs checkout. The `group` widget (a device's rail group on a screen)
moved to PR 2b to keep this one inside the size rule.

**Review** (round 1: two medium, four low; all fixed, confirmed in round 2)
changed: the screen's write path is pinned by running the real rail and
write scripts over a fake screen (a widget's note asks first, the pair, the
narrowed select, a bare PV disabled; it failed with the note dropped); one
redraw per frame on a screen (a Hare diff of N PVs had cost N full passes)
and one subscription per role on the device page (`device_source`'s
`read`); the field at the widget's precision; one spec builder
(`screens.role_spec`); an unsent choice goes back to the reading; a note-only
ask is titled "Write?".
