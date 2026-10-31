# Device pages — the brief (integration branch `feat/device-pages`)

Scaffolding per `Planning/README.md`: the promotion PR to `master` deletes this
file (and `Planning/device_panels.md`, which it supersedes) after moving
anything load-bearing into the owning `CLAUDE.md` files. The coordinator runs
the slices in order, one PR each into this branch, with the `/land` ritual, and
merges a lane PR after its review is dispositioned (maintainer's rule for
integration branches). `master` is merged forward at least weekly. Promotion
to `master` is the maintainer's merge; the GeecsPanels deletion is its own PR
under the kill rule, after the maintainer has looked at the rendered pages.

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
  and the scope traces streamed over SSE. Copy, adapt to the kit, then delete
  the originals in the deletion PR; GeecsControlRoom never imports GeecsPanels.
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
  dev-only dep, nothing from GeecsPanels, GeecsBluesky, the portal); facility
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
4. **Scope page.** Channel configuration and trigger through the module; the
   moved SSE trace component over the scope contract; screenshots.
5. **Motor and supply pages** from the family files' roles (small; the
   inspector's rich view and the drill-down).
6. **GeecsPanels deletion PR** (separate, under the kill rule): the package,
   its unit template, its row in root `CLAUDE.md`, `.importlinter`,
   `scripts/check.sh`, `scripts/ci_select.py`, the theme's template walk, the
   docs page, the CHANGELOG list; `Planning/device_panels.md` goes with it.
   Opened on this branch but merged only after the maintainer's look.
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
`/api/device/{name}/frame.jpg`, `/api/device/{name}/arrays` and
`/api/device/{name}/traces` (SSE). Read only; no write ever leaves the page
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
