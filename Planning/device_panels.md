# Device panels — a browser GUI per device kind (first: the AD camera)

Status: **design, for review.** Nothing here is built. Delete this file in
the PR that finishes the camera panel (`Planning/README.md`).

## Why

The LabVIEW camera GUI (`GEECS-General-GUIs/Device GUIs/GUI_Cameras`)
mostly shows what the PointGreyCamera device computes behind the scenes:
centroid, lineouts, background subtraction, saved crosshairs. A native
areaDetector camera computes the same things in its IOC's plugin chain
(geecs-epics#6, verified on `undulator:adtest_cam1:`), as standard PVs.
What is missing is the window an operator looks through. This note designs
it, and the home it shares with the device GUIs that will follow (motors,
delay generators, …).

Not in scope: legacy LabVIEW cameras (they keep the LabVIEW GUI until they
migrate; their control PVs differ), migrating any real camera, live
analysis recipes (`LIVE:` PVs — the panel shows them once they exist,
nothing here depends on them), Stats as scan scalars.

## Decisions

### 1. A service of its own, panels keyed by device *kind*

`GeecsPanels` — a FastAPI service on the worker host, port **8500**, its
own unit like the logbook. Not a scanner tab: the scanner would become the
old master-control panel once every device family adds one, and a scanner
restart must not take the camera views down.

One panel per device **kind**, not per device. The camera panel serves
every `aravis_camera` in the registry from its prefix; a new camera is a
registry entry, never code. A new kind is one new panel — but a kind
whose writes *move* things (motors, stages) does not get direct CA puts:
the scanner already moves settables through the queue (`POST /api/move`),
serialized with scans, and such a panel submits through that API. Direct
puts are for devices whose run ownership the device itself shows, as the
camera does (§6).

**The stock alternative, weighed.** EPICS already has a web path: the
Phoebus Display Builder web runtime (dbwr) serving `.bob` screens, over
pvws (a PV ↔ websocket bridge). ADCore ships screens for every plugin, so
it would cover *every* PV of every camera with no code, and it is the
"less custom = more joinable" choice. Against it for the operator view:
the point of this panel is the customisations the LabVIEW GUI earned —
drag-to-place crosshairs, lineouts through a clicked point, a scan guard,
a registry landing page, the GEECS kit look — and dbwr's widget set is a
subset of Phoebus's (image-widget interaction not verified). Both can
coexist: the panel is the operator's window onto the everyday set; the
stock `.bob` screens (desktop Phoebus, or dbwr later) remain the expert
view of everything else, at no cost to us. **Decision for the maintainer**
(§4): our own small PV bridge vs pvws under the panel.

- `/` — the experiment's native devices from the registry, grouped by kind,
  each linking to its panel.
- `/camera/{device}` — the camera panel (a route: people link to it,
  reload it, keep it open all day). The scanner, the portal and logbook
  entries can link here.

### 2. Where everything comes from

| Need | Source |
|---|---|
| Which devices, which kind | `NativeDevices` (`<configs>/experiments/<Experiment>/native_devices.yaml`, GEECS-Schemas) |
| PV prefix | derived, `<experiment>:<device>:` (GEECS-Core `pv_naming`, the same rule the worker uses) |
| Plugin PV names | **the chain contract** (below) |
| Stream URL | the IOC publishes it: `MJPG1:JPG_URL_RBV` |
| Experiment, configs path | `config.ini` (the facility-value home) |

No new registry fields. Hosts and ports stay out of the registry; the IOC
says where its own stream is.

**The chain contract.** The panel addresses plugins by suffix, so these
names become a contract every camera IOC keeps (they are what
geecs-epics#6 deployed): `cam1:` (driver), `TRANS1:`, `PROC1:`, `ROI1:`,
`STAT:`, `OVER1:` (overlays 1–8), `MJPG1:`, `HDF1:`, plus `STAT0:` (raw
stats, build step 0). Step 3 of the arc (the
chain as an ibek template) makes them impossible to get wrong per camera;
until then the panel shows a camera whose chain PVs do not connect as
`denied` with the missing suffix named, never a half-working page.

### 3. Image: one frame at a time, through the service

The browser asks the service for a frame, draws it on a `<canvas>`, then
asks for the next — never more than one frame in flight. The service
proxies `GET MJPG1:JPG_URL_RBV` from the IOC host.

- **Latency stays at one frame** whatever the client's speed (a pushed
  MJPEG stream built seconds of backlog in a browser; measured put → frame
  is 38–46 ms at the IOC).
- **Same origin**, so the canvas can read pixels: colour map, contrast and
  zoom are client-side, per viewer, and touch nothing in the IOC.
- **Rate**: the client asks no faster than a cap (default 10 fps) and the
  service holds at most one upstream request per camera, shared between
  viewers (the latest frame is reused by everyone who asks within its
  interval). N viewers cost the camera host one stream.

**A frozen picture must look frozen.** The `.jpg` reply carries no frame
identity (`Last-Modified` is the request time), so a stopped camera keeps
serving its last frame as if live. The image's age comes from a PV the
bridge already monitors, `MJPG1:ArrayCounter_RBV` (and `cam1:` for the
camera itself): no new count for a threshold → the image pane goes
`stale`, saying how long.

**Before step 1:** check the snapshot endpoint with no stream clients
(`MJPG1:CLIENTS_RBV` 0, `ALWAYS_ON` Off) — whether ffmpegServer still
encodes, or blocks a snapshot until the next frame (up to 1 s on a 1 Hz
triggered camera). Pin `MJPG1:ALWAYS_ON 1` at boot if needed.

Limitation, accepted for v1: the JPEG is 8-bit, after the IOC's own
mapping. Contrast stretching works within those 256 levels — enough to see
a 10-count beam, not to measure it. Measurement never comes from the image
(next section).

### 4. Numbers come from PVs, never from the picture

A PV bridge in the service: it monitors the camera's PVs over CA (aioca,
the client ophyd-async already uses) and fans the values to each open panel
over **SSE** (the scanner's pattern). Arrays (profiles, histogram) are
throttled to the frame cap. Writes come back as `POST`s that the service
turns into CA puts, each one allow-listed: the panel can put only the PVs
its controls name, never an arbitrary PV.

Every live value carries its age and goes `stale` (the kit's `.live`)
when its monitor stops updating — a frozen centroid must look frozen.

The bridge is ~150 lines of Python on the stack we already run. pvws is
the stock equivalent (a Java/Tomcat service, writes switchable per
deployment, no per-PV allow-list). **Maintainer's call**; the note
assumes our own bridge, and the panel's browser code is the same either
way except for the transport.

**Coordinates.** `STAT` centroid, profiles and `CursorX/Y` are in `ROI1`
coordinates; the image and overlays are in `PROC1` (post-transform)
coordinates. One module in the service converts both ways —
`full = ROI1:MinX_RBV + roi·ROI1:BinX_RBV`, with `ReverseX` — used for
"set crosshair to centroid" (ROI → full) and "lineout through a clicked
point" (full → ROI before the `CursorX` put). A transform other than None
moves the centroid, the overlays and the autosaved crosshairs off
raw-frame coordinates; the panel says so whenever `TRANS1:Type` is set.

### 5. What the camera panel shows and does (v1)

The LabVIEW GUI's everyday set, in areaDetector terms:

| Feature | PVs | Notes |
|---|---|---|
| Image, zoom, colour map, contrast, autoscale | — | client-side |
| Two crosshairs, labelled, drag to move, "set to centroid" | `OVER1:1:`, `OVER1:2:` (`CenterX/Y`, `SizeX/Y`, `Name`, `Use`) | burned into the stream by the IOC, so every viewer and tool sees them; autosaved. "Set to centroid" is a one-shot put of the centroid converted to full-frame coordinates (§4) |
| Centroid, max, mean, sigma | `STAT:` | live chips |
| X/Y lineouts (average and through a clicked point) | `STAT:ProfileAverageX/Y_RBV`, `STAT:ProfileCursorX/Y_RBV`, `STAT:CursorX/Y` | full precision, from the IOC |
| Histogram | `STAT:Histogram_RBV` | |
| Mean / centroid history | the bridge's own `STAT` monitors | kept in the service, per camera; the IOC's `STAT:TS` buffers are idle (`TSAcquire` Done) and IOC-global — not used |
| Analysis ROI, drawn as a box | `ROI1:MinX/MinY/SizeX/SizeY` | changes stats only, never scan data |
| Background: grab, on/off | `PROC1:SaveBackground`, `PROC1:EnableBackground` | grab with the beam blocked; lost on IOC restart (runbook) |
| Flip / rotate 90° | `TRANS1:Type` | display + stats only |
| Saturation warning | `STAT0:MaxValue_RBV` ≥ the pixel format's full scale | chip + optional sound. `STAT0` is a new raw-chain stats plugin (fed from the driver, statistics only): `STAT` sees background-subtracted, ROI-cropped data and would miss saturation |
| Exposure, gain, frame rate, trigger mode, acquire | `cam1:` | **guarded** (§6) |
| Camera (driver) ROI | `cam1:MinX/MinY/SizeX/SizeY` | **guarded**, and a `<dialog>`: it changes what scans save |

Dropped against the LabVIEW GUI: perspective correction (ruled out),
arbitrary-angle rotation, spatial low-pass, ellipse/Gaussian-fit overlays,
free-run "Save ON" (a scan or a Bluesky `count` saves frames), loading
saved images (the portal), the multi-camera grid (revisit after v1).

### 6. The scan guard

A panel must not change what a running scan is saving. Two kinds of
control:

- **Display and analysis** (crosshairs, analysis ROI, background,
  transform, colour, contrast) — never guarded: none of them reaches `HDF1`.
- **Acquisition** (`cam1:` exposure, gain, rate, trigger, acquire, driver
  ROI) — refused while the camera belongs to a run.

"Belongs to a run" is read **from the camera itself**, not from the
scanner: `HDF1:Capture_RBV` is `Capturing`. Capture alone covers the run
in the right order: ophyd-async starts capture after NativeCamera's
prepare, and unstage stops acquisition before it stops the writer.
Trigger mode is *not* a signal: unstage restores the pre-run mode, and a
camera may run hardware-triggered between scans (the test IOC boots in
free-run, a production camera need not), so it would lock the panel
permanently. The panel shows the acquisition group as `denied`
("in a scan") rather than hiding it. No scanner API change, no dependency
on the scanner being up.

**A stuck capture** (a killed worker, a halted run) would hold the lock
forever. The denied group offers an explicit override behind a `<dialog>`
that names the condition and asks for the operator's name (the web
surfaces carry no user identity); the service logs the name and the time.
The override **only unlocks the panel's own controls** — it never writes
to `HDF1`, so misjudging a live run as stuck cannot end its stack file.

This guard protects panel users only. A raw `caput` (or Phoebus, or an
agent) is not stopped by it. The EPICS answer — access security in the
IOC, a lock PV the run sets — is the right long-term fix for every client
and is out of scope here.

### 7. Package shape and dependencies

```
GeecsPanels → GeecsWebTheme (web: the kit, prefix middleware, templates)
            → GEECS-Schemas (NativeDevices)
            → GEECS-Core (pv_naming)
            + aioca, httpx (frame proxy)
```

Never imports GeecsBluesky, the scanner, or the portal. On the kit from
the first commit (`_SURFACES` in the theme's colour test). Plots
(profiles, histogram, history) are small canvas/SVG drawings styled through
tokens; if a plot component proves reusable it goes to the kit in its own
PR against `kit.html`, not inline here.

Facility values: the port, the bind interface, the experiment and the
configs path come from `site.env` / `config.ini` — none committed.

**CA addressing.** Clients get the CA address list from `config.ini
[epics] ca_addr_list`, applied to the environment by
`geecs_bluesky.epics_env` at import — which GeecsPanels must not import.
Without it the panel's CA client would search only the default list and
find no IOC. So **the env bootstrap moves to GEECS-Core** (beside
`geecs_core.configs_repo`, which it already reads) as part of build step
1, and GeecsBluesky re-exports it. Its one GeecsBluesky dependency
(`data_paths.pva_addr_tokens`, for the `[pva]` list) moves with it or
stays behind as a Bluesky-only extension. The deployed list must name the
IOC hosts as well as the gateway; at HTU it does today (a `site.env`
override set when the test camera went live), but the committed
`deploy/site.env.example` shows only the gateway — that example gains an
IOC-hosts line in the same step. The prefix rule and the configs base are
GEECS-Core's (`pv_name`, `configs_repo.scanner_configs_base`).

## Build order

0. **Chain contract v2** (geecs-epics): add `STAT0` (raw stats); pin
   `MJPG1:ALWAYS_ON` if the no-client check needs it. Colour crosshairs
   (a `ColorConvert` before `OVER1`) only if the maintainer wants them.
1. **Skeleton**: the CA env bootstrap moved to GEECS-Core (§7); package,
   unit, landing page from the registry, camera route with the frame loop
   and the live STAT chips (read-only). The package's registration rides
   along: root `CLAUDE.md` map, dependency graph and CHANGELOG list,
   `.importlinter` contract, the theme's `_SURFACES`.
2. **Interaction**: crosshairs, analysis ROI, background, transform,
   lineouts/histogram/history, saturation.
3. **Acquisition controls** behind the guard; driver ROI behind its dialog.

Each one PR, each verified against `ADTEST_CAM1` in a browser.

## Open questions

- **>8-bit cameras.** What does ffmpegServer do with `Mono12`/`UInt16`
  frames — scale, clip, or refuse? If it clips, the display needs a
  scale step on a display-only branch (`PROC2`?) so `STAT` keeps the raw
  values. To check on the test camera before a 12-bit camera migrates.
- **Host name resolution.** `JPG_URL_RBV` names the IOC host by its host
  name; the worker must resolve it (lab DNS or `/etc/hosts`), or the
  service rewrites the host to the IOC's CA server address.
- **Crosshair colour.** Burned-in overlays are grey levels on a mono
  camera. Drawing them client-side instead would give colour, at the cost
  of tools other than the panel not seeing them. v1 burns in.
