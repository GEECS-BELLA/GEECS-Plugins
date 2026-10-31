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

On the Mac sim stack (geecs-epics `hosts/sim`: the areaDetector sim camera
and the scope simulator, isolation rule respected) the camera page connects to
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
