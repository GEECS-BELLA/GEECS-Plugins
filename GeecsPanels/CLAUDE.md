# GeecsPanels — Developer Context for Claude

Device panels: a browser GUI per native-device **kind** (first: the
areaDetector camera), keyed by the experiment's `native_devices.yaml`. The
design note is `Planning/device_panels.md`; read it before adding a panel
or a control.

**It is a prototype** (maintainer, 2026-10-22): something to look at, play
with and ask for features on, run on a laptop over the VPN. Where it lives
in the end (its own service, inside another surface, a bespoke ops panel
composed from its parts) is open. So: keep each part (image view, live
readings, later lineouts and the crosshair layer) self-contained, so it can
move without a rewrite. Don't add deployment machinery (unit, service
account) until the maintainer asks.

## Layout

```
geecs_panels/
  __main__.py   # CLI; applies geecs_core.epics_env BEFORE anything imports aioca
  registry.py   # native_devices.yaml → Device(name, kind, prefix)
  camera.py     # the chain contract (PV suffixes), the freshness rule and the
                #   write allow-list (WRITABLE, check_write) — pure
  coords.py     # image pixels <-> the analysis ROI's output pixels — pure
  feed.py       # CameraFeed (monitored PVs → snapshot), FrameSource (shared JPEG)
                #   — transports injected, so tests need no CA library or IOC
  ca.py         # the real transports: aioca monitors, stream URL, httpx
  app.py        # routes: /, /camera/{device}, its frame.jpg and events (SSE),
                #   and the writes: put (allow-listed), crosshair colour, to-centroid
  templates/, static/   # on the GeecsWebTheme kit; colours through tokens only
                #   camera.js = the viewer (frame loop, zoom/pan, colour map);
                #   overlay.js = the crosshair + ROI layer, plugged in through
                #   window.GeecsCameraView's hooks (draw, grab, hover, snapshot);
                #   processing.js = background, rotate/flip, the saturation chip.
                #   Writes go through window.GeecsCameraView.write (one copy)
```

## Rules this package keeps

- **Numbers come from PVs, never from the picture.** The image is the
  IOC's 8-bit greyscale stream for looking at; colour map and levels are
  the viewer's own, client-side, written nowhere.
- **Fresh is judged by the producing plugin** (`ArrayCounter_RBV`), not by
  the value's own monitor — NDStats posts only on change. A value whose
  `Compute*` flag is off is `off`, not fresh. See `camera.py`'s docstring.
- **One upstream per camera.** Every viewer shares one feed and one frame
  source; the page pulls one frame at a time (never more than one in
  flight) and pauses while hidden.
- **Image pixels, not JPEG pixels.** ffmpegServer scales a frame to fit
  `MJPG1:MAXW/MAXH` (a rotated 1280x1024 frame arrives as 960x1200), so the
  page stretches the JPEG to the true size (`MJPG1:ArraySize0/1_RBV`) and
  every position is in image (PROC1) pixels.
- **Saturation is judged on the raw frame** (STAT0, before background and
  ROI) against the pixel format's full scale (`camera.saturation`).
- **The stream URL is rehosted to the CA address** (`ca.with_ca_host`): the
  IOC names itself by hostname, which a laptop cannot resolve.
- **Writes are allow-listed** (`camera.WRITABLE`): crosshairs (OVER1:1/2)
  and the analysis ROI (ROI1) only — display and analysis state, never
  what a scan saves. The page names an allow-listed suffix for a plain
  put; anything that needs computing goes through its own route (the colour
  route takes r/g/b — channel PV names in the script would also trip the
  theme's colour-word guard — and to-centroid converts on the server with
  `coords`). `--read-only` refuses every write. Acquisition controls (the driver) wait for the run lock
  (design note §6).
- Never imports GeecsBluesky, the scanner, the portal or the logbook
  (`.importlinter` `panels-edges`).

Testing: `poetry run pytest` — offline, fake transports. The live check is
running it against the test cameras and looking at the page.
