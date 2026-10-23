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
                #   processing.js = background, rotate/flip, the saturation chip;
                #   plots.js = lineouts, histogram, history (one canvas drawer:
                #   custom for now, a plotting library to be weighed later);
                #   acquisition.js = the camera's own controls behind the run lock.
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
- **Arrays are read on request, never monitored.** STAT's profiles and
  histogram would arrive at the camera's full rate (~0.6 MB/s over the VPN
  at 18 Hz); `ArraysSource` reads them at most every 0.5 s, shared, and only
  while a page has the plots open (`events?plots=1`). The history (centroid,
  mean, max) is kept in the service, sampled on STAT's frame counter.
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
  `coords`). `--read-only` refuses every write.
- **The camera's own controls sit behind the run lock** (design note §6;
  geecs-epics `run_lock.db` + `access.acf`, GeecsBluesky's NativeCamera
  sets it). Driver writes (`LOCKED_PREFIXES`: exposure, period, gain,
  trigger mode, acquire, the camera ROI) are refused with a 423 while
  `RUN:Lock` is Locked **or cannot be read**; the IOC refuses them too
  (its access security), so the panel's check only says why first. The
  camera ROI goes through a `<dialog>`: it changes what scans save. A stuck
  lock is released through the override dialog — it writes `RUN:Lock` 0
  and nothing else, and logs the operator's name.
- Never imports GeecsBluesky, the scanner, the portal or the logbook
  (`.importlinter` `panels-edges`).

Testing: `poetry run pytest` — offline, fake transports. The live check is
running it against the test cameras and looking at the page.
