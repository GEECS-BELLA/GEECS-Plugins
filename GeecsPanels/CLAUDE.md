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
  camera.py     # the chain contract (PV suffixes) and the freshness rule — pure
  feed.py       # CameraFeed (monitored PVs → snapshot), FrameSource (shared JPEG)
                #   — transports injected, so tests need no CA library or IOC
  ca.py         # the real transports: aioca monitors, stream URL, httpx
  app.py        # routes: /, /camera/{device}, its frame.jpg and events (SSE)
  templates/, static/   # on the GeecsWebTheme kit; colours through tokens only
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
- **The stream URL is rehosted to the CA address** (`ca.with_ca_host`): the
  IOC names itself by hostname, which a laptop cannot resolve.
- **Read-only so far.** A write route must name its PV in an allow-list
  (design note §4), and acquisition controls wait for the run lock (§6).
- Never imports GeecsBluesky, the scanner, the portal or the logbook
  (`.importlinter` `panels-edges`).

Testing: `poetry run pytest` — offline, fake transports. The live check is
running it against the test cameras and looking at the page.
