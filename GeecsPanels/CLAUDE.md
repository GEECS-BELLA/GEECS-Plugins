# GeecsPanels — Developer Context for Claude

Device panels: a browser GUI per native-device **kind** (the areaDetector
camera, the oscilloscope, the positioner, the power supply), keyed by the experiment's
`native_devices.yaml`. The design note is `Planning/device_panels.md`; read
it before adding a panel or a control.

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
  registry.py   # native_devices.yaml → Device(name, kind, prefix); the bench
                #   file (--bench): bench devices + per-device allow/note
  camera.py     # the chain contract (PV suffixes), the freshness rule and the
                #   write allow-list (WRITABLE, check_write) — pure
  coords.py     # image pixels <-> the analysis ROI's output pixels — pure
  feed.py       # CameraFeed (monitored PVs → snapshot), FrameSource (shared JPEG)
                #   — transports injected, so tests need no CA library or IOC
  scope.py      # geecs-epics' scope contract: its PV
                #   suffixes, the write allow-list, splitting ARR's capture and
                #   min/max trace decimation — pure, vendor-neutral
  scope_feed.py # ScopeFeed (monitored settings → snapshot), traces_payload
  scope_app.py  # /scope/{device}: page, events (SSE, ?traces=1), put
  motor.py      # the positioner: the motor record's fields, .MSTA by name, the
                #   state words, the write allow-list (a move checked against the
                #   soft limits first) — pure, every word the record's
  motor_feed.py # MotorFeed (every axis's fields → snapshot)
  motor_app.py  # /motor/{device}: page (one card per axis), events (SSE), put
  power.py      # the power supply: roles → PVs, the write allow-list (a setpoint
                #   inside DRVL/DRVH and below HIHI first), the state words — pure
  supply_feed.py # SupplyFeed (every role's PV and fields, the bits → snapshot)
  supply_app.py # /supply/{device}: page, events (SSE), put (by action)
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
                #   Writes go through window.GeecsCameraView.write (one copy);
                #   scope.js and motor.js are each their own page
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
- **The scope panel speaks the scope contract only**, never a vendor's
  records: every enum's choices are read from its PV (`ScopeFeed.enums`),
  so one page fits any scope. It writes every setting (`scope.WRITABLE`):
  driving the scope is the panel's point; capture products and the chain's
  outputs are never writable. Its traces are read like the camera's arrays (on request,
  shared, ``ArraysSource``) and min/max-decimated to the screen's width,
  so a one-sample spike survives. A bench scope is not in the native-device
  registry yet: `--scope NAME=PREFIX` gives it a panel.
- **The positioner panel speaks the motor record only** (`motor.py`): one
  card per axis of a registry `epics_motor` device, every word from the
  record (`.RBV`/`.VAL`, `.DMOV`, `.HLM`/`.LLM`, the switches, `.MSTA`
  decoded by name, `.VELO`/`.VMAX`), so one page fits every motor driver.
  A field whose channel dropped is `None`, never a stale value; the axis
  reads `unknown` while any field its state is judged from is unread (a
  known failure still wins). It writes `.VAL`, the tweak (`.TWV`,
  `.TWF`/`.TWR`), `.STOP`, homing (`.HOMF`/`.HOMR`) and `.VELO` behind an
  allow-list that refuses a target
  outside the soft limits before the record sees it (the record refuses
  it too, with `LVIO`); a move's put is dispatched without waiting (the
  record completes it at the end of the motion; the stream shows the
  move). **Direct puts are a prototype choice**: the design note (§1)
  sends a moving kind's writes through the scanner's `POST /api/move`,
  serialized with scans, which waits for the scanner to know native axes
  (GeecsBluesky `NATIVE_DEVICES.md` §10, slice L). Never `.OFF`, never the
  record's limits or `.SET`.
- **The power-supply panel speaks roles** (`power.py`): a supply family has
  no shared record set, so its bench entry maps roles (`current_sp`,
  `current`, `voltage`, `output_sp`, `output`, `reset`, optional
  `voltage_sp` and `loop_mode`) and labelled fault bits to its PVs. A
  setpoint must sit inside the record's drive limits, read and distinct,
  and below `HIHI` and above `LOLO` when their severity is INVALID (both
  severities, and an INVALID one's bound, read first): an `ao` CLAMPS a
  put past `DRVH` and writes the limit, so this check is the gate that
  refuses (the IOC's own refusal is HIHI + IVOA, set at load). A value
  whose channel dropped is `None`, never a stale number, and the supply
  reads `unknown` while a channel its state is judged from (the readback,
  the alarms, the output, each bit) is unread; a known failure still wins.
  The live put waits for the record and reads its alarm: a `#NAK`
  (INVALID) is a 502.
- **Per-device actions live in the bench file, not the registry.** A
  device's `allow` (the panel actions it permits, e.g. no homing for
  actuators on mirror mounts) and `note` are the panel's, so they live in
  the panels' own bench file (`registry.BenchFile`, `--bench`), which also
  lists bench positioners and power supplies that the worker's
  `native_devices.yaml` must not carry (a device the GEECS DB still owns); a
  bench positioner is the registry's `EpicsMotor` plus `prefix`, `allow` and
  `note`, a bench supply its roles and bits (`power_supply` exists only here). Bench scopes still come
  from `--scope NAME=PREFIX`. Each kind names its actions and the
  fields or roles they write (`motor.ACTIONS`, `power.ACTIONS`). A field outside them is a 403
  (`NotAllowed`), and the page leaves its control out. A device without
  `allow` keeps every action. Whether `allow` should move into the shared
  registry when a device migrates for real is open.
- Never imports GeecsBluesky, the scanner, the portal or the logbook
  (`.importlinter` `panels-edges`).

Testing: `poetry run pytest` — offline, fake transports. The live check is
running it against the test cameras and looking at the page.
