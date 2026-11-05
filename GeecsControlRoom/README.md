# GeecsControlRoom

The operator surfaces over [Canopy](https://github.com/slaclab/canopy) and
the GEECS device module (`GeecsCanopy`): the experiment's home, the
"Control Room", and the screens renderer — a control GUI is a screen
file drawn by one renderer (`Planning/operator_surfaces.md`).

A FastAPI service on the kit (`GeecsWebTheme`), plain JavaScript, no build
step. Live values come from Hare's WebSocket with a read key; every write
goes through the module's put route. `CLAUDE.md` holds the rules.

    GEECS_EXPERIMENT=… GEECS_CONFIGS_ROOT=… \
    GEECS_CONTROL_ROOM_CANOPY_URL=http://127.0.0.1:8600 \
    GEECS_CONTROL_ROOM_READ_KEY=… poetry run geecs-control-room

## Deploy

On the services host, beside Canopy (`GeecsCanopy/DEPLOYMENT.md`), from its
own clone `<GEECS_CHECKOUT_ROOT>/control-room-checkout` (`deploy/bootstrap_host.sh`
clones it, installs the env and renders the unit). The `GEECS_CONTROL_ROOM_*`
keys are in `site.env` (`deploy/site.env.example`); on a host without a front
door, `GEECS_CONTROL_ROOM_BIND=0.0.0.0` and `GEECS_CONTROL_ROOM_CANOPY_BROWSER_URL=http://<host>:<Canopy's port>`
(the browser opens Hare's WebSocket there; a path like `/canopy` would be this
page's own origin). The secret is
the display's read key, the `read@*` entry of Canopy's `CANOPY_API_KEYS`, never one
that carries write:

    sudo install -d -m 0755 /etc/geecs/control-room
    sudo install -m 0600 -o <service account> /dev/null /etc/geecs/control-room/secrets.env
    # then, as root: GEECS_CONTROL_ROOM_READ_KEY=<the display key>
    sudo install -m 0644 <staging>/geecs-control-room.service /etc/systemd/system/
    sudo systemctl daemon-reload && sudo systemctl enable --now geecs-control-room

CA addressing is `site.env`'s `EPICS_CA_*`.

### Raw frames

The camera page's Raw view reads the driver's frame over CA on this host and
relays it to the browser (`GET /api/devices/{name}/frame.raw`, gzip framed).
What it costs, measured 2026-11-03:

| Camera | Frame | Read over CA | On the wire, gzip level 1 |
|---|---|---|---|
| a 2592×2048 camera | 10.6 MB, read as 16-bit (`DBR_SHORT`; the native read was 21 MB of int32) | ~0.2 s | ≈ 3.8 MB per frame, ~0.1 s to compress |
| a Blackfly, 600×600 ROI | 0.72 MB | 20 ms | 0.2 MB |

A raw view polls at 2 Hz and the frame is read **once per camera**, shared by
every viewer (held 0.5 s; an unmoved camera costs one scalar read of
`ARR:ArrayCounter_RBV`, never an array read). Budget for twenty camera views:
~0.2 Gbit/s if all are MJPEG, ~1.3 Gbit/s if every one went raw at once, inside
the ~5 Gbit/s the services box's link keeps free for analysis reads. The
MJPEG view's JPEG is held the same way on MJPG1's `ArrayCounter_RBV`: a 1 Hz
camera's JPEG is fetched from the IOC once per second however many viewers
poll it at 10 Hz (before the hold, up to ten times a second: 7.5 measured with one viewer). A read
peaks at ≈ 15 MB of memory per camera (10.6 MB int16, viewed as uint16 without a
copy, + 4 MB gzip; 36 MB while the frame was read as int32).

- **A proxy in front of the Control Room must pass `Content-Encoding: gzip`
  through unchanged**: the body is gzip framed by this server and the
  browser's `fetch` decodes it. A front door that drops the header but keeps
  the gzip body, or compresses the body a second time, hands the page bytes
  that do not match the frame's size: the server answered 200, and the page
  says `the frame is <n> bytes for <w>×<h>` (not the server's `geometry
  disagrees`, which is a camera-ROI change).
  The body is already compressed: exclude `application/octet-stream` from the
  proxy's own compression.
- **A camera-family change rebuilds Canopy's image at deploy**: the module's
  roles (the raw view's `raw_frame`, `raw_width`, `raw_height`,
  `raw_counter`) come from the image's copy of GEECS-Schemas, so
  `dc build` and a `geecs-canopy` restart (`GeecsCanopy/DEPLOYMENT.md`) come
  before the Control Room's restart; until then the raw view answers 404
  (the module names no `raw_frame` role).

`EPICS_CA_MAX_ARRAY_BYTES` need
not be set for the raw frame (`/api/devices/{name}/frame.raw`, a 2592×2048
frame is 10.6 MB as it is read, `DBR_SHORT`): the client library aioca loads
(epicscorelibs' libca, EPICS 7) sizes its buffers itself while
`EPICS_CA_AUTO_ARRAY_BYTES` is unset or `YES`, its default. Measured on a
local soft IOC (5 308 416 `USHORT` values, read as int32): unset, the read
succeeds; with `EPICS_CA_AUTO_ARRAY_BYTES=NO` and no `MAX_ARRAY_BYTES` it
times out (libca drops the payload). So the unit must not set
`EPICS_CA_AUTO_ARRAY_BYTES=NO`; if a site does, it must also set
`EPICS_CA_MAX_ARRAY_BYTES` above the largest frame × 2 bytes (≥ 10.7 MB for a
2592×2048 camera). The IOC side's limit is the IOC's (geecs-epics).

Check: `GET /healthz` → `{"status": "ok"}`. The home and the screens read the
configs share per request; Canopy's containers do not (restart `geecs-canopy`).
