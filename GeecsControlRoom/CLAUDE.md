# GeecsControlRoom — Developer Context for Claude

The operator surfaces over Canopy (`Planning/operator_surfaces.md`): the
experiment's home, the **Control Room**, and the **screens renderer** —
a control GUI is a screen file in the configs repo drawn by one renderer,
never code per screen. FastAPI + Jinja on the kit, plain JavaScript, no
build step (§7: the first slice decides whether that stays).

## Layout

```
geecs_control_room/
  settings.py       Settings.from_env: GEECS_CONTROL_ROOM_* over the shared site keys
  app.py            create_app: the kit at /theme, /static, the pages, /healthz
  module_client.py  ModuleClient: GeecsCanopy's /geecs/v1 routes over HTTP (read key)
  home.py           the home at /: the strip (/api/home/strip, the scanner's API), the
                    "Not live" and "In alarm" lists (/api/home/devices), the launcher (home.yaml)
  screens.py        load(): a screen file bound by resolve_screen; GET /screens/{name},
                    its image files, and the two module pass-throughs (/api/devices/{name}[/put])
  static/hare.js    GeecsHare: Hare's WebSocket, reconnect + resubscribe
  static/home.js    the home's live half: polls the two /api/home reads, fills the device pickers
  static/screen.js  the renderer's live half: PVs and verdicts from the module, values from Hare, writes
  templates/        base.html (every page's shell), screen.html (EVERY screen), the pages
examples/sim_bench.yaml   the worked example: the geecs-epics simulator bench's four devices
docs/             the example rendered live (the PR's screenshots)
deploy/
  geecs-control-room.service   unit TEMPLATE (rendered by deploy/render_units.sh)
```

## Rules that hold here

- **A client of Canopy, never a part of it.** Canopy and GeecsCanopy are
  services: Hare's WebSocket for live values, the module's routes for
  verdicts and writes. Never import `geecs_canopy`, GeecsBluesky, the
  portal or the logbook (import contract `control-room-edges`).
- **Read keys only (condition (a), `Planning/geecs_canopy.md` §7.6).** The
  page carries the display's read key (on `<body>`, for Hare's socket);
  a write key never reaches this process. Every write is
  `POST /geecs/v1/devices/{name}/put` with the operator's name, sent once:
  the page shows the module's outcome word and code and never retries.
- **A screen binds, groups, confirms and calls; it never computes.** No
  expressions, scripts or conditions in a screen file, and no device logic
  in the renderer: verdicts are the module's (`freshness.py` there), alarms
  and connection are Hare's. A changing value is never evidence of
  freshness.
- **One template draws every screen.** `screen.html` + `screen.js`, from
  the screen document (GEECS-Schemas `Screen`, `resolve_screen`). A new
  screen is a file in `<experiment>/screens/`, never code. A widget's word
  is the module's verdict for a device role and Hare's connection and alarm
  for a bare PV (read only in this slice); "unknown" whenever either source
  is silent. PV names come from the module's device route, never derived
  here.
- **The home sorts the module's words; it judges nothing.** "Not live" is
  every device whose state is not `live`; "In alarm" every device whose
  state or any role's verdict is `error` (the module's word for severity
  major or invalid, and for a tripped fault bit). The strip's run word is
  the manager's `re_state`, as on the scanner's page. A list the module
  could not give reads "unknown", never "none". A row's cause is the module's evidence (heartbeat word,
  roles by verdict). The launcher is `<experiment>/home.yaml` (GEECS-Schemas
  `HomeLayout`); its link URLs are settings, never the file's.
- **The GEECS device name is the one identity key.** A binding names a
  device and a family role (and an axis), never a PV a second time; a bare
  PV is for a DB-served device only.
- **No facility value in code.** The experiment, the configs root and
  Canopy's URLs are settings; the home's title is "Control Room" unless
  `GEECS_CONTROL_ROOM_TITLE` says otherwise. The read key is a secret
  (`/etc/geecs/control-room/secrets.env`), never a `site.env` key.
- **The kit's rules** (`GeecsWebTheme/CLAUDE.md`): tokens, never a literal
  colour (this package's templates and static files are in
  `GeecsWebTheme/tests/test_no_literal_colours.py`'s `_SURFACES`); status
  is a dot and a word.

## Settings

| Key | Meaning |
|---|---|
| `GEECS_CONTROL_ROOM_EXPERIMENT` | the experiment (unset: `GEECS_EXPERIMENT`) |
| `GEECS_CONTROL_ROOM_CONFIGS_ROOT` | the configs repo root (unset: `GEECS_CONFIGS_ROOT`) |
| `GEECS_CONTROL_ROOM_CANOPY_URL` | Canopy as this process reaches it |
| `GEECS_CONTROL_ROOM_CANOPY_BROWSER_URL` | Canopy as the browser reaches it (the front door's `/canopy`); unset: the above |
| `GEECS_CONTROL_ROOM_READ_KEY` | the display's Canopy read key (secret) |
| `GEECS_CONTROL_ROOM_BIND`, `_PORT` | where it listens (127.0.0.1, 8700) |
| `GEECS_CONTROL_ROOM_TITLE` | the page title (default "Control Room") |
| `GEECS_CONTROL_ROOM_SCANNER_URL` | the scanner as this process reaches it (the strip); unset: "not configured" |
| `GEECS_CONTROL_ROOM_SCANNER_BROWSER_URL` | the scanner link as the browser reaches it; unset: the above |
| `GEECS_CONTROL_ROOM_PORTAL_URL`, `_LOGBOOK_URL`, `_CONFIG_EDITOR_URL` | the launcher's links (browser URLs); unset: a disabled entry |

## Testing

`poetry run pytest tests -q`: the module is stubbed with httpx's
`MockTransport`; `test_hare_js.py` runs `hare.js` under node with a fake
socket (skipped without node).
