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
  home.py           the home at /: the strip (/api/home/strip, the scanner's API; /api/home/baseline,
                    the golden presets' diff), the "Not live" and "In alarm" lists (/api/home/devices),
                    the launcher (home.yaml), the presets (/api/home/presets and the pass-throughs
                    /api/presets[/{id}/diff|golden|restore] to the module's preset routes)
  screens.py        load(): a screen file bound by resolve_screen; GET /screens/{name},
                    its image files, and the module pass-throughs (/api/devices/{name}[/put],
                    /api/actions, …/{plan}/run, …/runs/{id}[/stop])
  static/hare.js    GeecsHare: Hare's WebSocket, reconnect + resubscribe
  static/home.js    the home's live half: polls the /api/home reads (the presets and the baseline
                    slowly), fills the device pickers, the presets region (save dialog, diff drawer,
                    restore confirm, the run on the row, the golden mark)
  static/screen.js  the renderer's live half: PVs and verdicts from the module, values from Hare, writes
  static/screen_rules.js, home_rules.js   the pages' rules, pure (verdict word, value shown,
                    bounds, writable, differs, outcome state; the home lists' words)
  static/poller.js  the pages' one poll loop: one request in flight per target, late answers
                    dropped, a timeout, back-off to 10 s, at most 8 requests at once
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
  for a bare PV, read only forever; "unknown" whenever either source is
  silent. PV names come from the module's device route, never derived
  here. A DB device (on the module's roster, in neither device file) binds
  `{device, role: <GEECS variable>}` and is **deferred**: the page judges it
  by the module's device route ("unknown role" when the module lists no
  such role, "wrong kind" when its type does not fit the kind, Hare's alarm
  rules where the module gives no verdict) and writes it through the same
  put route. A writing kind on a bare PV is drawn disabled.
- **A button may name an action plan** (`bind: {action: <plan>}`, checked
  against the experiment's `action_library/actions.yaml`). The module runs
  it, every `set` through its gates; the page asks once (the plan's
  description and its hazard steps) when it writes a hazard role, sends
  the run once, and shows the module's run (`running 2/5`, `failed · step
  3 · out_of_bounds`) with Stop while it runs. A sequence is never
  scripted in a screen.
- **Dense, like the panel it replaces.** When no group spans columns,
  each column packs its groups top to bottom in row order (a LabVIEW
  panel's layout; grid row tracks would leave gaps under short groups);
  otherwise the grid aligns rows. A group's `columns` lays its widgets
  side by side, row by row.
- **Presets are the module's.** The home lists Squirrel's snapshots as the
  module's `/v1/presets` gives them, saves with the header's operator and a
  scope of families and screens (the module resolves both), shows a diff
  drawer (device · role · saved · live · tolerance · word, the module's
  words), restores a device subset as a run the module synthesizes (the
  confirm dialog first; the run shown on the row as on a screen button),
  and marks golden by name. The strip's "Baseline" chip is the golden
  snapshots' differing rows, polled every 10 s; hidden without a golden.
- **The home sorts the module's words; it judges nothing.** "Not live" is
  every device whose state is not `live` (a DB device's state is its
  gateway `connected` PV, so a Disconnected LabVIEW device is listed; a live
  one is not read in detail: its roles carry no module verdict); "In alarm" every device whose
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
`MockTransport`. The page scripts run under node (skipped without it; no
npm, no jsdom): `test_hare_js.py` with a fake socket, `test_poller_js.py`
with a fake fetch, and `test_screen_rules_js.py` / `test_home_rules_js.py`
as tables, one row per rule (`tests/_node.py` is the harness). A rule a
page needs goes in a `*_rules.js` file with a row here; the page script
keeps the DOM work only.

`test_module_contract.py` is the cross-service contract: GeecsCanopy's own
module (`geecs_canopy.testing.make_env`: its routes over a fake Hare) mounted
at `/geecs` behind `ModuleClient`, driven through this app's routes. It pins
the fields the pages read (a role's `role`, `axis`, `pv`, `read`, `writable`,
`verdict`; the put envelope's `outcome`, `code`, `journalId`, `detail`; a
journal row's fields) and the home's lists and `unread`. geecs-canopy is a
dev-only dependency; `geecs_control_room` never imports it.
