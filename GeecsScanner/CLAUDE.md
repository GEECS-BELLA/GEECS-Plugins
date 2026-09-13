# GeecsScanner — Developer Context for Claude

The web scanner console: a FastAPI service on the worker host that submits,
watches and stops scans through `geecs_bluesky.qs_client`. The third surface
on the GEECS surface kit (GeecsWebTheme), a peer of the Data Portal and the
logbook, and the replacement for GEECS-Console. The arc brief is
`Planning/native_bluesky/10_web_scanner.md` (#869); read it before changing shape.

## Layout

```
geecs_scanner/
  service/        pure Python — the tested surface; no FastAPI anywhere in it
    scanner.py    ScannerService: the verbs over the injected client + resolver
    models.py     every answer as a Pydantic model; `state` fields are kit words
    summaries.py  a stock plan item → one line + planned shots (ported from the console)
    streams.py    ProgressCache: the document + console streams reduced to a picture
    demo.py       DemoQueueClient / DemoResolver / demo_preflight — a manager in memory
    errors.py     ScannerError(kind, message, **extra); kind → HTTP status in one table
  web/
    app.py        create_app (the process) and create_scanner_router (the same as a router)
    pages.py      GET / — the page (Jinja2Templates with `root` in every context)
    api.py        one route per verb, three lines each
    events.py     GET /api/events — SSE: status, progress, console
  templates/console.html   the page; the kit's vocabulary, page-only classes from static/
  static/scanner.js        the page's one script: API + EventSource + form → Preset
  static/scanner.css       page-only classes; never re-skins a kit class
  __main__.py     geecs-scanner: --experiment | --demo, --port 8300, --root-path
deploy/           the unit template + DEPLOYMENT.md
```

## Rules

- **Client-side expansion, nothing re-derived.** A scan is a
  `geecs_schemas.Preset`; `expand_preset` turns it into the plan call; the
  scanner never binds a variable to a device itself and never computes
  detectors. `md["geecs"]` is provenance only. The scan number is claimed
  worker-side and read from the start document.
- **Plan-name generic.** `POST /api/submit` takes a preset whose plan call
  names any allowed plan. There is no mode enum in the API — an optimize
  plan, when it exists on the worker, is one more name and one more form
  section, not an API change (brief §5).
- **The catalog lists pseudo entries** with `scannable: false` and the
  refusal text. When the pseudo arc lands, the flag flips; nothing else in
  the API moves.
- **Operator policy, not agent policy.** No shot cap, no "refuse if
  anything is queued" — those are GEECS-MCP's posture. Preflight questions
  must be acknowledged; that is the one gate. Ownership (who may stop whose
  scan) arrives with the operator registry (arc PR 5); until then `force`
  is recorded and changes nothing.
- **One client, one lock, blocking calls.** Every service method holds the
  lock around its client calls; the web layer runs them on the threadpool.
  The stream consumer threads are daemons that are never stopped (a zmq
  socket touched from another thread can abort the process).
- **Imports.** `geecs_bluesky.qs_client`, `geecs_bluesky.config_resolver`,
  `geecs_bluesky.plan_names`, `geecs_schemas`, `geecs_web_theme`. Never the
  portal, the logbook, the console, GEECS-MCP, or the engine's
  `plans`/`devices`/`run_engine`/`namespace`. `tests/test_boundaries.py`
  pins this and the absence of facility literals in code.
- **Errors are the taxonomy.** Raise `ScannerError(kind, message, **extra)`;
  the web layer maps the kind. Never raise HTTPException in the service.
- **The demo is sample content**, chosen by a flag; nothing in it is a
  default for a deployment.

## Testing

`poetry install` then `pytest`. The suite runs over `DemoQueueClient` with
`period=0`, stepping the fake manager by hand — deterministic and fast.
`GET /api/events?once=1` is the one-round form of the stream for tests and
`curl`.

## The page

`console.html` is filled entirely by `scanner.js` from `/api/*` and kept
live over `/api/events`; Jinja renders only the experiment, identity and
version. Rules the logbook learned, pinned by `tests/test_page.py`:
`url_for(...).path` never the absolute URL; every literal `data-state`
(template or `setChip` in the script) is a kit word; every script passes
`node --check`; every class the page uses is styled by the kit, the theme
or `scanner.css`. The first three are `geecs_web_theme.testing`'s helpers
— the test file asserts over their findings and adds only what is the
scanner's own (the script's `K` table, `setChip` literals). Likewise the
forwarded-prefix middleware, the `/theme` mount and the templates factory
come from `geecs_web_theme.web` (the `web` extra): import them, never copy
them. The form builds a `Preset` and posts it — the page never
resolves a variable to a device.

## What is not here yet

The scan.log tail (needs the day folder; the SSE stream
carries the manager's console text now and gains a `log` event type then),
`mv` / `run_action` / calibration verbs (0.3.0), the operator registry and
ownership (arc PR 5).
