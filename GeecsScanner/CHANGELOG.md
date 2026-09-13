# Changelog

All notable changes to `geecs-scanner` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.2.0] - 2026-09-13

The page — PR 3 of the web scanner arc. Day one of the console's job, as
the reviewed mock drew it, on the kit.

### Added

- **`GET /`** renders `templates/console.html`: topbar (experiment, manager
  and doc-stream chips, operator, density and theme pickers), rail
  (sections, presets from the configs tree, recent scans), and three
  panels — **Now** (`.chip.lg` state, scan number, plan line, `.meter`,
  `.live` shots / planned / document age / manager age, Pause · Resume ·
  Stop, the manager console tail behind a `.seg` whose scan.log half
  arrives with the run's folder), **New scan** (mode `seg`: No-scan · 1D ·
  Grid · Background · Optimize greyed; acquisition `seg`, strict default;
  axes with `aria-invalid` validation; shots, trigger profile, shot period,
  description; the preset's devices as a table with save-images and
  essential boxes; the estimate; Start), **Queue** (running / waiting /
  finished, sticky header, Clear queue…).
- **`static/scanner.js`** drives it: the JSON API for configs and the queue,
  `EventSource` on `/api/events` for status, progress and console lines
  (with `since` on reconnect), the form → `Preset` builder (a `count`,
  `scan` or `grid_scan` plan call from the fields; the preset's device
  group edited in place), the acknowledgement dialog over the preflight
  questions, and the **replace-the-waiting-item** dialog for the manager's
  one-deep queue (a 409 with `pending_items` → resubmit with
  `clear_pending`). Operator name kept under `geecs.operator` beside the
  theme's keys until the registry (arc PR 5).
- **`static/scanner.css`** — the page's own classes only; the kit is not
  re-skinned.
- `tests/test_page.py` — the logbook's three template guards (`url_for`
  takes `.path`, literal `data-state` values are kit words — the script's
  `setChip` literals too, every script parses under `node --check`), the
  proxy-prefix render, and "every class the page uses is styled".

### Changed

- The JSON pointer that was `GET /` moved to `GET /api`.

## [0.1.0] - 2026-09-13

The service layer and the HTTP API — PR 2 of the web scanner arc
(`Planning/native_bluesky/10_web_scanner.md` (#869)). No page yet; that is 0.2.0.

### Added

- **`geecs_scanner.service.ScannerService`** — the verbs over an injected
  `geecs_bluesky.qs_client` client and configs resolver — the writes hold
  one lock, the reads none, so a two-minute graceful stop never freezes a
  status poll — every answer a Pydantic model: `status` (manager poll + readiness verdict),
  `queue` (running / waiting / finished, summarized), `list_configs`
  (`presets`, `trigger_profiles`, `scan_variables`, `actions`,
  `optimizer_configs` — the last listed now so the day an optimize plan
  exists the listing does), `scan_variables` (the catalog with `kind`;
  pseudo entries listed with `scannable: false` and the refusal text),
  `preset`, `devices`, `preflight`, `submit`, `pause`, `resume`, `stop`,
  `clear`, `progress`.
- **Submission policy**: preflight runs before every queue; a refusal is
  `invalid_request`; an unacknowledged question is `policy_refusal`
  carrying `needs_acknowledgement`; acknowledged checks are stamped
  `continued` into the `SubmissionRecord` in `md["geecs"]`, with the
  operator's name beside it. No shot cap, no queue-etiquette refusal —
  those are agent posture (GEECS-MCP), not an operator's.
- **`ProgressCache`** — the worker's pickled document stream and the
  manager's console text reduced to one picture (scan number, planned
  total from the start document with `max_iterations` as the adaptive
  fallback, shots from the **row streams** — `primary` for a strict run,
  the sampler's `shots` for a gated one, the same pair the s-file writer
  counts — exit from the stop document, the failed-move prefix as the
  paused reason, cleared by the next row) plus a ring of console lines
  with a cursor. Daemon consumer threads, never stopped.
- **`summaries.summarize_item`** — a stock plan item (`count`, `scan`,
  `grid_scan`, `list_scan`, the relative variants, `mv`, `run_action`,
  the calibration plans) read back into one line and a planned shot
  count; never raises on a shape it does not know. Ported from the
  console's `queue_panel`, rewritten for the stock verbs.
- **The web layer**: `create_app` / `create_scanner_router`, the portal's
  forwarded-prefix middleware, `/theme` mounted from GeecsWebTheme,
  `/health`, the JSON API (`no-cache` listings, errors as
  `{"error": {"kind", ...}}` mapped kind → status), and **`GET /api/events`**
  — Server-Sent Events with `status`, `progress` and `console` event types,
  a `since` cursor for reconnects and `once` for a single round.
- **`--demo`** — an in-memory manager (`DemoQueueClient`) that runs
  submitted items by itself, emits the same bluesky documents the worker
  would, pauses at step boundaries and stops gracefully; a `DemoResolver`
  with three real presets and a catalog carrying a pseudo entry; a
  `demo_preflight` that validates by the real expansion and asks one
  question. The test suite runs on it with the manager stepped by hand.
- `tests/test_boundaries.py` pins the import boundary (no portal, logbook,
  console, MCP or engine internals) and the absence of facility literals.
- `deploy/geecs-scanner.service` (template) + `deploy/DEPLOYMENT.md`; the
  unit joins `deploy/render_units.sh` and the fleet map (port 8300).

### Fixed (from adversarial review, before merge)

- Gated runs advanced no progress: only `primary` events counted, and a
  gated run's rows are the `shots` stream. Both row streams count now,
  pinned against `geecs_bluesky.callbacks.ROW_STREAMS`.
- One lock across reads and writes: a graceful stop (up to 120 s) froze
  every status poll and the SSE stream. The reads are lock-free.
- The demo disagreed with the worker three ways: a stopped run's stop
  document said `abort` (RunEngine.stop marks it `success`); pause was
  written into the progress picture (on the worker only the manager's
  `re_state` says paused); a second submission while one waited was
  accepted (the real client refuses with `pending_items` unless
  `clear_pending`). All three now match the worker.

### Not yet

The page (0.2.0). The scan.log tail (needs the run's folder; the stream
gains a `log` event type with it). `mv`, `run_action`, the calibration
verbs (0.3.0). The operator registry and ownership-gated stop (arc PR 5).
