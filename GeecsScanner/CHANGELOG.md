# Changelog

All notable changes to `geecs-scanner` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.1.0] - 2026-09-13

The service layer and the HTTP API — PR 2 of the web scanner arc
(`Planning/native_bluesky/10_web_scanner.md`). No page yet; that is 0.2.0.

### Added

- **`geecs_scanner.service.ScannerService`** — the verbs over an injected
  `geecs_bluesky.qs_client` client and configs resolver, one lock, every
  answer a Pydantic model: `status` (manager poll + readiness verdict),
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
  fallback, shots from primary events, exit from the stop document, the
  failed-move prefix as the paused reason) plus a ring of console lines
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

### Not yet

The page (0.2.0). The scan.log tail (needs the run's folder; the stream
gains a `log` event type with it). `mv`, `run_action`, the calibration
verbs (0.3.0). The operator registry and ownership-gated stop (arc PR 5).
