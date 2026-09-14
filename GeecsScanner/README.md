# GeecsScanner

The web scanner console: submit, watch and stop GEECS scans from a browser.
A FastAPI service on the worker host over the queueserver client
(`geecs_bluesky.qs_client`), the third web surface on the GEECS surface kit
beside the Data Portal and the logbook, and the replacement for the PySide6
GEECS-Console. Arc brief: `Planning/native_bluesky/10_web_scanner.md` (#869).

## What it is

- **`geecs_scanner.service`** — pure Python over the injected client and
  configs resolver. `ScannerService` answers status, queue, history, config
  listings, the scan-variable catalog (pseudo entries listed, not
  scannable), preflight, submit (every preflight question acknowledged, the
  `SubmissionRecord` stamped into the run metadata), pause/resume/stop,
  clear. `ProgressCache` reduces the worker's pickled document stream and
  the manager's console text to one small picture.
- **`geecs_scanner.web`** — the JSON API, `GET /api/events` (Server-Sent
  Events: `status`, `progress`, `console`), `/health`, and the kit served
  at `/theme`. The page arrives in 0.2.0.

The page is one Jinja template (`templates/console.html`) and one script
(`static/scanner.js`) over that API; open `http://<host>:8300/` in a browser.

The submission document is the `geecs_schemas.Preset`: a device group, a
plan call, a trigger profile. The scanner expands it with the same
`expand_preset` every client uses and binds no device itself.

## Run it

```bash
poetry install
poetry run geecs-scanner --experiment <Experiment>        # the real manager, from config.ini [qserver]
poetry run geecs-scanner --demo                           # an in-memory manager that runs scans by itself
curl -s localhost:8300/api/status | jq
curl -N localhost:8300/api/events                         # watch the stream
```

`--demo` contacts nothing: it is how the API and the page are exercised
without a worker.

## Routes

| route | answers |
|---|---|
| `GET /` | **the page** — Now, New scan, Queue on the kit |
| `GET /health` | liveness, manager reachable, readiness word, version |
| `GET /api/status` | one manager poll + the readiness verdict |
| `GET /api/queue?history=` | running / waiting / finished rows, summarized |
| `GET /api/progress` | the latest-run picture (also on the event stream) |
| `GET /api/configs/{kind}` | `presets`, `trigger_profiles`, `scan_variables`, `actions`, `optimizer_configs` |
| `GET /api/configs/presets/{name}` | one preset document |
| `GET /api/scan-variables` | the catalog with `kind` and `scannable` |
| `GET /api/devices` | every device reference the manager resolves |
| `POST /api/preflight` | validate + pre-check a preset; submits nothing |
| `POST /api/submit` | `{preset, acknowledged[], operator, clear_pending}` → the queued item |
| `POST /api/pause` · `/resume` · `/stop` · `/clear` | the verbs |
| `GET /api/settables` | every numeric settable of the experiment from the GEECS DB, aliased first; the value is the canonical `Device:Variable` |
| `GET /api/readback?variable=&units=` | one live reading of the gateway's **readback** PV (never `:SP`): value, stamp, age; `ok: false` when the gateway does not answer |
| `POST /api/move` | `{variable, value, operator}` → one `mv` queue item; idle-only (409 while a plan runs or anything waits) |
| `GET /api/actions` · `GET /api/actions/{name}` | the action library; the preview — every step, nested `run` plans inlined |
| `POST /api/actions/{name}/run` | one `run_action` queue item; idle-only; refused when the preview does not resolve |
| `GET /api/calibration` | the stored `shot_offsets.yaml`, summarized |
| `POST /api/calibration/check` · `/measure` | `check_shot_sync` / `measure_shot_offsets` queue items over `{devices[], trigger_profile, tolerance_s | shots, write}`; idle-only |
| `POST /api/configs/presets/{name}` | `{preset, overwrite}` → writes `presets/<name>.yaml` through the resolver (409 `exists` without `overwrite`) |
| `GET /api/scanlog?offset=` | the latest run's `scan.log` from the folder the start document names (read-only) |
| `GET /api/events?since=&once=` | the SSE stream: `status`, `progress`, `log`, `console` |

Errors are `{"error": {"kind", "message", ...}}` with the kind mapped to a
status: `invalid_request` 400, `not_found` 404, `policy_refusal` 409,
`manager_unreachable` 503.

## Deploy

Its own process, port 8300, unit `geecs-scanner`, behind the same front
door as the portal and logbook. See `deploy/DEPLOYMENT.md`.
