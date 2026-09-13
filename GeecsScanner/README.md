# GeecsScanner

The web scanner console: submit, watch and stop GEECS scans from a browser.
A FastAPI service on the worker host over the queueserver client
(`geecs_bluesky.qs_client`), the third web surface on the GEECS surface kit
beside the Data Portal and the logbook, and the replacement for the PySide6
GEECS-Console. Arc brief: `Planning/native_bluesky/10_web_scanner.md`.

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
| `GET /api/events?since=&once=` | the SSE stream |

Errors are `{"error": {"kind", "message", ...}}` with the kind mapped to a
status: `invalid_request` 400, `not_found` 404, `policy_refusal` 409,
`manager_unreachable` 503.

## Deploy

Its own process, port 8300, unit `geecs-scanner`, behind the same front
door as the portal and logbook. See `deploy/DEPLOYMENT.md`.
