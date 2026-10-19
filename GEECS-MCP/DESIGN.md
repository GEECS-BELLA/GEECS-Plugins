# GEECS-MCP as the agent's single write entry point — design note

Status: **built, 2026-10-17** (proposed 2026-10-16; PRs A–G of §9,
geecs-mcp 0.10.0). `CLAUDE.md` is the package as built; where the two
differ, the code and `CLAUDE.md` win and this note keeps the reasoning.
The verb names are the contract the htu-assistant deployment imports
(`tool_names.py`); the contract tests pin the endpoint and envelope
columns.

## 1. Decisions this note builds on (settled, not reopened here)

- `geecs-mcp` becomes a **thin HTTP client** of three services on the
  worker box: GeecsScanner (`:8300`), GEECS-DataPortal (`:8200`),
  GeecsLogbook (`:8400`). No engine imports, no `qs_client`, no
  ScanAnalysis. The server re-implements no preflight, no limits, no
  checks: what the agent does is what an operator sees in the scanner
  and on the portal's Analysis tab.
- Analysis runs through the **portal's analysis route**; the server's own
  `run_scan_analysis` runner (0.7.0, #686) is retired.
- Moves go through the scanner's `/api/move` (the Bluesky `mv` plan).
- Logbook posts go to the **scans** book; every submission and every
  entry carries the identity string `agent, admin supervision (<operator>)`.
- No campaign envelope, budget or token.
- Deployed as the existing `geecs-mcp` systemd unit on the gateway box,
  replacing the nohup 0.9.0 process.

The two PMQ alignment campaigns (2026-10-08, 2026-10-15) ran this loop
by hand against the same three HTTP APIs; this server packages that
practice as tools and nothing more. The approach page shared with the
OSPREY team (the "approach page" below) lives with the htu-assistant
deployment, not in this repo.

## 2. Shape

```
osprey (htu-assistant)  --MCP/http :8100-->  geecs-mcp
                                               |-- httpx --> geecs-scanner   :8300  /api/*
                                               |-- httpx --> geecs-data-portal :8200 /api/run/*, /run/*/artifact
                                               `-- httpx --> geecs-logbook   :8400  /api/entries*
```

Package after the rebuild:

```
geecs_mcp/
  server.py        FastMCP("geecs") + create_server()        (kept)
  __main__.py      stdio | --transport http                  (kept; stream warm-up gone)
  tool_names.py    the contract                              (rewritten, §3)
  errors.py        make_ok / make_error                      (kept; kinds in §5)
  config.py        the three base URLs + timeouts from config.ini [mcp]   (new, replaces runtime.py)
  clients.py       one httpx.AsyncClient per service; HTTP → envelope mapping; the test transport seam (new)
  scans/tools.py   scanner-backed verbs                      (replaces read_tools/control_tools/progress_stream)
  analysis/tools.py portal-backed verbs                      (replaces read_tools/run_tools/run_worker)
  logbook/tools.py post_logbook_entry                        (new domain)
```

Runtime dependencies: `fastmcp`, `httpx`, `anyio`, `pillow` (the
opt-in thumbnail only). **Removed**: `geecs-bluesky`, `geecs-data-utils`,
`scananalysis`, `pyyaml`, the `analysis-run` extra, the
`/figures/` route, the progress-stream consumer, `[mcp] client_identity`.
Tools become plain `async def` over `httpx.AsyncClient` — the
`anyio.to_thread` wrapper pattern goes with the sync seams it wrapped.

Import contract (`.importlinter`): the existing `package-layers`
contract already keeps `geecs_mcp` from importing its layer-mates
(`geecs_scanner`, `geecs_portal`, `geecs_logbook`). A new forbidden
contract `mcp-is-a-client` adds the lower layers: `geecs_mcp` never
imports `geecs_bluesky`, `scan_analysis`, `geecs_data_utils`,
`geecs_core` or `geecs_schemas` — documents (presets, entries, payloads)
pass through as dicts and the services validate them. The three services
stay **dev-group** path dependencies (for the in-process contract tests,
§7), which is also what makes `scripts/ci_select.py` run the MCP leg
when any of them changes.

## 3. Verb table

Class: **R** read (allow), **W** write (the tier decides; the approach
page puts them on allow at admin). Names are unchanged where kept so the
profile's existing entries survive.

| Verb | Class | Service | Endpoint(s) | Status |
|---|---|---|---|---|
| `scan_status` | R | scanner | `GET /api/status`, `GET /health` | keep |
| `scan_progress` | R | scanner | `GET /api/progress` | keep |
| `scan_history` | R | scanner | `GET /api/queue?history=N` | keep |
| `list_scan_configs` | R | scanner | `GET /api/configs/{kind}`; `name=` → `GET /api/configs/presets/{name}` | keep (+`name`) |
| `get_scan_analysis` | R | portal | `GET /api/run/jump/{day}?prefer=N` → `GET /api/run/{uid}/analysis` | keep |
| `get_scan_figure` | R | portal | same + `GET /run/{uid}/artifact?path=` | keep |
| `get_analysis_artifact` | R | portal | jump → `GET /run/{uid}/artifact?path=` (JSON/text sidecars, bounded) | **new** (§4.9) |
| `get_scan_scalars` | R | portal | jump → `GET /api/run/{uid}/binned?cols=&x=&bincfg=&filters=` | **new** (§4.9) |
| `list_analyzers` | R | portal | `GET /api/analysis/catalog` (**new portal route**, §4.8) | keep |
| `list_analysis_groups` | R | portal | same route, `groups` | keep (see §8) |
| `stop_scan` | W (halt) | scanner | `POST /api/stop {force, operator}` | keep |
| `pause_scan` | W (halt) | scanner | `POST /api/pause {force, operator}` | keep |
| `resume_scan` | W | scanner | `POST /api/resume {force, operator}` | keep |
| `clear_queue` | W | scanner | `POST /api/clear` | keep |
| `submit_preset` | W | scanner | `GET /api/configs/presets/{name}` → `POST /api/preflight` → `POST /api/submit` | **new** |
| `move` | W | scanner | `POST /api/move` → `GET /api/queue` → `GET /api/readback` | **new** |
| `wait_for_scan` | R | scanner | `GET /api/queue`, `GET /api/progress` (poll) | **new** |
| `run_analysis` | W | portal | jump → `POST /api/run/{uid}/analysis?analyzer=&confirm=` | **new** |
| `wait_for_analysis` | R | portal | `GET /api/run/{uid}/analysis` (poll) | **new** |
| `post_logbook_entry` | W | logbook + portal | logbook `POST /api/entries`; per figure, portal `POST /api/run/{uid}/logbook` (`path` source, §4.6) | **new** |
| `run_scan_analysis` | — | — | — | **drop** (→ `run_analysis`) |
| `get_scan_result` | — | — | — | **drop** — its numeric reads are replaced by `get_analysis_artifact` + `get_scan_scalars` (§4.9), which read what the loop actually decides on |

`tool_names.py` groups: `READ_TOOLS` (R rows), `WRITE_TOOLS` (W rows
except the halt family), `STOP_TOOLS` (`stop_scan`, `pause_scan`). The
old `QUEUE_TOOLS` name goes. The import path `geecs_mcp.tool_names` is
stable: the HTU profile derives its allow/ask lists and
`control_system.write_tools` from the three tuples, never retyped.
The deployment's posture (htu-assistant session, 2026-10-16):
`READ_TOOLS` on allow everywhere; `STOP_TOOLS` on allow wherever the
server renders; `WRITE_TOOLS` unprompted at the admin tier only — on
allow and on deny in the base profile, the deny removed by the admin
persona (lists only union across tiers; corrected 2026-10-18) — with the
profile's DEVIATION marker; the readonly and knowledge renders drop the
write verbs. The headless gate is the profile's `write_tools` list
derived from `WRITE_TOOLS`, as `deploy/DEPLOYMENT.md` describes today.
**Corrected 2026-10-18 by the OSPREY team's reply** (checked against
their `main`): the writes-check and approval hooks do not attach to a
profile-declared server by any route — a `hooks:` key on an
`mcp_servers` entry is dropped, and `claude_code.servers.<name>.hooks`
(which the htu-assistant render of 2026-10-16 appeared to wire) is
rewritten by the build for a server the profile declares — and the
writes-off deny skips custom servers. The levers today are
`control_system.write_tools` (headless floor, writes-check hook and
audit clamp), tiers by `deny` + `remove_deny`, and optionally a facility
PreToolUse hook for the kill switch; osprey is adding
`mcp_servers.<name>.hooks.pre_tool_use` and the custom-server deny.
`deploy/DEPLOYMENT.md` carries the profile shape.

## 4. Verb by verb

Every tool returns the JSON envelope of §5. Arguments shown as Python
signatures; defaults are the contract.

### 4.1 `submit_preset(name, acknowledged=(), operator)`

1. `GET /api/configs/presets/{name}` — 404 → `not_found` (with the
   `names` from `GET /api/configs/presets` so the agent picks from the
   catalog, never invents).
2. `POST /api/preflight` with the bare preset document. A `refusal` →
   `invalid_request` carrying `outcomes`. Any `questions` whose `check`
   is not in `acknowledged` → `policy_refusal` with
   `needs_acknowledgement: [questions verbatim]` and the preflight
   `summary`/`planned_shots`. **The server never acknowledges on its
   own**; the agent must pass the check names back, and the skill decides
   whether a human is asked first.
3. `POST /api/submit {preset, acknowledged, operator: <identity string>}`.
   The scanner reruns preflight; a 409 `needs_acknowledgement` here (a
   question that appeared between the two calls) is returned exactly as
   in step 2. A 409 `pending_items` (something already queued) →
   `policy_refusal` with the rows. `clear_pending` is never sent.
4. ok: `{item_uid, submitted_as, planned_shots, summary, plan:{name,args,kwargs}, preflight: outcomes}`
   — `item_uid`/`submitted_as` from the submit reply, `plan` and the
   `preflight` outcomes from step 2 (SubmitOut carries neither).

`item_uid` is the handle `wait_for_scan` takes. The scanner's submit
returns no scan number or run uid (they exist only once the worker has
claimed the scan).

### 4.2 `move(variable, value, operator, wait=True, timeout_s=60)`

1. `POST /api/move {variable, value, operator}`. The scanner resolves
   the variable (catalog name, `Device:Variable`, or bare device), checks
   finiteness, and is **idle-only**: 409 with `re_state` or
   `items_in_queue` → `policy_refusal` (unchanged text). Range limits
   are the DB's, enforced by the gateway on the `:SP` write — the MCP
   adds none.
2. With `wait`: poll `GET /api/queue` every 1 s until the `mv` item's
   `item_uid` appears in `finished[]` (or `timeout_s` elapses → ok with
   `finished: false`); then `GET /api/readback?variable=<Device:Variable>`.
   The readback route takes only the canonical `Device:Variable` (400
   otherwise), **not** the dotted `reference` the move reply returns, so
   the MCP reads back with the caller's `Device:Variable`, or, for a
   catalog name, with the `target` that `GET /api/scan-variables` lists
   for it; a bare device or a pseudo variable (`target` null — or the
   empty string, which the listing also emits and the MCP treats the
   same) has no readback and the field is `null` with a
   `readback_detail` saying why.
3. ok: `{item_uid, reference, finished, state, word, detail, readback:{value, units, timestamp, age_s} | null, readback_detail}`.

### 4.3 `wait_for_scan(item_uid, timeout_s=90)`

Polls `GET /api/queue?history=50` every 2 s (plus `GET /api/progress`
for the live counts). Returns as soon as the item is in `finished[]`:
`{finished: true, state, word, scan_numbers, run_uids, detail, progress}`.
On timeout: `{finished: false, where: "running"|"waiting", position?, progress}`
— "still running" is a **result, not an error**; the agent calls again.
An item in none of the three lists → `not_found` (the manager's history
is bounded and clearable; the message says so).

`timeout_s` defaults to and is capped at 90 s of polling, inside a
110 s wall clock for the whole call — the OSPREY team's answer to their
Q3 (2026-10-18): Claude Code moves an interactive call to the background
2 minutes after it started, drops an HTTP server's call after 5 minutes
without traffic and runs a first-byte timer, so a wait is short and the
skill calls again. Every poll sends an MCP progress notification, which
resets the idle timer only (`waits.py`).
`scan_numbers`/`run_uids` are the keys the agent carries into
`run_analysis` and `post_logbook_entry`.

### 4.4 `run_analysis(scan_number, analyzer, day=None, confirm=None)`

1. `GET /api/run/jump/{day}?prefer={scan_number}` (today when `day` is
   omitted). `matched: false` → `not_found` — the jump route otherwise
   returns the day's newest run, which must never be analysed by mistake.
2. `POST /api/run/{uid}/analysis?analyzer={id}[&confirm={n}]`. The portal
   answers 202 with the job record; 404 `no diagnostic` → `not_found`
   (with the catalog's ids); 409 `a run is in progress` → `policy_refusal`
   with the active `job`; 400 on a destructive kind without `confirm` →
   `invalid_request` (the agent passes `confirm=scan_number` itself, the
   same typed confirmation the UI asks for; the MCP never fills it in).
3. ok: `{uid, scan_number, analyzer, job}`.

Groups are not runnable here: the portal runs one diagnostic per job. The
catalog route lists a group's members so the agent can run them in turn.

### 4.5 `wait_for_analysis(scan_number, analyzer, day=None, timeout_s=90)`

Polls `GET /api/run/{uid}/analysis` every 2 s until
`analyzers[id].job.state ∉ {queued, running}`. Returns
`{finished, state ("done"|"failed"|"no_data"|"unknown"), error, artifacts, files, log_tail (≤40 lines, log_truncated flag), progress}`.
`job: null` with files on disk (the portal restarted) is reported as
`state: "unknown"` with the files, never guessed as done.

### 4.6 `post_logbook_entry(day, body_md, operator, scan=None, figures=(), payload=None)`

1. `POST <logbook>/api/entries {day, book: "scans", scan, author: <identity string>, body_md, template: "blank", kind: "agent_analysis", status: "draft", payload}`.
   The logbook **refuses a non-note kind created as `kept`**: an agent's
   entry is a draft until a person keeps it. That is the existing
   contract and it is the right one here; the MCP exposes no keep verb.
   `payload`, when given, is the schema's `AnalysisPayload`
   (`{kind: "analysis", analyzer, metrics, figures}`) passed through as a
   dict — the logbook validates it, as the thin-client rule says.
   (`EntryCreate` therefore stays in the logbook's `routes/entries.py`;
   GeecsLogbook's CLAUDE.md "deferred decisions" row that planned to move
   it beside `LogEntry` "for GEECS-MCP to validate" is updated in PR F to
   record this: the MCP validates nothing.)
2. For each figure — a portal artifact reference
   `{scan_number, day?, path, caption?}` — resolve `uid` with the §4.4
   jump (once per distinct scan, `matched` checked), then one call to
   the **portal**:
   `POST <portal>/api/run/{uid}/logbook {entry: <id>, path, caption, author: <identity string>}`.
   That route exists today for the Plot tab (`PlotToLogbook` with an
   `image` data-URL); PR B adds `path` as the alternative image source,
   the artifact fetched server-side with the same containment as
   `/run/{uid}/artifact`, and adds the attachment `link` to the route's
   reply (today `{entry_id, appended, url}`). Its gate — the run must be
   this experiment's, with a day and a scan number — refuses with 404,
   which is `not_found` here. The portal's `logbook_send` already owns the
   upload → re-read → PATCH-append → retry-on-409 sequence
   (`geecs_portal/logbook_send.py`, the logbook's one HTTP write client
   per its CLAUDE.md), so the MCP does not grow a second copy and never
   moves image bytes: nothing flows through the MCP host or model
   context.
3. ok: `{entry_id, url: "<logbook>/entry/<id>", day, scan, kind, status, author, attachments: [{path, scan_number, attached, link, appended} | {path, scan_number, attached: false, error_kind, message}], complete}`.

Order is the recovery rule (built in PR F). The entry's `scan` and every
figure's scan are resolved, and every figure path is found among that
run's listed images, **before** step 1, so a bad reference refuses the
call with nothing written. The sends carry `entry_required: true` (a
portal field added in PR F): a missing entry is the portal's 404, never
the Plot tab's fresh `note` — which the logbook would create `kept`, one
per figure, if the portal's logbook were not the MCP's. Once the entry
exists the call is ok whatever the figures do: a failed send is a row
with the service's kind and words, `complete: false` plus a `note` say
so, and the `entry_id` is never lost — posting again would make a second
entry. A figure's `day` defaults to the **entry's** day, not the MCP
host's today.

The script behind a quoted number (the approach page's provenance
habit) goes **inline in `body_md`** as a fenced block: the attachment
route accepts images and PDFs only.

### 4.7 Kept verbs, rewired

- `scan_status` → `GET /api/status` verbatim (`connected`, `re_state`,
  `items_in_queue`, `running_item_uid`, `readiness`, `identity`,
  `tiled_writer`) plus `/health.version`.
- `scan_progress` → `GET /api/progress` verbatim (`scan_number`,
  `planned_total`, `shots_done`, `state`, `exit_status`, `paused_reason`,
  `scan_folder`, `day`). The SSE stream is not consumed; the JSON routes
  are what the scanner page itself polls.
- `scan_history(limit=10)` → `GET /api/queue?history=limit`: `running`,
  `waiting`, `finished` rows (`item_uid`, `scan_numbers`, `run_uids`,
  `state`/`word`, `user`, `detail`).
- `list_scan_configs(kind, name=None)` → the names (+ `unavailable`
  reasons); with `name` on `presets`, the document.
- `get_scan_analysis(scan_number, day=None)` → jump → the analysis
  listing: per analyzer `id`, `device`, `applicable`, `job`, `files`,
  `artifacts`; the MCP bounds `job.log` to 40 lines.
- `get_scan_figure(scan_number, label=None, day=None, thumbnail=False)`
  → the `artifacts` with `kind == "summary"` or inline image suffixes;
  `figure_url` is the **portal's** artifact URL; `thumbnail=true` fetches
  it and returns a ≤768 px JPEG (the 0.6.0 payload doctrine, kept).
- `stop_scan(force=False, operator)` etc. → the scanner's verbs. `force`
  and `operator` pass through; the scanner records them and today acts
  regardless (its ownership registry is a later scanner PR). The MCP's
  own ownership check goes with `client_identity`: the queue item's
  `user` is now the scanner's identity for every submitter.

### 4.8 Two portal additions (PR B)

The portal has no analyzer catalog independent of a scan (the Analysis
tab lists per run; the editor's `/configs/api/list` exists only behind
`--config-editor`, a write surface). A small read route returning
`{analyzers: [{id, device, output_dir, destructive, kind}], groups: [{id, members}]}`
from the same processing tree the portal already loads
(`PortalState.processing_infos()` + the ScanAnalysis config store the
analysis extra brings) serves `list_analyzers` / `list_analysis_groups`
and is useful to the tab too. The second is the `path` image source on
`POST /api/run/{uid}/logbook` (§4.6). One portal PR, ~100 lines with
tests.

### 4.9 Reading numbers: `get_analysis_artifact` and `get_scan_scalars`

The analysis listing (§4.7) returns job status and artifact descriptors
only — no measured values — and `get_scan_figure` returns images. The
loop decides on numbers, so two bounded reads cover them; both are what
the PMQ campaigns read by hand.

- `get_analysis_artifact(scan_number, path, day=None, max_bytes=65536)`
  → jump → `GET /run/{uid}/artifact?path=`. For a non-image artifact
  (the analyzer's `*_summary_*.json` sidecars, per-shot `.txt`/`.csv`,
  anything `kind: summary|other` in the listing): if the body parses as
  JSON it is returned as `data` (an object, not a string); otherwise as
  `text`. The read is capped at `max_bytes` (server ceiling 256 KiB)
  with `truncated: true` and `size_bytes` when cut, so a large per-shot
  table never floods the context. An inline-image suffix is refused
  (`invalid_request`, "use get_scan_figure"); a path outside the scan's
  analysis folder is the portal's 404 → `not_found`. ok:
  `{uid, scan_number, path, content_type, size_bytes, truncated, data | text}`.
- `get_scan_scalars(scan_number, columns, day=None, x="", bincfg="", filters="")`
  → jump → `GET /api/run/{uid}/binned?cols=…&x=&bincfg=&filters=`: the
  portal's own data contract for per-bin centres and error bands over
  the union frame (Tiled event table + the s-file's appended columns),
  at most 4 columns per call (the portal's cap, 400 above it →
  `invalid_request`; an unknown column is its 404 → `not_found`). The
  MCP returns the numeric keys verbatim (`bins`, `x_centers`, the
  per-column centres/bands/counts) and **drops `figure` and `code`** —
  the plot and the notebook snippet are for browsers. `bincfg` and
  `filters` are the portal's JSON strings, passed through unparsed, so
  a binning an operator set up in the Plot tab is reproducible from the
  URL. With no `bincfg` the portal's default binning applies (the scan
  variable's steps on a stepped scan).

Neither read interprets a column or a file: column semantics stay in
`geecs_data_utils.tiled_schema` behind the portal, and the sidecar's
meaning is the recipe's.

## 5. Error envelope

Unchanged shape: `{"ok": true, ...}` or
`{"ok": false, "error_kind", "message", ...extra}`; strict JSON;
service text preserved verbatim in `message`. Kinds:

| kind | produced by |
|---|---|
| `invalid_request` | scanner `error.kind == invalid_request` (400), FastAPI 422 bodies, portal/logbook 400/422, the logbook's 413/415 on attachments (relayed through the portal send), argument checks (empty operator) |
| `not_found` | scanner 404; portal 404 (`run not found`, `no diagnostic`, `no such artifact`, jump `matched: false`); logbook 404 |
| `policy_refusal` | scanner 409 (`needs_acknowledgement`, `pending_items`, `re_state`, `items_in_queue`), the verbs' 200 `{ok:false}` (`nothing to stop (RE state: …)`, and the stop budget's `pause did not land within 120 s — scan still running`), portal 409 (run in progress), logbook 409 (version) |
| `service_unreachable` | **new** — connect/read failure or 5xx from any of the three; `extra.service ∈ {scanner, portal, logbook}` and `extra.url`. Replaces `manager_unreachable` and `tiled_unreachable` (the scanner's 503 `manager_unreachable` maps here with `service: "scanner"` and the scanner's message) |
| `internal_error` | the tools-never-raise backstop |

Dropped: `manager_unreachable`, `worker_refused`, `tiled_unreachable`,
`task_timeout` (no producers after the rebuild — the scanner maps the
kind to 504 but nothing raises it, and its stop budget expires as a 200
`{ok:false}`; the wait verbs' timeouts are **ok results** with
`finished: false`). The CHANGELOG names them.

HTTP budgets (httpx per client): connect 5 s; read 20 s, except
`POST /api/stop` 100 s (then `landed: false`, §4.7) (the scanner blocks up to 120 s for the pause)
and artifact fetches 60 s.

## 6. Identity

One string, composed by the MCP from the `operator` argument every
write verb requires: `agent, admin supervision (<operator>)`. An empty
operator is `invalid_request`. Where it lands:

| Service | Field | Note |
|---|---|---|
| scanner submit | `SubmitIn.operator` → `md.geecs.operator` on the run | `SubmissionRecord.client` stays the scanner's own identity (`geecs-scanner <version>`) — the record names the service that queued it; the operator field names who |
| scanner move / verbs | `MoveIn.operator`, `VerbIn.operator` | `ScannerService.move` never reads `body.operator` today (the verbs log it only on `force`); a two-line scanner change stamps `md.geecs.operator` on idle-only items (move, action, calibration) — proposed in §9 |
| logbook | `author` on create; `author` on each portal send (its `editor` on the append) | ≤120 chars; the entry's `kind: agent_analysis` + `status: draft` is the machine-readable half |

`operator` is an argument rather than a config value because it names a
person per shift, not a deployment; the skill supplies it from the
conversation (htu-assistant's call).

## 7. Contract tests

Hermetic and in-process: the three FastAPI apps are built from their own
factories and mounted behind httpx `ASGITransport`s; a tiny routing
transport dispatches by host (`scanner`, `portal`, `logbook`) so the
tools' base URLs resolve without sockets. The seam is
`clients.transport_for(service)`, patched in `conftest.py`.

| Service | Fixture | What is real |
|---|---|---|
| scanner | `create_app(ScannerService(DemoQueueClient(streams, period=0, user=...), DemoResolver(), preflight=demo_preflight, settables=DemoSettables(), readback=DemoReadback(...)))` (GeecsScanner's own `tests/conftest.py` shape) | routes, models, preflight flow (the demo preflight always asks `gateway_liveness` — the pass-through path is exercised by construction), the `mv` idle-only rule, queue history rows with `scan_numbers`/`run_uids`; the test advances scans with `manager.step()` between the wait verb's polls (the sleep is a patchable module attribute) |
| portal | `create_app(FakeCatalog(), processing_config_dir=tmp configs tree, analysis_factory=fake)` (the portal's `tests/test_analysis_runs.py` shape) | jump/run/analysis/artifact routes, the 202/409/400 ladder, the job record states, artifact containment |
| logbook | `create_app("Undulator", base_directory=tmp share, notes_db=tmp sqlite)` | the store, the draft rule, attachments, versioning, the mirror. The figure sends run end to end because PR B gives the portal's `create_app` a `logbook_client` injection (today `send_plot` builds its own socket `httpx.Client` unless one is passed): the fixture hands it a `starlette.testclient.TestClient` over this in-process logbook app (an `httpx.Client` subclass that routes absolute URLs to the app; `send_plot` drives a sync client, and `ASGITransport` is async-only), with `logbook_url` set |

Each contract test calls the **tool impl** and asserts on the envelope,
so a route or model change in any service fails `GEECS-MCP/tests` in the
same PR (`ci_select.py` adds the leg via the dev path deps). The portal
fixture needs the portal's `analysis` extra after all (amended in PR E):
a fake `analysis_factory` satisfies the run gate, but the listing and
the catalog are gated on the analysis core being importable, so the dev
dependency declares the extra. Registration test: every `tool_names` entry is on the
server and nothing else is.

## 8. Open points for Sam (short)

1. `get_scan_result` is absent from the keep list — dropped; its
   numeric reads come back as `get_analysis_artifact` + `get_scan_scalars`
   (§4.9, review finding on this PR). Two verb names added to the
   contract — the htu-assistant session has been told.
2. `list_analysis_groups`: keep via the new portal catalog route (§4.8),
   or drop since the portal runs single diagnostics?
3. `operator` as a required argument on every write verb (§6) rather
   than a per-deployment config value.
4. ~~Wait-verb default 240 s / cap 600 s until OSPREY answers their Q3.~~
   Answered 2026-10-18: 90 s of polling inside 110 s per call, progress notifications (§4.3).
5. Logbook figures = portal artifact references only; no inline bytes.
6. The scanner's move route does not stamp the operator on the item
   (§6) — include the two-line change in this arc?

7. **Identity, per the OSPREY team (2026-10-18):** a name should arrive
   through osprey's own door (`OSPREY_CONTROL_OWNER`, set by the terminal
   proxy), not as a string this server composes from an argument. For a
   central HTTP server that variable lives in the agent's environment,
   not the server's, and the `mcp_servers` loader keeps no `headers:`
   key to forward it — so the route (a header osprey forwards, or
   geecs-mcp over stdio per agent) is open, with them.

## 9. PR sequence (each small, each useful alone; /land ritual)

| # | Package | Content |
|---|---|---|
| A | GEECS-MCP | this note (`DESIGN.md`) |
| B | GEECS-DataPortal | `GET /api/analysis/catalog`; `path` artifact source on `POST /api/run/{uid}/logbook` (+ tests) |
| C | GeecsScanner | stamp `md.geecs.operator` on idle-only items (if §8.6 is yes) |
| D | GEECS-MCP | the client core: `config.py`, `clients.py`, `errors.py` kinds, scans domain over the scanner (kept verbs + `submit_preset`, `move`, `wait_for_scan`), `progress_stream.py` deleted, scanner contract tests. **Dependencies, `runtime.py` and the old analysis modules stay**: today's `analysis/read_tools.py` and `run_tools.py` import `runtime`, `geecs_data_utils.analysis_status` and `_run_guarded` from `scans/read_tools.py`, and `create_server()` imports both unconditionally — so D moves `_run_guarded` to a small `geecs_mcp/guard.py` and repoints those two imports, and the server starts with the scans domain on HTTP and the analysis domain unchanged. The import contract is **not** enabled yet. CLAUDE.md gets the scans half |
| E | GEECS-MCP | analysis domain over the portal (`get_scan_analysis`, `get_scan_figure`, `get_analysis_artifact`, `get_scan_scalars`, `list_*`, `run_analysis`, `wait_for_analysis`), then the deletions that D could not make: `run_tools.py`, `run_worker.py`, `/figures/`, `runtime.py`, `guard.py`, the `analysis-run` extra, the `geecs-bluesky` / `geecs-data-utils` / `scananalysis` / `imageanalysis` / `pyyaml` dependencies; the `mcp-is-a-client` import contract switched on; portal contract tests; CLAUDE.md/README rewrite finished |
| F | GEECS-MCP | logbook domain (`post_logbook_entry`) + logbook contract tests (the logbook app in process, reached by the MCP and by the portal's send); GeecsLogbook CLAUDE.md deferred-decisions row updated (§4.6) |

D and E are each useful alone: after D the agent can submit, move and
wait over HTTP while analysis still runs through the old path; after E
the package is the thin client of §2.
| G | deploy + docs | unit template (no extra; `After=` the three services; `MemoryMax` stays 1G — the planned 512M could not hold one thumbnail decode at the 80 Mpx cap, review of #1109), `bootstrap_host.sh` (`extras_of mcp` empty), `site.env.example` + `bootstrap_host.sh` rendering `[mcp] scanner_url/portal_url/logbook_url` from one `GEECS_SERVICES_HOST` key (ports are fleet constants, the archiver's `17665` precedent; `site.env.example` already cites the key in a comment without defining it — G defines it), `DEPLOYMENT.md`, `fleet_map.md`, `docs/geecs_mcp/*`, `agentic/index.md` ("no submit verb" prose goes) — then the host install (Sam's sudo block), kill the nohup process, live check from an osprey machine |

Version bump (`0.9.3 → 0.10.0`, or whatever the fleet rule says at the
time) happens in G, at the deploy, per the release policy. The
htu-assistant session edits `profile.yml` only after D lands and the
names in `tool_names.py` are final.

## 10. Configuration

`~/.config/geecs_python_api/config.ini` of the service account:

```ini
[mcp]
scanner_url = http://<services host>:8300
portal_url  = http://<services host>:8200
logbook_url = http://<services host>:8400
```

Rendered by `bootstrap_host.sh` from `site.env`'s `GEECS_SERVICES_HOST`
(the one new install-time key; on the reference deployment it equals
`GEECS_QSERVER_HOST`). An unset URL makes that service's verbs answer
`service_unreachable` with a message naming the key; the server always
starts. `[Experiment]`, `[qserver]`, `[tiled]` and `[Paths]` are no
longer read by this package.
