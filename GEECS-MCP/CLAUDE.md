# GEECS-MCP — Developer Context for Claude

The **GEECS facility MCP server** — the one entry point for every write an
AI agent (OSPREY's htu-assistant) makes to the machine, and the reads a
closed loop needs: submit a scan, wait for it, run analysis, read the
numbers, post to the logbook.  **A thin HTTP client of three services**
and nothing else: the web scanner (GeecsScanner, :8300), the Data Portal
(:8200) and the logbook (GeecsLogbook, :8400).  The design is
`DESIGN.md` (merged #1099) — read it first; this file is the package as
built.

## Boundaries (load-bearing)

- **No engine, no seams, no re-implementation.**  The server imports
  nothing from the repo at runtime — pinned by the `mcp-is-a-client`
  contract in `.importlinter` — and speaks the services' HTTP APIs
  through `httpx`.  Preflight, limits, ownership, config resolution and
  the analysis job queue stay where an operator already sees them — the
  scanner page and the portal's Analysis tab — so what the agent does is
  what an operator does.  The services appear in `pyproject.toml` only as
  **dev** path dependencies, for the in-process contract tests.  A rule
  a service enforces is not restated here: a column cap, a range limit,
  an ownership check come back as that service's refusal.
- **The agent never acknowledges a preflight question itself.**
  `submit_preset` returns the scanner's questions verbatim under
  `needs_acknowledgement`; the skill shows them to the supervising person
  and passes the `check` names back.  Nothing here fills `acknowledged`
  in.  Likewise a destructive analysis kind's `confirm` (the typed scan
  number) is passed through, never filled in.
- **Identity — a record, not an authorization.**  `submit_preset`,
  `move`, `resume_scan` and `post_logbook_entry` require `operator` (the
  supervising person's name) and send `agent, admin supervision
  (<operator>)` (`identity.py`): the scanner records a submission's on
  the run (`md.geecs.operator`; its move does not yet stamp the queued
  item), the logbook records it as the entry's author.  The halt family
  records it when given and is never blocked on a missing or malformed
  name.  `clear_queue` and `run_analysis` carry none today (the portal's
  analysis route has no field for one).  The string records a claim;
  who may call a verb is the profile's permission lists and osprey's
  hooks (`deploy/DEPLOYMENT.md`).
- **A scan is named by scan number and day** (today when omitted); the
  portal's `jump` route turns that into the run uid, and a number the day
  does not hold is `not_found` — the jump route otherwise answers the
  day's newest run, which must never be analysed by mistake.
- **No osprey imports.**  The integration surface is `profile.yml` + HTTP
  (or stdio for the dev loop); `tool_names.py` exists so permission lists
  import symbols, not strings — its import path is stable and the HTU
  profile derives `allow` / `ask` / `write_tools` from its three tuples.
- **Configuration is ONLY `~/.config/geecs_python_api/config.ini`**:
  `[mcp] scanner_url / portal_url / logbook_url` (`config.py`; rendered
  from `site.env`'s `GEECS_SERVICES_HOST` by `deploy/bootstrap_host.sh`).  An unset URL
  makes that service's verbs answer `service_unreachable` naming the key;
  the server always starts.  `deploy/DEPLOYMENT.md` is the one full key
  inventory.

## Layout

```
geecs_mcp/
  server.py          module-level FastMCP("geecs") + create_server()
  __main__.py        python -m geecs_mcp → stdio (default) or --transport http
  tool_names.py      THE tool-name constants and the three tuples
                     (READ_TOOLS / WRITE_TOOLS / STOP_TOOLS) the profile imports
  config.py          the three service URLs from config.ini [mcp]
  identity.py        identity_for(operator) → "agent, admin supervision (<operator>)";
                     InvalidArgument, the one exception the guard maps to
                     invalid_request (every local argument check raises it)
  clients.py         one httpx.AsyncClient per service; call() → JSON or
                     ServiceRefusal (a 4xx: the service's kind + words + the
                     context it sent, minus the envelope's own keys) /
                     ServiceUnavailable (no answer, 5xx, a non-JSON 200);
                     get_bytes() the capped streamed read; transport_for()
                     is the test seam
  guard.py           run_async(): refusals / outages / argument errors /
                     bugs → envelopes; every tool wrapper goes through it
                     (DESIGN §9 planned its deletion; the async half stays)
  waits.py           MAX_WAIT_S and bounded_wait(): the one budget rule for
                     both domains' wait verbs
  errors.py          make_ok / make_error; the five kinds (module docstring)
  scans/tools.py     the scans domain over the scanner's HTTP API
  analysis/tools.py  the analysis domain over the portal's HTTP API
  portal.py          naming a scan to the portal: resolve_uid (the jump
                     route, `matched` checked), the analysis listing, the
                     figure rule — shared by the analysis and logbook domains
  logbook/tools.py   post_logbook_entry: the logbook's create + the portal's
                     figure send
tests/
  conftest.py                the scanner's, the portal's and the logbook's own apps, in process
  test_scanner_contract.py   every scans verb against the scanner's real routes
  test_portal_contract.py    every analysis verb against the portal's real routes
  test_logbook_contract.py   post_logbook_entry against the logbook and the portal's send
  test_registration.py       the tool groups partition the registered set; identity; the envelope
```

## Conventions

- Every tool is `async def` and returns a JSON string envelope through
  `guard.run_async(impl, …)`; the `_impl` sibling is the tested surface
  and may raise `ServiceRefusal` / `ServiceUnavailable` /
  `InvalidArgument` — the guard maps them.  Tools never raise to the
  agent; any other exception is `internal_error`.
- The services' words are preserved verbatim in `message`; the kinds are
  the scanner's own vocabulary (`policy_refusal`, `invalid_request`,
  `not_found`) plus `service_unreachable` and `internal_error`; the
  portal's and the logbook's status codes map onto them in `clients.py`.
- No tool re-derives a service's rule.  `move` reads back with a
  `Device:Variable` or the scan-variable listing's `target` (the scanner's
  readback route takes only the canonical form); `wait_for_scan`
  correlates on `item_uid` through `/api/queue` and attaches
  `/api/progress` only when the snapshot is that item's (another scan's
  comes as `live_scan`); `run_analysis` is the portal's job and
  `wait_for_analysis` polls the portal's listing; `get_scan_scalars`
  forwards the column list and lets the portal cap it.
- **A write is reported once it has happened.**  `post_logbook_entry`
  resolves the entry's `scan` and every figure's run, and finds every
  figure path among that run's listed images, *before* the entry is
  created (a bad one refuses the call with nothing written); once the
  entry exists the result is ok whatever the figures do — each one's
  outcome is a row in `attachments`, `complete: false` says some failed,
  and the `entry_id` is never lost (a retry would be a second entry).
  Its sends ask the portal for `entry_required`, so a draft that has
  gone — or a portal pointed at another logbook — is a refused figure,
  never a stand-in `note` born `kept`.
  The entry is an `agent_analysis` **draft**: the logbook refuses a
  non-note kind born `kept`, and no verb here keeps one.
- The wait verbs' timeouts are **results** (`finished: false`), never
  errors; the budget is `waits.MAX_WAIT_S` (600 s) pending the OSPREY
  team's word.
- Payloads are context-sized: the scanner's shapes are small; the
  portal's job log is cut to a 40-line tail; `get_analysis_artifact` is
  byte-capped (64 KiB default, 256 KiB ceiling), refuses images and
  binary files, and image bytes ride model context only through
  `get_scan_figure(thumbnail=true)`'s ≤768 px JPEG, returned as an MCP
  **image** content block beside the text reference (never base64 inside
  the JSON, which reaches the model as thousands of characters of text;
  the wrapper carries no return annotation for that reason) — the 0.6.0 lesson (a
  247 KB inline PNG once blew a haiku-tier agent's context).  Figures are
  otherwise **references**: the portal's own artifact URL, for a client
  to fetch and save.

## Verb surface

| Verb | Class | Behind |
|---|---|---|
| `scan_status`, `scan_progress`, `scan_history`, `list_scan_configs`, `wait_for_scan` | R | scanner |
| `get_scan_analysis`, `get_scan_figure`, `get_analysis_artifact`, `get_scan_scalars`, `list_analyzers`, `list_analysis_groups`, `wait_for_analysis` | R | portal |
| `submit_preset`, `move`, `resume_scan`, `clear_queue` | W | scanner |
| `run_analysis` | W | portal (`POST /api/run/{uid}/analysis`: one document on one scan, one job per scan at a time) |
| `post_logbook_entry` | W | logbook (`POST /api/entries`, a draft in the scans book) + portal (`POST /api/run/{uid}/logbook` per figure, by artifact path — no image bytes pass through here) |
| `stop_scan`, `pause_scan` | S | scanner |

Retired by the rebuild: `get_scan_result` (→ the two numeric reads),
`run_scan_analysis` (→ `run_analysis`; the server runs no analysis of its
own and spawns no worker), the `/figures/` route (→ the portal's artifact
URL), the progress-stream consumer, `runtime.py`, `[mcp] client_identity`,
the `analysis-run` extra and every intra-repo runtime dependency.

## Testing

`poetry install --with dev`, `poetry run pytest`.  Hermetic and
in-process: the scanner's app is built from `geecs_scanner`'s demo
manager (`period=0`, stepped by hand) and the portal's from
`geecs_portal.app.create_app` over a fake catalog, a temp scan tree and
an injected fake analyzer factory, and the logbook's from
`geecs_logbook.app.create_app` over the same temp share and a temp
SQLite store — the shape of each service's own tests,
copied into `tests/conftest.py` because the subpackages' identically
named `tests` trees cannot be imported across packages (a
`geecs_portal.testing` module would be the one-copy home; a portal-side
change).  All three apps sit on `httpx.ASGITransport`s behind
`clients.transport_for`, and the portal's figure send reaches the same
logbook through a `starlette` `TestClient` (`create_app(logbook_client=)`;
the send drives a sync client).  A route or model change in any service
fails this suite in the same PR (`scripts/ci_select.py` adds the leg
through the dev path dependencies).  The portal fixture needs the
portal's `analysis` extra installed (its analysis routes are gated on
the analysis core even with a fake factory); the dev dependency declares
it.
