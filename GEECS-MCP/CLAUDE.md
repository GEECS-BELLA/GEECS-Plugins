# GEECS-MCP — Developer Context for Claude

The **GEECS facility MCP server** — the one entry point for every write an
AI agent (OSPREY's htu-assistant) makes to the machine, and the reads a
closed loop needs: submit a scan, wait for it, run analysis, read the
numbers, post to the logbook.  **A thin HTTP client of three services**
and nothing else: the web scanner (GeecsScanner, :8300), the Data Portal
(:8200) and the logbook (GeecsLogbook, :8400).  The design is
`DESIGN.md` (merged #1099) — read it first; this file is the state of
the package between its PRs.

## Boundaries (load-bearing)

- **No engine, no seams, no re-implementation.**  The server imports
  nothing from GeecsBluesky, ScanAnalysis, GEECS-Core or the services'
  code; it speaks their HTTP APIs through `httpx`.  Preflight, limits,
  ownership, config resolution and the analysis job queue stay where an
  operator already sees them — the scanner page and the portal's
  Analysis tab — so what the agent does is what an operator does.  The
  import contract `mcp-is-a-client` (`.importlinter`) is switched on in
  PR E, once the analysis domain has moved off its old imports.
- **Rebuild in progress (DESIGN.md §9).**  PR D (this state): the
  `scans/` domain is on the scanner's HTTP API — kept verbs plus
  `submit_preset`, `move`, `wait_for_scan`; the old queue-client scans
  modules and the progress stream are deleted.  The `analysis/` domain
  is UNCHANGED until PR E (it still reads the share through
  `runtime.py` and `geecs_data_utils`, and `run_scan_analysis` still
  spawns ScanAnalysis), which is why `geecs-bluesky`, `geecs-data-utils`
  and the `analysis-run` extra are still declared.  PR F adds the
  logbook domain; PR G the deploy.
- **The agent never acknowledges a preflight question itself.**
  `submit_preset` returns the scanner's questions verbatim under
  `needs_acknowledgement`; the skill shows them to the supervising person
  and passes the `check` names back.  Nothing here fills `acknowledged`
  in.
- **Identity.**  Every write verb takes `operator` (the supervising
  person's name) and sends `agent, admin supervision (<operator>)`
  (`identity.py`) — the scanner records it on the run; the logbook will
  record it as the author.  The halt family records it when given and
  is never blocked on a missing name.
- **No osprey imports.**  The integration surface is `profile.yml` + HTTP
  (or stdio for the dev loop); `tool_names.py` exists so permission lists
  import symbols, not strings — its import path is stable and the HTU
  profile derives `allow` / `ask` / `write_tools` from its three tuples.
- **Configuration is ONLY `~/.config/geecs_python_api/config.ini`**:
  `[mcp] scanner_url / portal_url / logbook_url` (`config.py`; rendered
  from `site.env` by `deploy/bootstrap_host.sh` in PR G).  An unset URL
  makes that service's verbs answer `service_unreachable` naming the key;
  the server always starts.  `deploy/DEPLOYMENT.md` is the one full key
  inventory.

## Layout

```
geecs_mcp/
  server.py        module-level FastMCP("geecs") + create_server()
  __main__.py      python -m geecs_mcp → stdio (default) or --transport http
  tool_names.py    THE tool-name constants and the three tuples
                   (READ_TOOLS / WRITE_TOOLS / STOP_TOOLS) the profile imports
  config.py        the three service URLs from config.ini [mcp]
  identity.py      identity_for(operator) → "agent, admin supervision (<operator>)"
  clients.py       one httpx.AsyncClient per service; call() → JSON or
                   ServiceRefusal (a 4xx, the service's kind + words) /
                   ServiceUnavailable (no answer or 5xx); transport_for()
                   is the test seam
  guard.py         run_async(): refusals/outages/bugs → envelopes (every
                   async tool); run_guarded(): the sync-impl wrapper the
                   analysis domain keeps until PR E
  errors.py        make_ok / make_error; the five kinds (module docstring)
  runtime.py       config singletons the ANALYSIS domain still uses
                   (experiment, catalog, resolver) — deleted in PR E
  scans/tools.py   the scans domain over the scanner's HTTP API
  analysis/        read_tools.py (get_scan_analysis, get_scan_figure,
                   the /figures route), run_tools.py (run_scan_analysis,
                   list_analyzers, list_analysis_groups), run_worker.py —
                   unchanged until PR E
tests/
  conftest.py              the scanner's own app, in process, behind the tools
  test_scanner_contract.py every scans verb against the scanner's real routes
  test_registration.py     the tool groups partition the registered set; identity
  test_analysis_*.py       the analysis domain's suite (unchanged)
```

## Conventions

- Every tool is `async def` and returns a JSON string envelope through
  `guard.run_async(impl, …)`; the `_impl` sibling is the tested surface
  and may raise `ServiceRefusal` / `ServiceUnavailable` / `ValueError`
  (an argument check) — the guard maps them.  Tools never raise to the
  agent.
- The services' words are preserved verbatim in `message`; the kinds are
  the scanner's own vocabulary (`policy_refusal`, `invalid_request`,
  `not_found`) plus `service_unreachable` and `internal_error`.
- No tool re-derives a service's rule.  `move` reads back with a
  `Device:Variable` or the scan-variable listing's `target` (the scanner's
  readback route takes only the canonical form); `wait_for_scan`
  correlates on `item_uid` through `/api/queue`.
- The wait verbs' timeouts are **results** (`finished: false` with the
  latest progress), never errors; the budget is capped at `MAX_WAIT_S`
  (600 s, pending the OSPREY team's word).
- Payloads are context-sized: the scanner's own shapes are small; the
  portal's job log is bounded in PR E.

## Verb surface (PR D state)

| Verb | Class | Behind |
|---|---|---|
| `scan_status`, `scan_progress`, `scan_history`, `list_scan_configs`, `wait_for_scan` | R | scanner |
| `get_scan_analysis`, `get_scan_figure`, `list_analyzers`, `list_analysis_groups` | R | the share / configs tree (old path, until E) |
| `submit_preset`, `move`, `resume_scan`, `clear_queue` | W | scanner |
| `run_scan_analysis` | W | ScanAnalysis subprocess (old path; → `run_analysis` in E) |
| `stop_scan`, `pause_scan` | S | scanner |

Deleted in D: `get_scan_result` (its numeric reads return as
`get_analysis_artifact` + `get_scan_scalars` in E), the `QUEUE_TOOLS`
tuple (now `WRITE_TOOLS`), the progress-stream consumer and its HTTP-mode
warm-up, `[mcp] client_identity` (the queue item's `user` is the
scanner's identity for every submitter; the operator rides in
`md.geecs.operator`).

## Testing

`poetry install --with dev -E analysis-run`, `poetry run pytest`.  The
scans domain is tested **against the scanner's real FastAPI app**, built
from `geecs_scanner`'s demo manager (`period=0`, stepped by hand) and
mounted on an `httpx.ASGITransport` through `clients.transport_for` — a
route or model change in GeecsScanner fails this suite in the same PR
(`scripts/ci_select.py` adds the leg through the dev path dependency).
The analysis domain's tests are unchanged (fakes on `runtime`).
