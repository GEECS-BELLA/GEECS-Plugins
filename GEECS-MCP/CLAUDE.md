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
- **Identity.**  Every write verb takes `operator` (the supervising
  person's name) and sends `agent, admin supervision (<operator>)`
  (`identity.py`) — the scanner records it on the run; the logbook will
  record it as the author.  The halt family records it when given and
  is never blocked on a missing or malformed name.
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
  from `site.env` by `deploy/bootstrap_host.sh` in PR G).  An unset URL
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
tests/
  conftest.py                the scanner's and the portal's own apps, in process
  test_scanner_contract.py   every scans verb against the scanner's real routes
  test_portal_contract.py    every analysis verb against the portal's real routes
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
- The wait verbs' timeouts are **results** (`finished: false`), never
  errors; the budget is `waits.MAX_WAIT_S` (600 s) pending the OSPREY
  team's word.
- Payloads are context-sized: the scanner's shapes are small; the
  portal's job log is cut to a 40-line tail; `get_analysis_artifact` is
  byte-capped (64 KiB default, 256 KiB ceiling), refuses images and
  binary files, and image bytes ride model context only through
  `get_scan_figure(thumbnail=true)`'s ≤768 px JPEG — the 0.6.0 lesson (a
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
| `stop_scan`, `pause_scan` | S | scanner |

`post_logbook_entry` (W, logbook + portal) arrives with PR F.  Retired
by the rebuild: `get_scan_result` (→ the two numeric reads),
`run_scan_analysis` (→ `run_analysis`; the server runs no analysis of its
own and spawns no worker), the `/figures/` route (→ the portal's artifact
URL), the progress-stream consumer, `runtime.py`, `[mcp] client_identity`,
the `analysis-run` extra and every intra-repo runtime dependency.

## Testing

`poetry install --with dev`, `poetry run pytest`.  Hermetic and
in-process: the scanner's app is built from `geecs_scanner`'s demo
manager (`period=0`, stepped by hand) and the portal's from
`geecs_portal.app.create_app` over a fake catalog, a temp scan tree and
an injected fake analyzer factory — the shape of the portal's own tests,
copied into `tests/conftest.py` because the subpackages' identically
named `tests` trees cannot be imported across packages (a
`geecs_portal.testing` module would be the one-copy home; a portal-side
change).  Both apps sit on `httpx.ASGITransport`s behind
`clients.transport_for`.  A route or model change in either service
fails this suite in the same PR (`scripts/ci_select.py` adds the leg
through the dev path dependencies).  The portal fixture needs the
portal's `analysis` extra installed (its analysis routes are gated on
the analysis core even with a fake factory); the dev dependency declares
it.
