# GEECS-MCP

The GEECS facility MCP server — the one entry point for an AI agent's
writes to the machine, built as a **thin HTTP client** of the web scanner
(:8300), the Data Portal (:8200) and the logbook (:8400). The design is
`DESIGN.md`; the rebuild lands in small PRs (its §9) and this README
describes the current state.

**Scans (over the scanner).** Reads: `scan_status`, `scan_progress`,
`scan_history`, `list_scan_configs`, `wait_for_scan` (blocks until a
queue item finishes; a timeout is a result, not an error). Writes, each
taking the supervising operator's name: `submit_preset` (the scanner's
own preflight; its questions come back verbatim and are never
acknowledged here), `move` (the scanner's `mv` plan, idle-only, with the
readback), `resume_scan`, `clear_queue`. The halt family — `stop_scan`,
`pause_scan` — is never behind the headless gate.

**Analysis (over the Data Portal).** `list_analyzers` /
`list_analysis_groups` (the portal's catalog), `run_analysis` +
`wait_for_analysis` (one document on one scan through the portal's
Analysis-tab route), `get_scan_analysis` (the per-scan picture),
`get_analysis_artifact` (a run's JSON/text sidecar, byte-capped) and
`get_scan_scalars` (per-bin statistics of up to four columns) for the
numbers, `get_scan_figure` (a reference to the portal's artifact URL;
`thumbnail=true` for a bounded JPEG as an MCP image block).

Permission lists import `geecs_mcp.tool_names` (`READ_TOOLS`,
`WRITE_TOOLS`, `STOP_TOOLS`); `deploy/DEPLOYMENT.md` has the profile
shape and the gating semantics.

## Run

```bash
python -m geecs_mcp
```

A stdio MCP server (FastMCP). Configuration is the standard
`~/.config/geecs_python_api/config.ini` — `deploy/DEPLOYMENT.md` carries
the full key inventory. Anything unconfigured degrades to an honest
refusal/empty answer — the server always starts.

OSPREY integration: the central HTTP service (`--transport http`) with a
`url:` in `profile.yml`, or a stdio `command:` for the dev loop;
permission lists import `geecs_mcp.tool_names` symbols.

## Development

`poetry install --with dev`, `poetry run pytest`. See `CLAUDE.md` for the
architecture, the verb roadmap, and the safety doctrine.

## Deployment

See `deploy/DEPLOYMENT.md`: central HTTP service on the qserver box (the
multi-machine mode — osprey machines need no GEECS install, just a URL
in `profile.yml`) or per-machine stdio (the dev loop).
