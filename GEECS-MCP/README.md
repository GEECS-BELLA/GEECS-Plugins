# GEECS-MCP

The general GEECS MCP server — AI-agent access to GEECS, one server with
domains as modules. The first domain is **scans**: the
bluesky-queueserver RE Manager, the configs repo, and the Tiled archive.
Future domains (health, DB metadata, log triage, analysis) register on
the same server.

**v0 + v1 (current).** Read-only: `scan_status`, `scan_history`,
`get_scan_result`, `list_scan_configs`, `validate_scan_request`,
`scan_progress`, `get_scan_analysis`, `get_scan_figure` (analysis
results + rendered figures — needs the data share mounted). Control (put these under OSPREY `ask`; list them in `config:` `write_tools` for headless):
`submit_scan` (one scan in flight, 1,000-shot cap, preflight warnings
need explicit acknowledgement), `stop_scan` (graceful; `force` for
another client's scan is approval territory), `clear_queue` (the one
remover). Gating semantics — what osprey actually enforces for custom
servers — are documented in `deploy/DEPLOYMENT.md` (verified: hook
presets and the interactive kill switch do NOT apply; the native ask
prompt is the interactive gate).

## Run

```bash
python -m geecs_mcp
```

A stdio MCP server (FastMCP). Configuration is the standard
`~/.config/geecs_python_api/config.ini` — `deploy/DEPLOYMENT.md` carries
the full key inventory. Anything unconfigured degrades to an honest
refusal/empty answer — the server always starts.

OSPREY integration: point a `mcp_servers:` stdio `command:` at this
module in `profile.yml`; permission lists import
`geecs_mcp.tool_names` symbols (all v0 tools are read-only →
`allow`).

## Development

`poetry install --with dev`, `poetry run pytest`. See `CLAUDE.md` for the
architecture, the verb roadmap, and the safety doctrine.

## Deployment

See `deploy/DEPLOYMENT.md`: central HTTP service on the qserver box (the
multi-machine mode — osprey machines need no GEECS install, just a URL
in `profile.yml`) or per-machine stdio (the dev loop).
