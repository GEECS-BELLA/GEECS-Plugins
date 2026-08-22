# GEECS-Scan-MCP

MCP server giving AI agents (the OSPREY HTU assistant) access to the GEECS
scan service: the bluesky-queueserver RE Manager, the configs repo, and
the Tiled archive.

**v0 + v1 (current).** Read-only: `scan_status`, `scan_history`,
`get_scan_result`, `list_scan_configs`, `validate_scan_request`,
`scan_progress`. Control (put these under OSPREY `ask` + hooks):
`submit_scan` (one scan in flight, 1,000-shot cap, preflight warnings
need explicit acknowledgement), `stop_scan` (graceful; `force` for
another client's scan is approval territory — and keep stop OUT of the
kill-switch hook so a halt is always possible), `clear_queue` (the one
remover).

## Run

```bash
python -m geecs_scan_mcp
```

A stdio MCP server (FastMCP). Configuration is the standard
`~/.config/geecs_python_api/config.ini`: `[Experiment] expt` (the
experiment), `[qserver] host` (the RE Manager), `[tiled] uri`/`api_key`
(the archive), and the configs-repo path (`GEECS_SCANNER_CONFIG_DIR` or
`[Paths] scanner_config_root_path`). Anything unconfigured degrades to an
honest refusal/empty answer — the server always starts.

OSPREY integration: point a `mcp_servers:` stdio `command:` at this
module in `profile.yml`; permission lists import
`geecs_scan_mcp.tool_names` symbols (all v0 tools are read-only →
`allow`).

## Development

`poetry install --with dev`, `poetry run pytest`. See `CLAUDE.md` for the
architecture, the verb roadmap, and the safety doctrine.
