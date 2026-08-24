# GEECS-MCP deployment

Two transports, one server.

## Central HTTP service (the multi-machine mode — recommended)

One server runs on the **qserver box** (it already reaches everything
locally: the RE Manager, the CA gateway, Tiled, the DB, and — crucially
— it resolves configs from the SAME checkout the worker uses, so
client-side validation exactly matches worker truth). Every osprey
machine then needs **no GEECS install at all**:

```yaml
# profile.yml on every osprey machine
mcp_servers:
  geecs:
    url: http://192.168.6.14:8100/mcp
    transport: http
    permissions:
      allow: [scan_status, scan_history, get_scan_result,
              list_scan_configs, validate_scan_request, scan_progress,
              get_scan_analysis, get_scan_figure, describe_action]
      ask:   [submit_scan, stop_scan, clear_queue,
              run_action, move_scan_variable, pause_scan, resume_scan]
```

(The lists mirror `geecs_mcp/tool_names.py` — `READ_TOOLS` under
`allow`, `QUEUE_TOOLS` + `STOP_TOOLS` under `ask`; update both together
when a verb lands.)

**Hook/kill-switch semantics (VERIFIED against osprey, 2026-08-22)**:
a `hooks:` key on a profile-level custom-server block is **silently
ignored** — osprey hook presets do not attach to custom servers, and
the interactive writes kill switch does not cover custom-server tools
either (the framework's deny augmentation walks its own
`FRAMEWORK_SERVERS` only). The actual gates are therefore:

- **Interactive**: the native `ask` permission prompt on every control
  verb — a human sees each `submit_scan`/`stop_scan`/`clear_queue`/
  `run_action`/`move_scan_variable`/`pause_scan`/`resume_scan` call
  with its arguments (this is also the backstop for the
  acknowledge-warnings residual). The halt family (`stop_scan`,
  `pause_scan`) is NOT behind the kill switch on either path:
  interactively because the kill switch does not cover custom servers
  (upstream gap), headless by the deliberate `write_tools` omission
  below — halts are never blocked.
- **Headless** (`osprey query`): the framework reads
  `hook_config.json`'s `write_tools` (populated from the profile's
  `config:`) — list every `tool_names.QUEUE_TOOLS` entry there
  (`submit_scan`, `clear_queue`, and since 0.5.0 `run_action`,
  `move_scan_variable`, `resume_scan` — resume restarts motion, so it
  gates like a submission) and **deliberately NOT the halt family**
  (`stop_scan`, `pause_scan`): exempt by omission, so a halt is never
  blocked on any path. E.g. `write_tools: [mcp__geecs__submit_scan,
  mcp__geecs__clear_queue, mcp__geecs__run_action,
  mcp__geecs__move_scan_variable, mcp__geecs__resume_scan]` (the
  deployed htu profile predates 0.5.0 and lists the first two —
  extend it when the v2 verbs deploy).  This is the ONLY headless
  gate — a profile missing these entries leaves an unattended agent's
  writes ungated.

Two upstream osprey issues (to be filed from the osprey side): the
silent acceptance of unknown keys like `hooks:` on custom-server
blocks, and custom servers being excluded from the interactive kill
switch. Until they land, do not add a `hooks:`
key here — it would document intent the framework does not enforce.

Host setup (same checkout + ritual as the worker):

```bash
python3.11 -m venv /opt/geecs-mcp-venv
/opt/geecs-mcp-venv/bin/pip install <checkout>/GEECS-MCP   # non-editable: code
                                                            # bakes into the venv
sudo cp <checkout>/GEECS-MCP/deploy/geecs-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now geecs-mcp
```

Config is the standard `~/.config/geecs_python_api/config.ini` of the
service user: `[Experiment] expt`, `[qserver] host`, `[tiled]`, the
configs-repo path, and `[mcp] client_identity` (e.g.
`osprey-htu-assistant`) + optional `[mcp] max_shots`.

Update ritual: `git pull` in the checkout, re-run the pip install (the
install is non-editable by design — a pull never mutates code under the
running service), `systemctl restart geecs-mcp`.

Interim security posture: no transport auth, lab-network-internal —
identical to the manager's control socket; issue #660 (CurveZMQ / fleet
auth) covers the eventual answer for both.  This covers the
``/figures/{day}/{scan}/{label}`` route too (0.6.0): it serves raw
analysis-figure bytes off the data share on the same port, bounded to
each scan's own analysis folder with a 50 MB cap — same posture, same
eventual auth answer.

## Per-machine stdio (dev loop / single host)

`profile.yml` launches the process per session:

```yaml
mcp_servers:
  geecs:
    command: /opt/geecs-mcp-venv/bin/python
    args: ["-m", "geecs_mcp"]
    permissions: { ... as above ... }
```

Installing from the shared-drive clone works (`pip install
<share>/GEECS-Plugins/GEECS-MCP`) — **non-editable only**: an editable
install pointing at an SMB share reads source off the share at runtime,
and share visibility blips are a documented production failure class in
this fleet.
