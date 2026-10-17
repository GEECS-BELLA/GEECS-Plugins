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
      allow: [scan_status, scan_progress, scan_history, list_scan_configs,
              wait_for_scan, get_scan_analysis, get_scan_figure,
              get_analysis_artifact, get_scan_scalars,
              list_analyzers, list_analysis_groups, wait_for_analysis,
              stop_scan, pause_scan]
      ask:   [submit_preset, move, resume_scan, clear_queue, run_analysis,
              post_logbook_entry]
```

(The lists mirror `geecs_mcp/tool_names.py` — `READ_TOOLS` and
`STOP_TOOLS` under `allow`, `WRITE_TOOLS` under `ask` here; the HTU
deployment's posture — the write verbs on `allow` at the admin tier only,
with a DEVIATION marker — is the profile owner's call, derived from the
same tuples.  Update both together when a verb lands.)

Since the HTTP-client rebuild (DESIGN.md) the server reaches the lab
through three services, named in the service account's `config.ini`:

```ini
[mcp]
scanner_url = http://<services host>:8300
portal_url  = http://<services host>:8200
logbook_url = http://<services host>:8400
```

An unset key leaves that service's verbs answering `service_unreachable`
with the key's name; the server always starts.  `bootstrap_host.sh`
renders them from `site.env` (PR G of the arc).

**Hook/kill-switch semantics (VERIFIED against osprey, 2026-08-22)**:
a `hooks:` key on a profile-level custom-server block is **silently
ignored** — osprey hook presets do not attach to custom servers, and
the interactive writes kill switch does not cover custom-server tools
either (the framework's deny augmentation walks its own
`FRAMEWORK_SERVERS` only). The actual gates are therefore:

- **Interactive**: the native `ask` permission prompt on every control
  verb — a human sees each `submit_preset`/`move`/`clear_queue`/
  `resume_scan`/`run_analysis`/`post_logbook_entry` call with its arguments (the HTU
  deployment puts them on `allow` at the admin tier with its DEVIATION
  marker — the profile owner's call). The
  halt family (`stop_scan`,
  `pause_scan`) is NOT behind the kill switch on either path:
  interactively because the kill switch does not cover custom servers
  (upstream gap), headless by the deliberate `write_tools` omission
  below — halts are never blocked.
- **Headless** (`osprey query`): the framework reads
  `hook_config.json`'s `write_tools` (populated from the profile's
  `config:`) — list every `tool_names.WRITE_TOOLS` entry there
  (`submit_preset`, `move`, `resume_scan` — resume restarts motion, so
  it gates like a submission — `clear_queue`, `run_analysis`,
  `post_logbook_entry`) and
  **deliberately NOT the halt family** (`stop_scan`, `pause_scan`):
  exempt by omission, so a halt is never blocked on any path. E.g.
  `write_tools: [mcp__geecs__submit_preset, mcp__geecs__move,
  mcp__geecs__resume_scan, mcp__geecs__clear_queue,
  mcp__geecs__run_analysis, mcp__geecs__post_logbook_entry]`.  **A profile deployed before 0.9.0
  names removed tools** (`submit_scan`, `run_action`,
  `move_scan_variable`, `describe_action`, `validate_scan_request`):
  drop those entries — a permission naming a tool the server does not
  register is inert, but it misreads as a capability that exists.  This is the ONLY headless
  gate — a profile missing these entries leaves an unattended agent's
  writes ungated.

**Analysis runs through the portal.** `run_analysis` is the portal's
`POST /api/run/{uid}/analysis` — the Analysis tab's own run, on the
portal's worker thread, under the portal's memory ceiling — so this
server needs no analysis extra, no configs root and no data share: it
needs the portal's address.  The portal's `DEPLOYMENT.md` owns the share
and the configs tree.

Two upstream osprey issues (to be filed from the osprey side): the
silent acceptance of unknown keys like `hooks:` on custom-server
blocks, and custom servers being excluded from the interactive kill
switch. Until they land, do not add a `hooks:`
key here — it would document intent the framework does not enforce.

Host setup (same checkout + ritual as the worker):

```bash
# as the service account; <root> = GEECS_CHECKOUT_ROOT from site.env, the
# worker's clone is <root>/qs-checkout (deploy/bootstrap_host.sh does all of
# this — see docs/platform/site_profile.md)
python3.11 -m venv <root>/geecs-mcp-venv
<root>/geecs-mcp-venv/bin/pip install "<root>/qs-checkout/GEECS-MCP"   # non-editable:
                                                                       # code bakes into the venv
<root>/qs-checkout/deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
sudo install -m 0644 ~/deploy-staging/geecs-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now geecs-mcp
```

Config is the standard `~/.config/geecs_python_api/config.ini` of the
service user: `[mcp] scanner_url / portal_url / logbook_url` (above) —
nothing else is read.  `[mcp] client_identity` is no longer read:
the queue item's `user` is the scanner's identity for every submitter
and the operator rides in the run's `md.geecs.operator`.  `[mcp]
max_shots` is no longer read either (0.9.0).

Update ritual: `git pull` in the checkout, re-run the pip install (the
install is non-editable by design — a pull never mutates code under the
running service), `systemctl restart geecs-mcp`.

Interim security posture: no transport auth, lab-network-internal —
identical to the manager's control socket; issue #660 (CurveZMQ / fleet
auth) covers the eventual answer for both.  The server serves no bytes
of its own any more: a figure reference points at the portal's artifact
URL, under the portal's posture.

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
