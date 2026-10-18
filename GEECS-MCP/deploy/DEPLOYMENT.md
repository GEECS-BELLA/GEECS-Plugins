# GEECS-MCP deployment

Two transports, one server.  Since 0.10.0 the server is a thin HTTP
client of three services — the web scanner (`:8300`), the Data Portal
(`:8200`) and the logbook (`:8400`) — and reaches nothing else
(`DESIGN.md`).

## Central HTTP service (the multi-machine mode — recommended)

One server runs on the services box as the `geecs-mcp` systemd unit,
`:8100/mcp`.  Every osprey machine then needs **no GEECS install at all**.

### Configuration

The service account's `~/.config/geecs_python_api/config.ini` — nothing
else is read:

```ini
[mcp]
scanner_url = http://<services host>:8300
portal_url  = http://<services host>:8200
logbook_url = http://<services host>:8400
```

`deploy/bootstrap_host.sh` renders the section from `site.env`'s
`GEECS_SERVICES_HOST` (ports are fleet constants), and appends it to an
existing `config.ini` that has none.  A file that already has a
pre-0.10 `[mcp]` section (`client_identity`, `max_shots`) is not edited:
the bootstrap prints the missing lines to add by hand, and the old keys
are ignored.  A `site.env` without the key deploys no `geecs-mcp`.  An unset key leaves that service's verbs answering
`service_unreachable` with the key's name; the server always starts.

The server needs no data share, no configs root, no Tiled key and no
analysis extra: analysis runs on the portal (`run_analysis` is the
Analysis tab's own job, under the portal's memory ceiling), figures are
sent to the logbook by the portal, and a figure reference is the
portal's artifact URL.

### The profile and the gates

The verb names and their classes come from `geecs_mcp/tool_names.py` —
`READ_TOOLS`, `WRITE_TOOLS`, `STOP_TOOLS`.  A profile derives its lists
from those tuples; never retype them.

**What gates a custom server today** (the OSPREY team's ruling,
2026-10-18, checked against their `main`).  Osprey's `writes_check` and
`approval` hooks do **not** attach to a server the profile declares: an
`mcp_servers.<name>` entry keeps `command/args/env/url/transport/port` and
`permissions.{allow,ask}`, a `hooks:` key there is dropped at load, and
`claude_code.servers.<name>.hooks` does not reach a profile-declared
server either (the build rewrites the entry wholesale).  The writes-off
kill switch's deny list covers framework servers only.  So:

- **`control_system.write_tools`** — every `WRITE_TOOLS` verb, as
  `mcp__geecs__<verb>`.  It folds into `hook_config.json`'s
  `write_tools`, which the headless disallow floor, the writes-check
  hook and the audit middleware's clamp all read.  This is the lever a
  facility server has today.
- **Tiers by `deny` and `remove_deny`.**  Lists only union across tiers,
  so a tier cannot take a verb back out of the base profile's `allow`.
  The write verbs go on `claude_code.permissions.deny` in the base
  profile and `remove_deny` in the admin persona (or a tier drops the
  whole server with `exclude: {mcp_servers: [geecs]}`), with the
  DEVIATION marker.  Interactive sessions only — never dispatch or a
  headless runner.
- **Optional, until osprey closes the gap: a facility PreToolUse hook**
  (`claude_code.hooks.PreToolUse`, `matcher: mcp__geecs__.*`, a hook file
  the facility ships — osprey's built-in hooks are refused there) that
  refuses a `WRITE_TOOLS` verb while the kill switch is thrown.  Without
  it, `writes_enabled: false` does **not** stop these verbs in an
  interactive session.
- **Coming in osprey:** `mcp_servers.<name>.hooks.pre_tool_use:
  [approval | writes_check | limits]`, and the kill-switch deny extended
  to a custom server's declared write verbs.  When it lands, this
  section moves to it.

```yaml
# profile.yml (base)
mcp_servers:
  geecs:
    url: http://192.168.6.14:8100/mcp
    transport: http
    permissions:
      allow: [<READ_TOOLS>, <STOP_TOOLS>]
      # DEVIATION 2026-10-18: write verbs allowed at admin, interactive sessions only;
      # approval lives in the conversation (goal + stopping conditions stated up front).
claude_code:
  permissions:
    deny: [<WRITE_TOOLS>]
control_system:
  write_tools: [<WRITE_TOOLS>]

# personas/admin.yml
claude_code:
  permissions:
    remove_deny: [<WRITE_TOOLS>]
```

(`<WRITE_TOOLS>` is all six: `submit_preset`, `move`, `resume_scan`,
`clear_queue`, `run_analysis`, `post_logbook_entry`, each as
`mcp__geecs__<verb>`.)

**Headless** (`osprey query`, dispatch): `write_tools` above is the
floor — a profile missing an entry leaves an unattended agent's write
ungated.  `resume_scan` is listed because it restarts motion; in
dispatch a custom server's tools are further confined to a trigger's
`allowed_tools`.

**The halt family is never gated.**  `stop_scan` and `pause_scan` are
deliberately absent from `write_tools` and sit on `allow`: a halt is
never blocked on any path.

**Names a pre-0.10 profile may still carry**: `get_scan_result`,
`run_scan_analysis` (0.10.0), and `submit_scan`, `run_action`,
`move_scan_variable`, `describe_action`, `validate_scan_request`
(0.9.0).  Drop them — a permission naming a tool the server does not
register is inert, but it misreads as a capability that exists.

### Blocking calls

Claude Code, not osprey, sets the ceilings a tool call lives under: an
interactive call still running after 2 minutes moves to a background
task, an HTTP server's call is dropped after 5 minutes without traffic
(30 minutes over stdio), and HTTP calls run a first-byte timer.  So the
wait verbs default to 90 s and cap at 110 s, answer `finished: false`
when the budget ends (the skill calls again), and every blocking call —
the waits' polls, and a halt the scanner holds open until the pause
lands — sends MCP progress notifications, which reset those timers.

### Identity is a record, not an authorization

Write verbs that take `operator` send
`agent, admin supervision (<operator>)`: `submit_preset` (the scanner
records it on the run, `md.geecs.operator`), `move` and `resume_scan`
(sent to the scanner, which does not yet stamp it on the queued item),
`post_logbook_entry` (the entry's author).  `stop_scan` / `pause_scan`
record it when given and never refuse over it.  `clear_queue` and
`run_analysis` carry no operator today (the portal's analysis route
has no field for one).

The string records who the agent says supervised; it proves nothing.
Who may call which verb is decided by the profile's permission lists
and osprey's hooks above, never by this server.

OSPREY's own rule goes further: a name arrives only through one of its
two doors (the terminal proxy or a sidecar login gate, as
`X-Osprey-Owner`), and a service reads it from `OSPREY_CONTROL_OWNER`
in the agent's environment — never from a free string.  The composed
string is the interim shape (osprey's bundled Bluesky server has it
too); taking the owner from that door instead is open, because the
variable lives in the agent's environment, not this central server's
(`DESIGN.md` § 8).

### Host setup

Same clone and ritual as the worker (`deploy/bootstrap_host.sh` does all
of this — see `docs/platform/site_profile.md`):

```bash
# as the service account; <root> = GEECS_CHECKOUT_ROOT from site.env, the
# worker's clone is <root>/qs-checkout
python3.11 -m venv <root>/geecs-mcp-venv
<root>/geecs-mcp-venv/bin/pip install "<root>/qs-checkout/GEECS-MCP"   # non-editable:
                                                                       # code bakes into the venv
<root>/qs-checkout/deploy/render_units.sh /etc/geecs/site.env <root>/deploy-staging
sudo install -m 0644 <root>/deploy-staging/geecs-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now geecs-mcp
```

**Moving a host off a hand-started server** (a `nohup python -m
geecs_mcp …` from 0.9): stop that process first — it holds `:8100` and
the unit would fail to bind — then install and enable the unit as
above, and check `ss -ltnp | grep 8100` names the unit's venv.

Update ritual: `git pull` in the checkout, re-run the pip install (the
install is non-editable by design — a pull never mutates code under the
running service), `systemctl restart geecs-mcp`.

Health check: a `scan_status` tool call from an agent.  Every verb that
reaches a service names it on failure (`service_unreachable`,
`service`, `url`): `scan_status` checks the scanner's address and
`list_analyzers` the portal's.  No read reaches the logbook; its
address is first exercised by a real `post_logbook_entry`, which is a
write — do not post a test entry to check it.

**The portal moves with this server.**  `post_logbook_entry` sends each
figure through the portal with `entry_required` (GEECS-DataPortal
0.41.0): a portal older than that ignores the field, and a figure whose
draft has gone becomes a fresh `kept` note instead of a refused row.
Deploy the portal first, or together.

Interim security posture: no transport auth, lab-network-internal —
identical to the manager's control socket; issue #660 (CurveZMQ / fleet
auth) covers the eventual answer for both.  The server serves no bytes
of its own.

## Per-machine stdio (dev loop / single host)

`profile.yml` launches the process per session:

```yaml
mcp_servers:
  geecs:
    command: /opt/geecs-mcp-venv/bin/python
    args: ["-m", "geecs_mcp"]
    permissions: { ... as above ... }
```

(The `mcp_servers` route: no hooks, as above.)  The machine running it
needs the same `[mcp]` section, pointing at the services box.

Installing from the shared-drive clone works (`pip install
<share>/GEECS-Plugins/GEECS-MCP`) — **non-editable only**: an editable
install pointing at an SMB share reads source off the share at runtime,
and share visibility blips are a documented production failure class in
this fleet.
