# Leveraging OSPREY

[OSPREY](https://github.com/als-apg/osprey) is an agent framework
for accelerators developed at the ALS: it hosts an operator-facing AI
assistant, gives it facility tools (EPICS channel access, archiver
lookups, logbooks), and enforces per-tool permissions around everything
the assistant does. BELLA runs an OSPREY-based assistant, and the GEECS
MCP server is how that assistant reaches GEECS.

## The division of labour

The integration splits along the
[architecture](overview.md#where-it-sits-in-the-architecture):

- **Raw EPICS channels — OSPREY's own tools.** BELLA is an EPICS
  facility as far as OSPREY is concerned, via the
  [GEECS Gateway](../geecs_gateway/client_overview.md)'s PVs. The
  assistant reads any served channel with the framework's generic tools,
  and can set individual setpoints by writing the `:SP` PVs through
  OSPREY's own write tool — bounded by a limits database (extracted from
  the GEECS device limits) and OSPREY's own permission gating.
- **GEECS-semantic operations — this MCP server.** Scans from presets,
  single moves through the scanner, analysis on the portal, numbers,
  figures, logbook drafts: anything that needs the GEECS vocabulary goes
  through the server's verbs, and through the same services an operator
  uses. The raw `:SP` path stays a simple bounded knob — it bypasses the
  scanner's preflight and the GEECS client-side hardening (put-failure
  visibility, confirm/pseudo semantics, mid-scan refusals), so the
  richer the operation, the more it belongs behind an MCP verb.

The GEECS MCP server never imports OSPREY and OSPREY never imports GEECS
code — the entire integration surface is configuration.

## How the connection works

OSPREY registers the server as a custom MCP server in its build profile
(`profile.yml`): either a command that launches it over stdio, or the URL
of the central HTTP service (`:8100/mcp`) — one server process on the lab
server that every OSPREY machine shares. The profile's permission lists
are derived from the server's `geecs_mcp/tool_names.py` tuples
(`READ_TOOLS`, `WRITE_TOOLS`, `STOP_TOOLS`), never retyped. The exact
profile shape and the verified gating semantics live in
`GEECS-MCP/deploy/DEPLOYMENT.md`; in outline:

- **Every `WRITE_TOOLS` verb in `control_system.write_tools`.** That one
  list is what OSPREY's headless floor, its writes-check hook and its
  audit clamp read. The halt family is deliberately omitted, so a halt
  is never blocked on any path.
- **Interactive sessions, by tier.** Reads and the halt family sit on
  *allow*. The write verbs are *denied* in the base profile and the
  deny is removed at the admin tier only (lists union across tiers, so
  a tier cannot shrink *allow*); never in dispatch or a headless runner.
- **No OSPREY hooks attach to this server yet.** A profile-declared
  server cannot be put behind OSPREY's write-check or approval hooks
  today, and the writes-off kill switch does not cover it; a facility
  PreToolUse hook can make the kill switch bite until OSPREY's own fix
  lands.
- **One write route to the queue.** If the deployment also enables
  OSPREY's bundled `bluesky` server against the same queue, deny its
  queueing verbs at every tier and keep its reads and halts, so scans go
  through this server's `submit_preset` and the scanner's preflight
  (`GEECS-MCP/deploy/DEPLOYMENT.md`, "The profile and the gates").

**Identity is a record, not an authorization.** The write verbs that
take `operator` — `submit_preset`, `move`, `resume_scan`,
`post_logbook_entry` — send `agent, admin supervision (<operator>)`: the
scanner records it on a submitted run, the logbook as the entry's
author. (The scanner's move does not yet stamp it on the queued item;
`clear_queue` and `run_analysis` carry no operator today.) The string
says who the agent reports as supervising; it proves nothing. What the
assistant may call is decided by OSPREY's permission lists, never by
this server. (OSPREY would rather the name came from its own door — the
terminal proxy's `OSPREY_CONTROL_OWNER` — than from an argument; that
change is open.)

## What an operator can ask

With the server connected, the assistant can carry conversations like:

- *"Is anything running right now? What's in the queue?"* —
  `scan_status`, `scan_history`.
- *"Run the Amp4In preset."* — `submit_preset`, after `list_scan_configs`
  for the real name. The scanner's preflight questions come back to the
  operator word for word; the agent submits only once they are answered.
- *"How is it going?"* — `wait_for_scan`, or `scan_progress` for the live
  shot counts.
- *"Move the screen to 12 mm and tell me where it ended up."* — `move`,
  idle-only, with the readback.
- *"Run the standard analysis on it. What's the centroid?"* —
  `run_analysis`, `wait_for_analysis`, then `get_analysis_artifact` or
  `get_scan_scalars` for the numbers and `get_scan_figure` for the plot.
- *"Write it up in the logbook."* — `post_logbook_entry`: a draft on the
  scan's card, with the figures attached, for a person to keep.
- *"Stop the scan."* — `stop_scan`, gracefully, always available.

Each step is an auditable tool call with its own gate, against the same
scanner page, Analysis tab and logbook an operator watches — which is
exactly the trade the whole design makes: flexible language in front, a
fixed and reviewable machine surface behind.
