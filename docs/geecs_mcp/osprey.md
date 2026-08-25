# Leveraging OSPREY

[OSPREY](https://github.com/als-computing/osprey) is an agent framework
for accelerators developed at the ALS: it hosts an operator-facing AI
assistant, gives it facility tools (EPICS channel access, archiver
lookups, logbooks), and enforces per-tool permissions around everything
the assistant does. BELLA runs an OSPREY-based assistant, and the GEECS
MCP server is how that assistant reaches GEECS.

## The division of labour

The integration deliberately splits along the
[write-surface doctrine](overview.md#where-it-sits-in-the-architecture):

- **Raw EPICS channels — OSPREY's own tools, read-only.** BELLA is an
  EPICS facility as far as OSPREY is concerned, via the
  [GEECS Gateway](../geecs_gateway/client_overview.md)'s PVs. The
  assistant can read any served channel with the framework's generic
  tools; it does not write them.
- **GEECS-semantic operations — this MCP server.** Scans, configs,
  results, analysis: everything that changes machine state or needs the
  GEECS vocabulary goes through the server's gated verbs.

The GEECS MCP server never imports OSPREY and OSPREY never imports GEECS
code — the entire integration surface is configuration.

## How the connection works

OSPREY registers the server as a *custom MCP server* in its build profile
(`profile.yml`): either a `command:` that launches the server over stdio,
or the URL of the central HTTP service — one server process on the lab
server that every OSPREY machine shares. Permission lists in the profile
import the server's `tool_names` constants rather than retyping strings,
so a renamed tool cannot silently strand a permission entry.

Gating has two layers, and the distinction matters (the semantics were
verified against OSPREY directly; details and the exact `write_tools`
list live in `GEECS-MCP/deploy/DEPLOYMENT.md`):

- **Interactive sessions**: every control verb surfaces OSPREY's native
  *ask* prompt — a human sees the tool name and its arguments and
  approves each call.
- **Headless runs**: the profile's `write_tools` allowlist is the gate.
  Every queueing verb is listed there; the stop family deliberately is
  **not**, so a halt is never blocked on any path.

Each deployment sets a client identity (for the HTU assistant:
`osprey-htu-assistant`) that is stamped onto every queue item and
submission record — the machine's history shows *which agent* ran what.

## What an operator can ask

With the server connected, the assistant can carry conversations like:

- *"Is anything running right now? What's in the queue?"* —
  `scan_status`, `scan_history`.
- *"Run a no-scan with the Amp4In save set, 20 shots."* — names checked
  against `list_scan_configs`, then `submit_scan` behind the ask prompt,
  preflight warnings surfaced for explicit acknowledgement.
- *"How is it going?"* — `scan_progress`, per-shot counts from the
  worker's document stream.
- *"What did scan 12 measure?"* — `get_scan_result` from the Tiled
  archive.
- *"Run the standard analysis on it and show me the beam profile."* —
  `run_scan_analysis`, polled via `get_scan_analysis`, figure fetched
  through its `/figures/…` URL.
- *"Stop the scan."* — `stop_scan`, gracefully, always available.

Each step is an auditable tool call with its own gate — which is exactly
the trade the whole design makes: flexible language in front, a fixed and
reviewable machine surface behind.
