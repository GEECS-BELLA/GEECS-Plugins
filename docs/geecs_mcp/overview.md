# GEECS MCP Server — overview

**GEECS-MCP** is the general GEECS server for AI agents: one deployed
service that exposes the lab's GEECS-semantic operations — the scan
service, the config catalogs, archived results, post-scan analysis — as
typed tools an agent can call. It is what turns "ask the assistant to run
a scan and tell me what it measured" from a demo into a governed,
auditable interaction with the machine.

This page is a concepts-first orientation. The authoritative detail lives
in the package alongside the code: `GEECS-MCP/CLAUDE.md` (architecture and
boundaries), `GEECS-MCP/deploy/DEPLOYMENT.md` (the one full inventory of
configuration keys, transports, and the permission-gating semantics).
Treat those as the source of truth if anything here disagrees.

## What "MCP" is

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an
open standard for connecting AI agents to external systems. A server
declares a set of **tools** — named, typed, callable functions with
documented parameters — and an agent connected to that server can invoke
them during a conversation. The agent decides *when* to call a tool from
the conversation's needs; the server decides *what each call is allowed to
do* and returns a structured result.

That split is the whole point. The language model brings flexible intent
("rerun yesterday's failed analyses"); the server brings a fixed,
reviewable surface (a `run_scan_analysis` tool with exactly these
parameters, exactly these refusal conditions). Nothing the agent says can
make the server do something outside its tool surface.

## Why GEECS needs its own server

BELLA is already EPICS-fronted through the
[GEECS Gateway](../geecs_gateway/client_overview.md), so an agent framework
with generic EPICS tools can read any process variable without this server
existing. But raw PV access cannot express the operations that actually
matter:

- "Run a 1D scan of the jet position using the `Amp4In` save set" — that
  is a **ScanRequest** submitted to the queueserver, not a PV write.
- "What did scan 12 measure?" — that is a **Tiled archive** lookup with
  the run's metadata and per-column statistics.
- "Which save sets and trigger profiles exist for this experiment?" —
  that is the **configs repository**, resolved and validated the same way
  the console does it.
- "Run the standard analysis on this scan" — that is the **ScanAnalysis
  pipeline** with its task queue and figure outputs.

The MCP server exposes exactly these GEECS-semantic surfaces. It
deliberately does **not** duplicate raw-PV channel tools — the agent
framework's own EPICS tools cover channel-level access, including
bounded setpoint writes (see the
[division of labour](osprey.md#the-division-of-labour)).

## Where it sits in the architecture

The server is a **peer client of the queueserver, with the same standing
as the console** — and never an engine. It submits `ScanRequest`s through
the same client seam the console uses (`geecs_bluesky.qs_client`),
resolves configs through the same resolver, and reads results from the
same Tiled catalog. Scan execution stays entirely in the GEECS engine (the
queueserver worker); the MCP never drives devices shot-by-shot.

```mermaid
flowchart LR
    subgraph agent side
        O[Agent framework<br/>e.g. OSPREY assistant]
    end
    subgraph geecs-mcp [GEECS MCP Server]
        S[scans domain]
        A[analysis domain]
    end
    subgraph services [GEECS services]
        Q[Queueserver worker<br/>RunEngine]
        T[Tiled archive]
        C[Configs repo]
        D[Data share]
    end
    O -- "MCP tool calls" --> S
    O -- "MCP tool calls" --> A
    S -- "qs_client (submit / status / stop)" --> Q
    S -- "resolver (listings / validation)" --> C
    S -- "results" --> T
    A -- "statuses / figures / analysis runs" --> D
    Q -- "writes scans" --> D
```

Two standing doctrines shape everything above:

- **Write-surface doctrine.** GEECS-*semantic* writes — scans, actions,
  manual moves, analysis — go through MCP verbs only, each a named,
  gated tool with its own refusal logic, and scans stay in the GEECS
  engine. Channel-level setpoint writes are deliberately *not* MCP
  territory: the agent framework's own EPICS write tool can set gateway
  `:SP` PVs directly, bounded by its limits database and gating — but
  that raw path bypasses the GEECS client-side hardening (put-failure
  visibility, confirm/pseudo semantics, mid-scan refusals), which is
  exactly why operations with GEECS semantics belong behind an MCP verb.
- **A client, never an engine.** The server imports only the shared
  public seams, never engine internals. If a tool needs something
  private, the right move is to promote a small public module in
  GeecsBluesky — the server's needs surface real seams rather than
  growing a second engine.

## Domains

Tools are organised into **domains** — subpackages added as they earn
their keep, never speculatively:

| Domain | Status | What it covers |
|---|---|---|
| [Scan service](scan_service.md) | Built (v0 read, v1 control, v2 verbs) | Status, history, results, config listings, request validation, submit/stop/queue, actions, manual moves, pause/resume |
| [Analysis](analysis.md) | Built (read + execution) | Task statuses and output trees, figures, on-demand ScanAnalysis execution |
| Health / DB / Logs | Candidates | Gateway and archive probes, device-variable metadata, log triage as a tool |

Capabilities that require Windows-only acquisition SDKs (some analysis
diagnostics) do not force this server onto Windows — the pattern is a
small *satellite* MCP server on a Windows box, registered as a second
server entry with the same conventions.

## The safety model

Every tool belongs to one of three classes, and the class determines how
it is gated (the constants live in `geecs_mcp/tool_names.py`, so
permission lists import symbols rather than retyping strings):

- **R — read-only.** Status, listings, results, figures, dry-runs. Safe
  to auto-allow; calling them changes nothing.
- **Q — queueing.** Anything that starts work or changes state:
  `submit_scan`, `clear_queue`, `run_action`, `move_scan_variable`,
  `resume_scan`, `run_scan_analysis`. Interactively these surface a
  native *ask* prompt (a human sees each call with its arguments before
  it runs); headless/unattended operation gates them through an explicit
  `write_tools` allowlist in the agent framework's configuration.
- **S — stop direction.** `stop_scan` and `pause_scan`. These are asked
  about interactively but **deliberately never listed in `write_tools`**
  — a halt must never be blocked on any path. Making the machine quieter
  is always allowed.

On top of the classes sit per-verb protections: submissions carry a shot
cap and an **acknowledge-warnings loop** (the server never silently
continues past a preflight question — the agent must explicitly
acknowledge, and the acknowledgement is stamped into the run's
`SubmissionRecord` for provenance); stop and resume carry **ownership
etiquette** (another client's scan needs `force=true`, which is
approval-gated). Every run submitted through the server is attributed to
a configured client identity, so runs trace back to the agent that
started them.

## Conventions every tool follows

- **Tools never raise.** Every result is a JSON envelope: `ok: true`
  plus the payload, or `ok: false` with an `error_kind` from a fixed
  taxonomy (`invalid_request`, `not_found`, `policy_refusal`,
  `manager_unreachable`, …) and a message that preserves the engine's
  own wording — those strings are the operator vocabulary.
- **Payloads are context-sized.** Results return metadata, column names,
  and capped statistics — never full event tables; figures return
  *references* (path + fetch URL) rather than image bytes, with a
  bounded thumbnail as an explicit opt-in. Anything truncated says so.
- **No tool blocks on completion.** Long work is submit-and-poll: a
  submit tool returns as soon as the work is enqueued, and a read tool
  polls progress. A stuck tool call cannot wedge an agent conversation.
- **Everything degrades honestly.** The server always starts; an
  unconfigured or unreachable dependency turns the affected tools into
  clear refusals that name what is missing, never crashes.

## Configuration and deployment

Configuration is only the standard
`~/.config/geecs_python_api/config.ini` — the same fleet-wide file every
GEECS Python tool reads; the server introduces no new config format.
`GEECS-MCP/deploy/DEPLOYMENT.md` is the one full key inventory and the
deployment runbook. Two transports exist: **stdio** (the dev loop — the
agent framework launches the server as a subprocess) and **central HTTP**
(the multi-machine mode — one server process on the lab server, reachable
from any machine on the network).
