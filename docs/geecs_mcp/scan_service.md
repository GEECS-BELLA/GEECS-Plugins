# The scan service

The scans domain is the server's first and largest surface: everything an
agent needs to **observe** and **halt** scans — as a client of the
queueserver, exactly like the web scanner.

!!! warning "No submit path since 0.9.0"

    This server has no write verbs. `submit_scan`, `run_action`,
    `describe_action`, `move_scan_variable` and `validate_scan_request`
    were removed when the native-Bluesky rebuild retired the queue-client
    calls they stood on — the submission surface is now
    `submit_plan`/`submit_preset` over the `count`/`sweep`/`optimize`
    plans, and the pre-submit preflight takes a preset rather than a
    `ScanRequest`.

    The verbs were deleted rather than rewired: this server is an
    experiment, not an operator surface. **Scans are submitted from the
    [web scanner](../geecs_scanner/overview.md)**, which is the operator
    front end. See issue #727 if an agent-facing write path is ever
    wanted back.

Tool classes below follow the [safety model](overview.md#the-safety-model):
**R** read-only (auto-allow), **Q** queueing (asked/gated), **S** stop
direction (asked, never blockable).

## Observing (R)

| Tool | What it returns |
|---|---|
| `scan_status` | The RE Manager's picture: manager/RunEngine state, queue length, the running item |
| `scan_history` | Recent queue history items, newest last, field-tolerant |
| `get_scan_result` | A completed run from the Tiled archive: metadata, column names, capped per-column statistics — never the full event table |
| `list_scan_configs` | The experiment's config catalogs, by kind: trigger profiles, presets, optimizer configs, scan variables, actions. (Save sets are **not** a kind — the rebuild removed them; a preset carries its device group) |
| `scan_progress` | Poll-friendly progress: manager state plus a best-effort per-shot picture from the worker's document stream (planned totals, shots completed, exit status, and — while paused — the failed-move reason) |

Names always come from `list_scan_configs` — an agent is told never to
invent catalog names, and unknown names come back as clear `not_found`
refusals rather than half-submissions.

## Steering (Q) and stopping (S)

| Tool | Class | Semantics |
|---|---|---|
| `pause_scan` | S | Deferred pause — lands at the next plan checkpoint (the in-flight shot always finishes; expect 1–2 shots of latency by design) |
| `resume_scan` | Q | Resumes, retrying a failed move — it *restarts motion*, so it gates like a submission, with stop's ownership etiquette |
| `stop_scan` | S | Graceful stop (from running: pause-then-stop sequencing; partial data is kept). Another client's scan requires `force=true`, which is approval-gated |
| `clear_queue` | Q | The one queue remover — explicit recovery from a failed item at the front of the queue. Never clears the running item, and nothing clears implicitly |

**Identity and ownership.** The server's configured client identity
(`[mcp] client_identity`) is what `stop_scan`, `pause_scan` and
`resume_scan` compare the running item's submitted-as value against: a
scan this deployment did not submit is *foreign*, and halting it requires
`force=true` — approval territory, and always recorded in the result.

The stop family is the deliberate exception to headless gating: halting
must work on every path, so `stop_scan`/`pause_scan` are never listed in
the `write_tools` gate and are never blocked.

## The observe-and-halt lifecycle

No tool blocks on scan completion. A typical agent interaction, with the
scan itself started by an operator from the web scanner:

```text
list_scan_configs            → read the real catalog names
scan_status                  → is anything running, and whose is it?
scan_progress (repeat)       → shots completed / totals / paused reason
stop_scan                    → graceful halt, if something is wrong
get_scan_result              → archived metadata + capped statistics
```

Each step is one bounded request/response; the conversation stays
responsive throughout, and every state transition the agent acted on is
visible in the manager history and run metadata afterwards.
