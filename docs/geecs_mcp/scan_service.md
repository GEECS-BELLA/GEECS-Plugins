# The scan service

The scans domain is the server's first and largest surface: everything an
agent needs to observe, validate, submit, steer, and stop scans — as a
client of the queueserver, exactly like the console.

Tool classes below follow the [safety model](overview.md#the-safety-model):
**R** read-only (auto-allow), **Q** queueing (asked/gated), **S** stop
direction (asked, never blockable).

## Observing (R)

| Tool | What it returns |
|---|---|
| `scan_status` | The RE Manager's picture: manager/RunEngine state, queue length, the running item |
| `scan_history` | Recent queue history items, newest last, field-tolerant |
| `get_scan_result` | A completed run from the Tiled archive: metadata, column names, capped per-column statistics — never the full event table |
| `list_scan_configs` | The experiment's config catalogs, by kind: save sets, trigger profiles, presets, optimizer configs, scan variables |
| `validate_scan_request` | Full dry-run validation of a `ScanRequest` without submitting anything |
| `scan_progress` | Poll-friendly progress: manager state plus a best-effort per-shot picture from the worker's document stream (planned totals, shots completed, exit status, and — while paused — the failed-move reason) |
| `describe_action` | A dry-run step table for a named action plan (runs on the worker, needs an idle manager, changes nothing) |

Names always come from `list_scan_configs` — an agent is told never to
invent catalog names, and unknown names come back as clear `not_found`
refusals rather than half-submissions.

## Submitting (Q)

`submit_scan` accepts either a saved **preset** by name or a composed
`ScanRequest` dictionary — the same one submission shape as the console,
validated against the schema at the tool boundary. Standing protections,
all enforced server-side:

- **Shot cap.** Agent submissions are capped (1,000 shots by default,
  deployment-configurable); optimization runs must state an explicit
  iteration budget.
- **The acknowledge-warnings loop.** The pre-submit preflight (the same
  checks the console runs: engine validation, unserved variables, device
  liveness, free-run staleness) can raise *questions*. The server never
  silently continues past one — the submission is refused with the
  question, and the agent must resubmit with an explicit
  acknowledgement. Every acknowledgement is stamped into the request's
  `SubmissionRecord`, so the run's metadata records who was asked what.
- **Identity.** The queue item and the `SubmissionRecord` both carry the
  server's configured client identity — runs trace back to the agent
  deployment that submitted them, and ownership checks compare against
  it.
- **The failed-item guard.** A failed item sitting at the front of the
  queue is surfaced, never silently cleared.

`clear_queue` is the one queue remover, and it never clears the running
item.

## Steering (Q) and stopping (S)

| Tool | Class | Semantics |
|---|---|---|
| `pause_scan` | S | Deferred pause — lands at the next plan checkpoint (the in-flight shot always finishes; expect 1–2 shots of latency by design) |
| `resume_scan` | Q | Resumes, retrying a failed move — it *restarts motion*, so it gates like a submission, with stop's ownership etiquette |
| `stop_scan` | S | Graceful stop (from running: pause-then-stop sequencing; partial data is kept). Another client's scan requires `force=true`, which is approval-gated |
| `run_action` | Q | Queue a named action plan — idle-only: an active scan refuses rather than silently queueing the action to fire later |
| `move_scan_variable` | Q | A manual move of a catalog scan variable through the worker's own move machinery (scan-identical completion semantics); idle-only, bounded-blocking |

The stop family is the deliberate exception to headless gating: halting
must work on every path, so `stop_scan`/`pause_scan` are never listed in
a `write_tools` allowlist and never blocked.

## The submit-and-poll lifecycle

No tool blocks on scan completion. A typical agent interaction:

```text
list_scan_configs            → pick a save set / preset by real name
validate_scan_request        → optional dry-run of a composed request
submit_scan                  → refused with a preflight question?
submit_scan (acknowledged)   → queued + started; identity stamped
scan_progress (repeat)       → shots completed / totals / paused reason
get_scan_result              → archived metadata + capped statistics
```

Each step is one bounded request/response; the conversation stays
responsive throughout, and every state transition the agent acted on is
visible in the manager history and run metadata afterwards.
