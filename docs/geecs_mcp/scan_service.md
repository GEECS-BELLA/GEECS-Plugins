# The scan service

The scans domain is the agent's way onto the machine: it reads the
[web scanner](../web_services/scanner.md)'s state, queues scans from
saved presets, makes single moves, waits for them, and halts — all over
the scanner's own HTTP API (`:8300`), so every request passes the same
preflight and refusals as the operator's Start button.

Tool classes follow the [safety model](overview.md#the-safety-model):
**R** read (auto-allowed), **W** write (the deployment's tier decides;
gated headless), **S** the halt family (never blockable).

## The verbs

| Tool | Class | What it does |
|---|---|---|
| `scan_status` | R | The scanner's status snapshot: manager connection, RE state (`idle` / `running` / `paused`), queue depth, the running item, readiness, the Tiled writer's state, the scanner version |
| `scan_progress` | R | The running (or last) scan: scan number, shots done of planned, state, exit status, the paused reason, scan folder and day |
| `scan_history(limit=10)` | R | The queue as the scanner shows it: running, waiting and recently finished rows, each with `item_uid`, `scan_numbers`, `run_uids`, state and the submitter |
| `list_scan_configs(kind, name?)` | R | The experiment's catalog by kind (`presets`, `trigger_profiles`, `scan_variables`, …), with the scanner's `unavailable` reasons; `name` on `presets` returns that preset's document |
| `submit_preset(name, operator, acknowledged?)` | W | Queue one scan from a saved preset, through the scanner's preflight (below). Returns `item_uid` |
| `move(variable, value, operator, wait=true, timeout_s=60)` | W | One move through the scanner's `mv` plan, **idle-only**; with `wait`, returns once the move finished, with the readback |
| `wait_for_scan(item_uid, timeout_s=240)` | R | Block until the queue item finishes (cap 600 s); returns its state, `scan_numbers` and `run_uids` |
| `pause_scan(force?, operator?)` | S | Deferred pause — lands at the next step boundary; `scan_status` says `paused` once it has |
| `resume_scan(operator, force?)` | W | Resume a paused scan. It *restarts motion*, so it is a write, not a halt. Refused when nothing is paused |
| `stop_scan(force?, operator?)` | S | Graceful stop after the current shot (partial data is kept); blocks until the scanner reports it landed (the scanner's budget is 120 s) |
| `clear_queue` | W | Remove every **waiting** item; the running one is untouched |

Names always come from `list_scan_configs` — an agent is told never to
invent catalog names, and an unknown preset comes back as `not_found`
with the catalog's names.

## Submitting a preset

`submit_preset` does what the scanner page does, in the same order:

1. Fetch the preset document from the scanner (`not_found` if it is not
   in the catalog).
2. Run the scanner's **preflight** on it. A hard refusal is
   `invalid_request` with the preflight outcomes. If preflight asks
   questions the call has not acknowledged, the result is
   `policy_refusal` with the questions **verbatim** under
   `needs_acknowledgement`, plus the summary and planned shot count.
3. Submit, with the acknowledged `check` names and the identity string.
   The scanner reruns preflight itself; a question that appeared in
   between comes back exactly as in step 2. Anything already queued is a
   `policy_refusal` (`pending_items`) — the server never clears the
   queue to make room.

**The server never acknowledges a question on its own**: the person
supervising answers, and the agent passes their `check` names back in
`acknowledged`. Success returns `item_uid` with the plan and preflight
outcomes; the scan number and run uid exist only once the worker claims
the scan, so they come from `wait_for_scan`.

## Moving one variable

`move` sends the variable — a catalog name, a `Device:Variable` or a bare
device, resolved by the scanner exactly as a scan axis is — to the
scanner's `mv` plan. It is refused (`policy_refusal`, the scanner's
words) while a plan runs or anything waits. Range limits are the
database's, enforced by the gateway on the setpoint; the server adds
none. With `wait`, the result carries the readback (`value`, `units`,
`timestamp`, age) — or `null` with a `readback_detail` for a bare device
or a pseudo variable, which have none. A timeout is `finished: false`,
not an error.

## Identity and halting

`submit_preset`, `move` and `resume_scan` require `operator` and send
`agent, admin supervision (<operator>)`; an empty name is
`invalid_request`. The scanner records a submission's operator on the run
(`md.geecs.operator`); its move route does not yet stamp it on the queued
`mv` item. `clear_queue` takes none. `stop_scan` and `pause_scan` pass
`operator` on when given and never refuse over it (the result notes a
name that could not be recorded); `force` is passed through to the
scanner, and the server keeps no ownership check of its own. A halt with
nothing to act on is a `policy_refusal` in the scanner's words. The
identity string records who supervised — it authorizes nothing; access
is [OSPREY's permission lists and hooks](osprey.md#how-the-connection-works).

## A typical sequence

```text
list_scan_configs("presets")         → the real preset names
scan_status                          → idle, nothing queued?
submit_preset("Amp4In", operator=…)  → policy_refusal: needs_acknowledgement [...]
   (the supervising person reads the questions and agrees)
submit_preset("Amp4In", operator=…,
              acknowledged=[...])    → ok, item_uid
wait_for_scan(item_uid)              → finished, scan_numbers: [12], run_uids
   (or finished: false — call again; scan_progress for the live counts)
stop_scan(operator=…)                → if something is wrong, at any point
```
