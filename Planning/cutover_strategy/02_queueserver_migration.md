# Queueserver adoption — engine service-ification scope (2026-08-19)

Decided 2026-08-19 (maintainer + agent discussion, held in the OSPREY
deployment session; full GeecsBluesky subsystem trace at repo tip
`c8aed444`). This note resolves the REPLACE-WITH-STANDARD line in
`01_gui_feature_inventory.md` (MultiScanner → bluesky-queueserver) into a
concrete scope — and widens it: the queueserver is not just MultiScanner's
replacement, it is the engine's **service surface**. The driver is
multi-client execution: GEECS-Console, a future web panel, and an external
agent framework (OSPREY, deployed for HTU) all become peer clients of one
RE Manager. Every one of those clients is gated on the same move — the RE
leaves the console's process.

## What the trace found (classification: where does it execute?)

- **(A)** inside plan generators / ophyd-async devices → runs unchanged in
  a headless worker
- **(B)** host-process machinery around the RE → relocates to worker
  startup or a plan preamble; mechanical
- **(C)** crosses to a human/GUI mid-run → no stock equivalent; redesign
  or drop

| Subsystem | Class | Verdict |
|---|---|---|
| step / free-run / single-shot / t0-sync / adaptive plans | A | port unchanged — the bespoke acquisition physics is *not* the problem |
| shot-control stubs (`arm`/`disarm`/`quiesce`/`fire`, `shot_controller.py:195-290`) | A | port unchanged |
| `save_enable_plan` + save-off finalize (`plans/run_wrapper.py:83-129`) | A | port unchanged (note: `os.makedirs` inside the plan) |
| `action_compiler` (`plans/action_compiler.py:160`) | A | cleanest piece in the package — pure generator, factory-injected |
| per-row telemetry read (`devices/ca/telemetry.py`) | A | port unchanged |
| `run_scan_request` prologue (~370 lines, `scan_request_runner.py:1350-1597`) | B | **the** refactor target — see "the one structural task" |
| per-scan device construction + connect on `RE._loop` (`session.py:227-235`) | B | moves inside the plan preamble (`ensure_connected`) |
| `ShotController` construction + `connect_setters` (`session.py:559-605`) | B | worker startup / preamble |
| `claim_scan_number` + scan-folder creation (`run_wrapper.py:43-74`) | B | relocates into the worker preamble (worker needs the share) |
| TiledWriter subscription (`session.py:217`) | B | trivially moves to worker startup |
| s-file export (`session.py:1672-1685`, post-`RE()`, uid-dependent) | B | becomes a stop-document callback |
| telemetry selection / DB policy / asset config (MySQL `GeecsDb`) | B | worker startup; all already failure-tolerant |
| `trigger_profile`/`trigger_variant` resolution | B | name+variant are already JSON; resolution moves worker-side |
| preflight dialogs (unserved / liveness / staleness, `preflight.py`) | B+C | logic portable; dialogs move **client-side pre-submit** (they run pre-claim today) |
| `OperatorChannel` / `DialogRequest` transport (`events.py:115-154`) | C | deleted — see decisions |
| `action_direct` + pause-window actions (`action_direct.py`, `pause_supervisor.py:316-351`) | C | **dropped** — see decisions |
| `BlueskyScanner` thread/flags/progress (`scanner_bridge/bluesky_scanner.py`) | C | deleted — RE Manager's queue/status API *is* this |
| optimize ask/tell loop (`session.py:1009-1047`, `plans/optimize.py`) | B | moves wholesale into the worker — see decisions |

Test posture: everything above is hermetically covered except the bridge
(whose tests pin GUI-thread semantics and die with the bridge) — the
migration works over a tested core.

## Decisions (2026-08-19, maintainer)

1. **Pause means quiesce.** "Stop triggering" is already a plan stub;
   pause semantics become plan-level pause handling (checkpoints placed so
   rewind is safe, quiesce on pause). The pause *supervisor* otherwise
   deletes.
2. **Pause-window actions are dropped.** Rationale: a paused RE never
   locks the machine — manual intervention via Phoebus/GEECS GUIs works
   during pause exactly as it always has, so the *capability* survives;
   what `action_direct` added was a formalized in-engine path, judged not
   worth its migration cost. Actions become ordinary queue items; the
   manager's state-gating ("a scan is paused — resume or stop first")
   replaces the bespoke interlock. Known trade, accepted: long scans lose
   fix-and-resume (stop → action → restart claims a new scan number).
   Deletes with this decision: `action_direct.py`, the three-way
   `ActionDecisionRequest` dialog, most of `pause_supervisor.py`.
3. **Preflight questions move to the client, pre-submit.** They run
   pre-claim today, so nothing about them is mid-run; in the queue world
   the console runs the checks and asks the operator with ordinary
   synchronous GUI code *before* queueing. The cross-thread
   `DialogRequest`/`threading.Event` transport and `NullOperator`
   layering delete.
4. **Mid-run operator interaction reduces to the stock pause verbs.** The
   remaining mid-run question ("didn't reach position — continue or
   not?") is binary and maps 1:1: plan catches the failed move status,
   records the reason, `bps.pause()`; clients render the paused state;
   resume = retry from checkpoint, stop = end gracefully. No structured
   question side channel is needed.
5. **Optimization stays in-worker; the loader call moves, not the code.**
   `OptimizationSpec` is already a Pydantic schema in `geecs_schemas`
   (rides in ScanRequest — JSON by construction) and the console's
   `optimization_loader` (`GEECS-Console/geecs_console/services/
   optimization.py`) is already a config→Xopt-stack factory. The
   queueserver change is only the injection site: the worker preamble
   invokes the loader when `request.optimization` is present, instead of
   the GUI injecting live objects into the bridge. Keeping ask/tell
   in-worker preserves one-scan-one-number. bluesky-adaptive is *not*
   adopted now; it remains the future shape for cross-scan campaigns
   (where each evaluation legitimately is its own scan).
   Prerequisite: the Xopt/evaluator stack relocates out of legacy
   `geecs_scanner.optimization` into an importable-headless home —
   already implied by the M6 deletion of GEECS-Scanner-GUI.

## The one structural task

There is no plan generator meaning "run this ScanRequest" —
`run_scan_request()` is a plain function that builds/connects devices,
claims the scan number, then calls `session.scan`, which builds the plan
and calls `RE`. Queueserver registers plans by name with JSON args, so the
new unit is:

    def geecs_scan_request_plan(request: dict):  # ScanRequest.model_dump()
        # prologue as plan preamble, inside the worker:
        # validate → resolve save sets/actions/trigger → construct+connect
        # devices → claim scan number → inner plans exactly as today

This **dissolves the serialization question**: bound methods and closures
(ShotController stubs, per-step callables, the optimize loader's output)
never cross a process boundary — the preamble constructs them worker-side
from the JSON request. The prologue is mostly pure, tested functions;
this is relocation, not rewrite.

## Scope

**Build**
- `geecs_scan_request_plan` preamble (above)
- quiesce-on-pause plan logic + checkpoint placement
- worker startup script: config.ini, `EPICS_CA_ADDR_LIST` *before*
  `geecs_bluesky` import (`__init__.py:12` contract), TiledWriter
  subscription, ZMQ document publisher (replaces bridge-derived progress
  events for GUIs), scan-log handler, optimization loader registration
- s-file export as a stop-document callback

**Relocate**
- Xopt/evaluator stack out of `geecs_scanner.optimization` (couples to M6)

**Delete**
- `BlueskyScanner` thread/state/progress machinery and its tests
- `action_direct.py`, `ActionDecisionRequest`, most of `pause_supervisor`
- `OperatorChannel` transport (`DialogRequest` events/threading)

**Console (client refactor)**
- pre-submit preflight (keeps its dialogs as plain GUI code)
- queue submission via manager API; paused-state rendering (device-error
  Continue/Abort dialog renders `re_state: paused` + recorded reason)
- progress from the ZMQ document stream (also re-drives per-shot beeps,
  per `01` RE-HOME)
- MultiScanner capability = the queue itself (closes `01`'s
  REPLACE-WITH-STANDARD line)

## Worker environment requirements

The worker runs where the console runs today: lab network (CA gateway
reachable), NetApp share mounted (scan-folder claim + ScanInfo writes),
MySQL reachable, `~/.config/geecs_python_api/config.ini` present,
`GEECS_SCANNER_CONFIG_DIR` → configs repo. Queueserver control-plane
security (CurveZMQ keypair) is new operational surface — key management is
an open item below.

## Open items

- CurveZMQ key management / who may submit (auth model per client).
- Verify manager behavior details before build: function-execution while
  paused (not relied on — informational), deferred-pause boundaries with
  GEECS blocking puts.
- Resumable long scans: dropping pause-window actions makes
  stop→fix→restart the recovery path; if operationally painful, the answer
  is resumable scan design (progress checkpointing), not resurrecting
  `action_direct`.
- Manual-intervention provenance: PV changes made from Phoebus during a
  pause are recorded by readbacks/telemetry but not annotated; consider a
  console habit or annotation hook.

## Amendments from the console test-scan review (2026-08-20)

- **scan.log is the worker's per-scan record — built early, on purpose.**
  The 2026-08-19 test scans showed scan.log, the terminal, and the GUI
  ticker telling three different stories; the terminal was the only
  complete one, and under a worker the "terminal" is a machine-global
  journal. Landed pre-migration (GeecsBluesky 0.51.0, cleanup pass):
  `scan_log.py` now attaches at the **root logger** and a pre-claim buffer
  (started at submission) opens the file with the connects/telemetry-drop
  window. The doc's "scan-log handler" worker-startup build item is
  therefore this existing mechanism relocated, not new design. Client-side
  pre-submit lines (preflight answers) stay client-side by decision 3 —
  the client stamps outcomes into request metadata, not into scan.log.
- **`reinitialize` disposition (bridge deletion detail).** Its essential
  content is already the plan's first phase: the fail-fast
  `validate_scan_request` call is the same function `run_scan_request`
  runs at execution (issue #529, no-drift by construction). Under the
  queue it runs twice by design — authoritative in the worker preamble,
  and **client-side pre-submit** for immediate feedback (a queue makes
  submission-to-execution gaps long; a typo must fail at submit, not at
  queue-front). The optimize-loader presence check becomes a
  worker-startup fact (decision 5); request storage and progress counters
  delete (the queue item and the document stream own them). No
  "validation plan stub" — it would run at the same moment as the
  preamble anyway and costs queue-item atomicity.

## External context (informational)

The HTU OSPREY deployment (separate repo) integrates against this repo
read-only today (Tiled, scan folders, analysis outputs, log triage) and
composes ScanRequests without executing them. When the queueserver exists,
OSPREY attaches its bluesky bridge to it and the agent becomes another
peer client, subject to the same queue and review semantics as every other
client. Nothing in this migration depends on OSPREY; the ordering benefit
is one-directional.
