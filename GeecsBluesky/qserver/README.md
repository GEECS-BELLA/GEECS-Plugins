# GEECS Queueserver Launch Assets

This directory contains the user-level launch mechanics for a local
bluesky-queueserver RE Manager, plus the startup profile itself
(`startup/startup.py`) that turns the launched manager into a runnable GEECS
worker: it builds the module-level `RE` the manager keeps alive across
queue items (Tiled + s-file callbacks subscribed), exports every device of
the experiment as a noun (`GeecsNamespace`) and registers the stock
`bluesky.plans` verbs over them (`geecs_bluesky.plan_names.GEECS_PLAN_NAMES`).
See `startup/startup.py`'s module docstring for the import-order and
experiment-resolution contracts, and
`Planning/native_bluesky/03_clean_room_rebuild.md` for where the rebuild
stands (phase 1: the plan layer rebinds these names with the strict
`take_reading`).

## Launch

From this directory:

```bash
./launch_re_manager.sh
```

The launcher expects `start-re-manager` on `PATH`. It checks
`127.0.0.1:6379` and starts a local Redis server when that port is not
answering. The Redis binary can be overridden with `QS_REDIS_SERVER`; the
default is `redis-server`.

```bash
QS_REDIS_SERVER=/path/to/redis-server ./launch_re_manager.sh
```

The startup directory defaults to the `startup/` folder beside the
launcher (so the script works from any working directory) and can be
overridden with `QS_STARTUP_DIR`.

```bash
QS_STARTUP_DIR=/path/to/startup ./launch_re_manager.sh
```

The launcher passes `user_group_permissions.yaml` explicitly. Keep that
argument: without a permissions file, RE Manager may accept startup but reject
queue submissions with `success: False`, and the command-line failure can be
silent.

## Verify

In another terminal:

```bash
qserver status
geecs-qserver-ensure-ready     # opens the environment if closed, waits for idle,
                               # asserts plans_allowed lists every GEECS plan; exit 0 = ready
```

(`qserver environment open` is the raw gesture underneath; the entry point
adds the plan-list assertion. Under systemd the `geecs-qserver-ready`
oneshot runs it after every manager start — `deploy/DEPLOYMENT.md` § 2.)

Once the environment is open, the stock plans are registered over the
namespace devices — a queue item names a plan and its devices by name:

```bash
qserver queue add plan '{"name": "count", "args": [["U_S1H"], 3], "item_type": "plan"}'
qserver queue add plan '{"name": "scan", "args": [["U_S1H"], "U_S1H.current", -1, 1, 5], "item_type": "plan"}'
qserver queue add plan '{"name": "mv", "args": ["U_S1H.current", 0.0], "item_type": "plan"}'
qserver queue add plan '{"name": "run_action", "args": ["Amp4_DUMP_HP"], "item_type": "plan"}'
qserver queue start
```

`run_action` runs a named plan from the experiment's action library
(`action_library/actions.yaml`, read from disk on every item) as plain
stubs over the same devices: no run is opened, nothing is claimed.

The `qserver` CLI parses that argument as a **Python literal, not JSON**:
`null` / `true` / `false` are rejected with an unhelpful "Error occurred
while parsing the plan" — use `None` / `True` / `False` instead (or omit
optional fields; the example above works because it contains neither).
A full `ScanRequest.model_dump(mode="json")` payload contains `null`s, so
programmatic submitters should write `repr(item)` (Python literal) for
the CLI, or use `bluesky-queueserver-api`, which takes real dicts.

`QS_EXPERIMENT` (or `config.ini`'s `[Experiment] expt`) must resolve before
the manager starts — the profile fails loud at import time otherwise (see
`startup/startup.py`).

## Document stream (GUI progress)

The startup profile publishes every bluesky document to a
`bluesky-0MQ-proxy`, which the launcher starts alongside Redis (in port
`QS_DOC_PROXY_IN`, default 5567; out port `QS_DOC_PROXY_OUT`, default
5568). GUI clients get live per-shot progress by subscribing to the out
port:

```python
from bluesky.callbacks.zmq import RemoteDispatcher

dispatcher = RemoteDispatcher("<worker-host>:5568")
dispatcher.subscribe(lambda name, doc: ...)
dispatcher.start()  # blocking — run it in a background thread
```

The subscription contract for clients beyond the console — firewall,
wire format, late joiners, transport posture, stability — is
`deploy/DEPLOYMENT.md` § "External subscribers".

Do not confuse this with the manager's `--zmq-publish-console` stream
(port 60625): that one carries captured stdout/stderr **text** for log
tails, never documents. The two are complementary — documents for
progress, console text for the failed-move reason lines and log tail.

Opt out with `QS_DOC_PROXY=OFF` (launcher) plus `QS_DOC_PUBLISH_ADDR=OFF`
(worker). The stream is best-effort: a worker without it still runs scans
correctly — only live GUI progress is lost.

## Manual moves and action plans

A manual move is a queue item of the stock `mv` plan (above): idle-only
ordering and queue provenance for free; an action plan is a `run_action`
item.  The `function_execute` verbs of the retired funnel
(`geecs_move_variable`, `geecs_describe_action`, `geecs_run_action_plan`)
are gone with it (#807 phase 1); a step preview is client-side
(`plans.action_compiler.flatten_action_steps`).  The Console's actions
menu still calls the funnel's submitter verb until its rewire (§10.5 of
the plan of record).

## Troubleshooting

- **Every submission fails with `Plan 'count' is not in the list of
  allowed plans`** (any plan name, and `qserver status`
  otherwise looks healthy) — the worker environment is **closed**, so the
  manager's plan list is empty and every name fails validation
  identically; the message points at the plan, the cause is the
  environment (`worker_environment_exists: False`, `re_state: None`,
  `plans_allowed: {}`). A fresh clone or an unattended restart leaves the
  manager this way — bluesky-queueserver never opens the environment on
  its own. Fix: `systemctl restart geecs-qserver-ready` (or
  `geecs-qserver-ensure-ready` / `qserver environment open` by hand); the
  console's `worker_ready` preflight names this state instead of relaying
  the manager string (GEECS-Plugins#793). The same state without any unit
  failing: the RE worker *child* died while the manager survived — no
  systemd event fires, `geecs-qserver-ready` stays `active (exited)` from
  its last successful run, and only the console/MCP preflight refusal
  names the gesture (`systemctl restart geecs-qserver-ready`).
- **Allowed plans empty while the manager is idle with its environment
  open** (`worker_environment_exists: True`, `re_state: idle`,
  `plans_allowed: {}` for every user group, every submission refused
  "not in the list of allowed plans", `geecs-qserver-ready` still
  `active (exited)`) — the manager's own **download of the plan list
  from the worker timed out** (journal: `Failed to download the list of
  existing plans and devices from the worker process: Timeout`), seen
  while the host thrashed in swap (GEECS-Plugins#838). It is not a closed
  environment and needs no restart: `systemctl restart
  geecs-qserver-ready` — `geecs-qserver-ensure-ready` asks the manager to
  restore the lists from the worker's on-disk copy when the list is empty
  or incomplete with the environment up (`permissions_reload` with
  `restore_plans_devices=True`; by hand: `qserver permissions reload
  lists`). That copy (`existing_plans_and_devices.yaml` in the startup
  dir) is written by the *worker* from its namespace at every environment
  open (`--update-existing-plans-devices` default `ENVIRONMENT_OPEN`), so
  it is the running environment's list, not a stale one. `qserver
  environment update` is NOT the fix: the worker re-downloads only when
  its regenerated descriptions differ from its stored copy, so on an
  unchanged namespace it is a no-op (and it needs an idle manager). The
  file is stale only after a deploy that changed the plan set without
  re-opening the environment — and then the worker process is equally
  stale, so `systemctl restart geecs-qserver` is the gesture.
- **`queue add` returns `success: False` with no reason at the CLI** — the
  manager was launched without a permissions file, or the file lacks the
  group the client submits as (the `qserver` CLI uses `primary` by
  default). See `user_group_permissions.yaml`.
- **`queue start` succeeds but the item bounces back and nothing runs** —
  the startup profile does not define `RE = RunEngine({})` (paired with
  the launcher's `--keep-re`). Only the manager log shows the cause:
  `Run Engine is not found in the RE Worker environment`.
- **`qserver function execute` fails with `RE Manager must be in idle
  state`** — function execution requires a fully idle manager; it is not
  available while a plan is running *or paused*. Nothing in the GEECS
  design may rely on it mid-run.
- **`queue start` re-runs an old failed item, or the queue keeps
  growing** — on plan failure the manager returns the failed item to the
  *front* of the queue (default `ignore_failures: false`). Clear the
  queue (or remove the item) before resubmitting a corrected request;
  clients that blindly add-and-start will re-execute the failed item
  first.
- **A gated (or strict) scan's first arm fails with `no frame within 8 s`
  right after a camera server restart** — the PVA gateway's file plugin
  arms its session on the *first frame* after `Capture=1`, and a freshly
  restarted DG645 comes up in its external-edges default: with the laser
  off no edge reaches the camera, so no frame ever arms the plugin (2b
  acceptance, 2026-09-12). Fire a few shots by hand (the box in internal
  mode, then back) before the first scan of the day; the plugin's stale
  `NumCaptured_RBV` from the previous session is zeroed by the scan itself
  (GEECS-Plugins#853 is the plugin-side fix). The same symptom on a
  camera whose LabVIEW device was started *after* its gateway is the
  gateway's subscription gap (GEECS-Plugins#854).
- **`qserver history get` shows a literal `'...'` entry** — the CLI
  truncates long histories for display; the newest items may not be
  shown. Read history through the API (`bluesky-queueserver-api`) for
  anything programmatic.
