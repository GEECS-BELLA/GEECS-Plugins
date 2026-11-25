# GEECS-Interlocks

The software interlock server. It watches device values and external
status packets against one named configuration, and **reports** the
verdict to Master Control on a TCP stream. It never pauses a scan or
writes to a device: what a trip stops is Master Control's decision. It runs
headless (`geecs-interlocks serve`) as one systemd unit per experiment
(`deploy/`, `DEPLOYMENT.md`); configurations are edited in the
configurator (`interlocks`, tier `maintainer`). The operator's view is
`docs/web_services/interlocks.md`; this file holds the rules for changing
the code.

## Layout

| Module | What it owns |
|---|---|
| `conditions.py` | The pure verdict: `evaluate(condition, Inputs, tripped=)` → `Verdict`. No I/O, no clock |
| `monitor.py` | The loop: one tick (`TICK_S`), every interlock's state, one immutable `Snapshot` per tick, events, per-check trends, arming |
| `sources.py` | `DeviceSource`: one `GeecsDevice` subscription per device, first connection retried in the background |
| `external.py` | UDP listeners (`ExternalSignals`), owned by the engine |
| `engine.py` | The unit: file, source, monitor, stream, listeners; `reload(path)` |
| `wire.py`, `status_server.py` | Master Control's frames and the stream that sends them |
| `config.py` | `config.ini [interlocks]` settings and the configuration paths |
| `present.py` | The words every reader uses: the verdict line, values, the kit's state words, the stall rule |
| `web/` | The page on the kit, and its two actions (`POST /api/arm`, `/api/reload`) |
| `notify.py` | systemd's notify protocol: `READY=1`, the watchdog fed by the loop (`Monitor(on_tick=)`) |
| `demo.py` | `--fake-devices`: fake GEECS devices, their scripted values, the fake UDP sender |

## Rules

- **Fail-safe, everywhere.** Not heard from yet, disconnected, silent past
  `stale_after_s`, not a finite number, an external signal before its first
  packet or after one it cannot read, a settings check not armed: all
  unsafe. A new condition kind or input path states its unsafe cases first
  and tests them.
- **Report only.** The `interlocks-edges` import contract keeps scan
  execution, analysis and the other surfaces out. No route acts on the
  experiment: the page's two actions act on the server alone.
- **An action is a person's, named and confirmed.** `POST /api/arm`
  (`{operator, interlock?}`, `Monitor.arm`) and `POST /api/reload`
  (`{operator}`, `Engine.reload` of the running file, never another path)
  refuse a blank operator (400), log the operator with the outcome, and
  answer the engine's own message (`ActionOut`: 200 done, 409 refused). The
  page asks for the name (remembered in the browser), confirms in
  the kit's dialog, sends once and never retries. The Arm buttons show only
  when an enabled interlock holds settings (`armable`): limits come from the
  file, so a configuration without a settings check has nothing to arm. A reload, like a start,
  leaves every settings check unarmed.
- **One snapshot, many readers.** The stream and the page draw the
  monitor's snapshot and compute no verdict of their own, so they never
  disagree. A new reader reads `Engine.snapshot()`.
- **Master Control's frame holds one keyword.** Master Control fires a shot
  on a new frame holding `SAFE` (and treats no frame as unsafe), so every
  frame holds exactly one keyword, its leading word: `SAFE` for an
  all-clear, `WARNING! TRIPPED` otherwise. "safe" / "warn" inside a name,
  reason or packet is broken up by `wire.clean`, and the server's own
  reasons are worded so they never need cleaning (`clear again`, `alarm`).
  The original server's per-interlock `SAFE: <name>` form is gone: it put
  `SAFE` in a frame while another interlock was tripped. Pinned over every
  mix of states in `tests/test_wire.py`.
- **The shot gate is a second stream, never a change to Master Control's.**
  With `--gate-port` (site.env `GEECS_INTERLOCKS_GATE_PORT`; blank: config.ini's
  `gate_port`, else none) the
  same snapshot is served on that port, on the stream's host, as the JSON v1
  frame GeecsBluesky's `ShotGate` reads (`gate.py`); a trigger profile's
  `shot_gate {host, port}` names it, and the scan worker holds each strict
  fire for it. One `StatusServer`, two renderers. A `data_stamp` is when
  the oldest device reading an interlock rests on arrived (wall clock),
  read before the inputs so it can only undersell freshness, never make old
  data look fresh; external signals add none (they report on a change), and
  an interlock reading no device carries the tick's time.
  The worker compares it with its own clock: same host, or agreeing clocks.
- **UDP listeners outlive a reload.** Senders transmit only on a change, so
  a rebuilt listener would forget the last state and read unsafe until the
  next one. A reload that names a port it cannot bind is refused whole, and
  a failed reload closes only the ports it opened.
- **A packet reads safe only when it says `safe` and nothing against it.**
  The whole text is searched, any case, whole words: `warning` / `unsafe`
  (also `un-safe`, `non-safe`) anywhere is unsafe and wins; otherwise `safe`
  anywhere is safe (`stage1 safe`), unless the packet holds a negation
  anywhere (`not currently safe`, `safe: false`, `isn’t safe`), which is
  unrecognised. Any other hazard word (`fault`, `fail-safe`) does not
  override `safe`: a sender reports a hazard with `warning` or `unsafe`. A packet with none of the
  words is unrecognised too, and an unrecognised packet makes the signal
  unknown (unsafe), never keeps the last state. Pinned in
  `tests/test_external.py` and `tests/test_monitor.py`.
- **A stalled evaluation trips the stream.** Past `present.STALL_S` (ten
  ticks) without a new snapshot, judged on the snapshot's monotonic stamp
  (`Snapshot.mono`) so a wall-clock step cannot hide it, `StatusServer`
  sends `render_stalled` instead of re-sending the last verdict. Every
  reader treats an old snapshot as no snapshot.
- **The page never keeps a verdict it stopped hearing.** Past 2 s without
  a snapshot it says "No signal", shows the last verdict with its time and
  greys every chip (`web/static/interlocks.js`).
- **Logging has one owner.** `main` configures it (stderr, which journald
  keeps under a unit); uvicorn runs with `log_config=None`, because its own
  config replaces the handlers already installed.
- **Arming records the moment, so it is a person's act.** `config_change`
  holds the settings seen when it is armed. `--arm-at-start` arms the first
  values that arrive and must never run in an automatic restart (a unit's
  `ExecStart`): it would arm a setting someone had just changed. The server
  refuses it when `NOTIFY_SOCKET` is set (under systemd), exit 2, which the
  unit's `RestartPreventExitStatus` keeps from looping.
- **A start a restart cannot fix exits 2.** A missing or invalid file, a
  blank `--config`, no experiment: `serve` logs why and returns
  `EXIT_REFUSED`, so the unit stops `failed` instead of looping. A blank
  host argument is no value (the config.ini or `127.0.0.1` default), never
  every interface. At run time a missing file still reads as no interlocks.
- **`/healthz` is healthy only while it evaluates something:** 503 with a
  `reason` when the snapshot is stalled or no interlock is armed.
- **The actions take JSON only** (`_json_only`, 415 otherwise): a
  cross-origin `text/plain` POST needs no preflight, so the check is the
  server's own, not left to the FastAPI version.
- **The watchdog is fed by evaluations, not by the process.** Under the
  unit (`Type=notify`, `WatchdogSec`), `Monitor._loop` calls `on_tick` after
  each evaluation that succeeds (rate-limited in `Notifier.watchdog`), so a
  stalled or failing loop gets the server restarted. Never feed it from
  another thread.
- **An external signal can name its sender.** With `ExternalCondition.sender`
  set (an IP address; a host name is refused), the condition reads only that
  address's packets: each sender keeps its own reading on the listener, so
  a stray host (or a forgotten `--fake-devices` run) can neither clear the
  signal nor overwrite its last state.
- **Facility values have one home:** ports and bind hosts come from `config.ini [interlocks]`; without them every server listens
  on `127.0.0.1` only.

## Running and testing without the lab

`serve --fake-devices` runs `geecs_core.testing.FakeGeecsServer` devices
through the real subscribe path and starts a fake UDP sender; `listen
HOST:PORT` prints what Master Control receives. The tests need no extras
and no network.

## Owed on hardware

- Whether a real device's subscription delivers frames between changes, so
  `stale_after_s` times receipt rather than activity (`acq_timestamp`). The
  shot gate depends on it too: a device that sends only on a change never
  stamps a verdict after a fire, so every gated shot would hold until the
  profile's `hold_pause_s` and pause the run (closed, but unusable).
- Master Control firing on this server's frames with several interlocks
  armed. Its maintainer has said (on #1403) that it needs a new `SAFE`
  frame for each shot, treats a missing stream as unsafe, reconnects, and
  finds the server's address and port in the GEECS database; the lab test
  confirms it against this server.
