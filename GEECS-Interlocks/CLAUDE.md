# GEECS-Interlocks

The software interlock server. It watches device values and external
status packets against one named configuration, and **reports** the
verdict to Master Control on a TCP stream. It never pauses a scan or
writes to a device: what a trip stops is Master Control's decision. It runs
headless (`geecs-interlocks serve`); configurations are edited in the
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
| `web/` | The read-only page on the kit |
| `demo.py` | `--fake-devices`: fake GEECS devices, their scripted values, the fake UDP sender |

## Rules

- **Fail-safe, everywhere.** Not heard from yet, disconnected, silent past
  `stale_after_s`, not a finite number, an external signal before its first
  packet or after one it cannot read, a settings check not armed: all
  unsafe. A new condition kind or input path states its unsafe cases first
  and tests them.
- **Report only.** The `interlocks-edges` import contract keeps scan
  execution, analysis and the other surfaces out. No route acts on the
  experiment.
- **One snapshot, many readers.** The stream and the page draw the
  monitor's snapshot and compute no verdict of their own, so they never
  disagree. A new reader reads `Engine.snapshot()`.
- **Master Control's frame holds one keyword.** It reads a frame only when
  it holds a single `SAFE` or `WARNING`. `summary` (the default) keeps
  that: its one keyword is the leading word, and "safe" / "warn" inside a
  name, reason or packet is broken up by `wire.clean`. The server's own
  reasons are worded so they never need cleaning (`clear again`, `alarm`).
  `legacy` repeats the keyword per interlock and goes once the hardware
  test settles which format Master Control reads. Pinned over every mix of
  states in `tests/test_wire.py`.
- **UDP listeners outlive a reload.** Senders transmit only on a change, so
  a rebuilt listener would forget the last state and read unsafe until the
  next one. A reload that names a port it cannot bind is refused whole, and
  a failed reload closes only the ports it opened.
- **A packet reads safe only when it plainly says so.** Its first word
  decides (`safe`, `warning`, `unsafe`); a `safe` followed by a negation,
  and every other packet, is unrecognised (`not safe`, `safe: false`,
  `motion done: safe`), and an unrecognised packet makes the signal
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
- **Facility values have one home:** ports, bind hosts and the wire format
  come from `config.ini [interlocks]`; without them every server listens
  on `127.0.0.1` only.

## Running and testing without the lab

`serve --fake-devices` runs `geecs_core.testing.FakeGeecsServer` devices
through the real subscribe path and starts a fake UDP sender; `listen
HOST:PORT` prints what Master Control receives. The tests need no extras
and no network.

## Owed on hardware

- Whether a real device's subscription delivers frames between changes, so
  `stale_after_s` times receipt rather than activity (`acq_timestamp`).
- Master Control reading the `summary` frame with several interlocks armed,
  and what it does when the stream is absent (asked on #1403).
