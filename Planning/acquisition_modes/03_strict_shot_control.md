# Mode 1 — strict_shot_control

## Hardware reality check (June 2026) — what "strict" actually does

The original idea was "put the DG645 in single-shot mode and have the plan fire
every trigger." Reading the **real** shot-control configs
(`geecs-plugins-configs/scanner_configs/experiments/<exp>/shot_control_configurations/`)
showed that does not fit the deployed hardware:

- On **Undulator**, `Amplitude.Ch AB` gates a **gas jet**, not the trigger
  edges. `SCAN`=4.0 (jet on, data), `STANDBY`/`OFF`=0.5 (jet off). The DG645
  free-runs on external edges the whole time; data-taking is gated by raising
  the amplitude, not by firing single shots.
- The `SINGLESHOT` state only sets `ExecuteSingleShot=on` — it does **not**
  raise the amplitude — so it fires at jet-off level. That is exactly right for
  its real purpose (advancing `acq_timestamp` to establish t0/timing without
  firing the jet) but it is **not** a full-power data shot.
- No existing state composes "full amplitude + single-shot source + execute,"
  and authoring one touches shared configs used by the still-supported legacy
  (non-Bluesky) path. The amplitude-as-gas-jet-switch is acknowledged
  pre-existing hackiness; the proper fix (general per-scan setup/teardown of
  arbitrary device variables) is **deferred future work**, not this branch.

**Decision (updated 0.7.0):** strict mode now has **two** behaviors, chosen by
whether the shot-control config defines an `ARMED` state:

- **With `ARMED`** → plan-owned single-shot (fire-and-wait): arm `ARMED`
  (jet on + single-shot source, halting the free-run) once, confirm the
  trigger has stopped (`geecs_confirm_quiescent`), then fire one shot per row
  and await every device (`geecs_single_shot`).  This is the real strict
  contract.
- **Without `ARMED`** → fall back to **`SCAN` (jet on, free-running) +
  `trigger_and_read`**, `STANDBY` (jet off) between moves.  Every device must
  still catch each free-running shot or the scan aborts; the plan just doesn't
  own the firing.

The deployed experiment configs do **not** yet define `ARMED`, so they get the
fallback today.  To enable true single-shot, add this state to the shot-control
YAML (additive — the legacy non-Bluesky path ignores unknown states):

```yaml
# add an ARMED column alongside the existing OFF/SCAN/STANDBY/SINGLESHOT
Amplitude.Ch AB:
  ARMED: '4.0'                               # gas jet ON (full output)
Trigger.Source:
  ARMED: Single shot external rising edges   # single-shot mode (stops free-run)
Trigger.ExecuteSingleShot:
  ARMED: ''                                  # no-op; SINGLESHOT does the firing
  SINGLESHOT: 'on'                           # (already present) fires one shot
```

`ARMED` arms; `SINGLESHOT` (already in the configs) fires.  Amplitude set in
`ARMED` carries through the `SINGLESHOT` fire (which leaves it untouched), so
shots fire at full power.  The rest of this document describes the
`geecs_single_shot` primitive that backs this.

### `ARMED` is config-specific: internal vs external single-shot

`ARMED` means different things depending on whether the laser (external
trigger source) is running, so its `Trigger.Source` value differs per config:

| Config | `ARMED` → `Trigger.Source` | why |
|---|---|---|
| laser on (`HTU-Normal`) | `Single shot external rising edges` | a real external edge fires each single shot |
| laser off (`HTU-LaserOFF`) | `Single shot` | no external edge exists; the DG645 must self-generate the shot internally (mirrors `SCAN: Internal` in the laser-off config) |

This distinction lives **entirely in the YAML** — the Python is agnostic.
`BlueskyScanner` applies whatever `values_for_state(ARMED)` /
`values_for_state(SINGLESHOT)` contain, and `geecs_confirm_quiescent` only
watches `acq_timestamp` stop advancing; neither cares whether the single-shot
source is internal or external.  So adding the right `ARMED` per config is the
only thing needed — no code change distinguishes laser on/off.

## The geecs_single_shot primitive (built, not wired as the strict default)

The plan owns each shot, literally: the shot controller is put in single-shot
mode and the plan fires every trigger itself.

### Contract

- One fire → one complete event row. All sync devices are required.
- A required device not responding to *the plan's own shot* is a hard failure
  (timeout → abort; refire/retry policy possible later).

## Per-shot stub — ordering is load-bearing

`GeecsTriggerable.trigger()` baselines the current `acq_timestamp` *at call
time* and waits for it to advance. If the fire happens before the detectors
baseline, the frame can arrive first and the waiters time out on a shot that
already passed. So the fire must be injected **between trigger initiation and
wait** — which `bps.trigger_and_read` cannot do. New stub:

```python
def geecs_single_shot(detectors, read_devices, fire, *, group="shot"):
    yield from bps.create()
    for det in detectors:
        yield from bps.trigger(det, group=group, wait=False)  # baseline + arm waiters
    yield from fire()                                         # DG645 single-shot fire
    yield from bps.wait(group)                                # all devices saw the shot
    for dev in read_devices:
        yield from bps.read(dev)
    yield from bps.save()
```

`geecs_step_scan` keeps its loop/ScanContext/finalize structure but calls this
stub instead of `trigger_and_read`, and gains
`acquisition_mode="strict_shot_control"` metadata.

## Shot control YAML

The existing state map (SCAN / STANDBY / OFF, driven by `_set_trigger_state`)
gains:

- a `SINGLE_SHOT` state — set once at run start (trigger source → single shot)
- a fire action — the variable write that emits one trigger (DG645
  software-trigger equivalent), exposed as the `fire` plan callable

Empty-string values still mean "no-op for this state" (legacy
TriggerController convention).

## Timeout semantics improvement

The trigger timeout changes meaning from "did a free-running shot happen to
arrive within 3 s" to "did this device respond to **my** shot" — a clean,
attributable failure signal per device per shot.

## Tests (FakeGeecsServer)

- Fire → all devices advance → one complete row; companion columns
  `shot_offset=0`, `valid=True`.
- One device does not respond to the fire → `GeecsTriggerTimeoutError`
  attributed to that device; run aborts; disarm finalizer still runs.
- Fire-before-baseline regression test: a frame arriving before `trigger()`
  is called must not satisfy the wait (guards the stub's ordering).
