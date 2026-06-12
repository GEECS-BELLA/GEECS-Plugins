# Mode 1 — strict_shot_control

The plan owns each shot, literally: the shot controller (DG645) is put in
**single-shot mode** and the plan fires every trigger itself.

## Contract

- One fire → one complete event row. All sync devices are required.
- A required device not responding to *the plan's own shot* is a hard failure
  (timeout → abort; refire/retry policy possible later).
- Replaces the previous strict behavior (arm SCAN for a whole step while the
  trigger free-runs). No legacy third behavior is kept.

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
