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

**Decision:** deployed strict mode = **`SCAN` (gas jet on, free-running) +
`trigger_and_read`** during shots, `STANDBY` (jet off) during motor moves —
which is what `geecs_step_scan` already does with `arm_trigger`/`disarm_trigger`
and no `fire_shot`. The real strict-vs-free-run distinction on this hardware is
therefore **`trigger_and_read` (every device must catch each free-running shot,
else abort)** vs **reference-paced (one device gates, others contribute by
timestamp)** — both riding on `SCAN`, not single-shot vs free-running.

`geecs_single_shot` (below) is a correct, tested **primitive**, but it is left
*unwired* in the scanner dispatch until either a full-power single-shot config
exists or it is used on a non-amplitude-gated experiment (e.g. Thomson, whose
YAML has no `Amplitude.Ch AB`). The rest of this document describes that
primitive.

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
