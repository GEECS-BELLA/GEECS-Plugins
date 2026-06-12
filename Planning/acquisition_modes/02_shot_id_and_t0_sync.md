# Shot ID Derivation and Coordinated t0 Sync

The foundation both modes share. Replaces the current
`configure_shot_numbering` / `_derive_shotnumber` machinery in
`GeecsGenericDetector` (no back-compat needed).

## Shot ID — incremental derivation

A device's `shot_id` is its physical trigger-opportunity number: which tick of
the free-running external trigger an acquisition belongs to.

**Wrong (current) way — absolute:**

```
shot_id = round((acq_timestamp − t0) × rep_rate) + 1
```

Accumulates rep-rate error over the run: if the true trigger rate differs from
nominal by 0.05%, a 30-minute scan accumulates ~0.9 shots of error and
misquantizes near the end.

**Right way — incremental:**

```
Δ = round((acq_timestamp − last_acq_timestamp) × rep_rate)
shot_id = last_shot_id + max(Δ, 1)   # a new event is at least a new shot
```

Each step's rounding error is independent (0.05% of a 1 s period = 0.5 ms
against a 500 ms threshold); nothing accumulates. Seeded by the t0-sync stage:
`(t0_acq_timestamp, shot_id=1)`.

Properties:

- Exposure-independent — `acq_timestamp` is back-dated to acquisition start.
- Clock-skew-immune — each device only ever compares against its own history.
- Jumps > 1 across stage-move dead time are expected (trigger keeps ticking
  while outputs are blocked / shots aren't read). Matching across devices is
  **equality**, never consecutiveness.
- A device that timed out repeats its last `acq_timestamp` → `shot_id`
  unchanged → stale data detectable.

## Implementation: `ShotIdTracker`

Small stateful helper in `geecs_bluesky/devices/shot_id.py`, owned by anything
that emits sync-device companion columns (`GeecsGenericDetector`, the free-run
timestamped readable):

```python
tracker = ShotIdTracker(rep_rate_hz)
tracker.seed(t0_acq_timestamp)          # from the t0-sync stage
shot_id = tracker.update(acq_timestamp) # idempotent for repeated timestamps
```

`update()` with an unchanged timestamp returns the same ID (no advance).
Unseeded tracker → no companion columns derivable → `valid=False`, NaN IDs.

## Coordinated t0 sync — plan stage

Formalizes the existing (non-Bluesky DAQ) fast procedure. Runs once at the
start of a free-run run, while the shot-control outputs are disarmed
(STANDBY), before the first arm:

1. Read every sync device's last `acq_timestamp` from its TCP cache.
2. `spread = max(timestamps) − min(timestamps)`.
3. If `spread ≤ window` (default 0.2 s; machines NTP-synced to ~50 ms over a
   week, trigger period 1 s): all caches hold frames from the **same physical
   trigger**. Seed each device's tracker with its own timestamp; record
   `device_t0s` and `t0_sync_window_s` into run metadata.
4. If the spread exceeds the window: fail loudly (configurable retry — wait
   for the next shot to propagate and re-check — but never proceed unseeded).

Strict mode does not need the sync stage for correctness (every read follows
its own awaited trigger), but runs it anyway when possible so strict rows
carry comparable `shot_id`s too. If unseeded in strict mode, seed each device
from its first awaited shot.

## Tests (FakeGeecsServer)

- Incremental ID immune to rep-rate mismatch: simulate a 0.05%-off trigger
  period over enough shots that the absolute method would misquantize; assert
  the incremental method tracks exactly.
- Dead-time jump: gap of N periods between events → ID advances by N.
- Repeated timestamp (device timeout) → ID does not advance.
- t0 sync: timestamps within window → all trackers seeded, metadata recorded;
  spread beyond window → raises.
