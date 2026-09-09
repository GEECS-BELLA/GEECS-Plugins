# Device-layer audit — do we have the right ophyd-async devices?

Written 2026-09-09 after the first phase-1 attempt built a *parallel* device
class (`GeecsDevice`) instead of asking this question first. **Acted on the
same day**: `GeecsDevice` was deleted and `namespace.py` re-cut to compose
the classes below (`01_device_namespace.md`, "Design — compose, do not
invent"); items 1–4 under "What is missing" are done, the "amend" items are
scheduled by phase as stated. Sam's rule for
the whole effort: added code must be justified line by line, never a second
copy of a solved problem — and where a better copy replaces an old one, the
old one goes in the same change.

## The question

Stock Bluesky plans need *nouns*: long-lived objects that implement the
protocols the plans call — `Readable`, `Triggerable`, `Movable`,
`Stageable`, later `Pausable` and `Flyable`. GeecsBluesky already has a
device layer under `geecs_bluesky/devices/`. Is it the right set for that
job, and what (if anything) must be added or amended?

## Inventory — what exists, what it solves, and its fitness as a long-lived noun

| Class | Protocols | Solved problem it owns | Long-lived fitness | Verdict |
|---|---|---|---|---|
| `CaSettable` | Movable + Readable | `:SP` put through `GatewaySetpointPut` (addressing, coercion, timeout policy), readback child | No per-scan state | **Keep, reuse as-is** as the settable child |
| `CaMotor(CaSettable)` | Movable | readback-convergence poll within `tolerance`, move timeout, `GeecsMotorTimeoutError` | No per-scan state | **Keep, reuse as-is**; the DB `tolerance` (verified present for every catalog motor that exists in the DB) becomes the "is a motor" signal |
| `CaConfirmSettable(CaSettable)` | Movable | set-X / confirm-Y topology (catalog `confirm:`) | No per-scan state | **Keep, reuse** for catalog entries with `confirm` |
| `CaPseudoMovable` | Movable + Stageable | composite/pseudo variables with compiled `forward`, relative baselines captured at `stage()` | Baselines are per-run *by design* (stage/unstage) | **Keep, reuse** for catalog `kind: pseudo` entries; the namespace registers one per catalog entry |
| `CaAcqTimestampReadable` / `CaTriggerable` | Readable / +Triggerable | persistent `acq_timestamp` monitor; `trigger()` = wait for the next shot, no blind window | No per-scan state | **Keep**; monitor/trigger now live in `shot_monitor.py` mixins (moved, not copied) |
| `CaGenericDetector(ShotIdSupport, NonScalarSaveSupport, CaTriggerable)` | Triggerable + Readable + asset docs | **the acquirer noun**: readables + shot stamp, shot-ID companion columns, `nonscalar_save_path` column, external asset (Resource/Datum) docs, `localsavingpath`/`save` controls | Per-run state pushed in by the plan: `configure_shot_id(rep_rate)`, `configure_nonscalar_file_logging(path)`, `configure_external_asset_logging(scan_number, …)`; save flags fixed at **construction** | **Keep as THE triggerable device**; needs two amendments (below) |
| `CaTimestampedReadable(… FreeRunContributorSupport …)` | Readable (non-blocking) | free-run contributor: reference-relative shot offset + validity | `set_reference` per run | **Keep until phase 6** (free-run → fly scan), then retire with the free-run plan |
| `CaSnapshotReadable` | Readable | asynchronous device: one sample per row, optional `save` control | No per-scan state | **Keep as THE non-triggered device** |
| `CaTelemetryReadable` / `CaTelemetryGroup` | Readable (+Stageable) | soft tier: a failed read degrades to a null cell, never fails the row | No per-scan state | **Keep**; becomes the member type handed to `SupplementalData` in phase 2 |
| `CaActionSignalFactory` | (not a device) | per-`(device, variable)` cache of settables/readables for compiled action plans | per-scan | **Retire in phase 3**: with devices as nouns, actions target namespace children directly |
| `ScanContext` | Readable (synthetic) | per-row scan metadata columns built inside the plans | per-scan | **Retire in phase 3**: stock plans carry step metadata natively |
| `ShotController` / `CaPutSetter` | plan verbs (not a device) | trigger-profile state machine over `:SP` setters: arm / disarm / quiesce / fire, ordered multi-device writes | stateless | **Keep; amend in phase 4** (below) |
| `NonScalarSaveSupport`, `ShotIdSupport`, `FreeRunContributorSupport` | mixins | asset docs / save-path column; shot IDs from `acq_timestamp` + rep rate; contributor validity | per-run configure methods exist | **Keep** (they are what my `GeecsDevice` silently dropped) |

**Conclusion of the inventory:** the device layer is *right*. Every solved
problem — tolerances, `:SP` semantics, the shot monitor, shot IDs, asset
docs, save controls, telemetry soft-fail, pseudo axes — has exactly one
home, and none of it should be rewritten. What was missing was never a
device class; it was three small things around them.

## What is missing (justified additions)

1. **Composition: a device that is one GEECS device.** Today a
   `CaGenericDetector` holds the *readables* of a device, and that
   device's settables (`exposure`, `trigger`, `localsavingpath`) are built
   as separate objects by whoever needs them. For a noun, the settables
   must hang off the same object so `UC_Amp4_IR_input.exposure` resolves.
   ophyd-async supports this natively: a `Device` attribute set after
   `__init__` is a child (it connects with the device, is not a readable
   unless registered). **Addition: the namespace attaches `CaMotor` /
   `CaSettable` / `CaConfirmSettable` children to the `CaGenericDetector`
   or `CaSnapshotReadable` instance.** No new class. ~30 lines (naming +
   attach), lives in `namespace.py`.
2. **Per-device typing.** `CaSnapshotReadable`/`CaTriggerable` take one
   `datatype` for all variables (default `float`); the hardware run proved
   the served set mixes numerics, enums and char-array paths, and one wrong
   child fails the whole connect. **Amend: accept `datatype=None` (inferred
   at connect — ophyd-async 0.19 supports it) and, optionally, a
   per-variable mapping.** Small change in two constructors; `CaSettable`
   already accepts `None` (done in this branch).
3. **Which class for which device** — the triggerable rule. The DB has no
   `acq_timestamp` row yet, so: trigger-named devicetype variables minus
   the trigger-source devicetypes (Sam's shortcut, verified against all 105
   Undulator devices), overridable per device, DB row authoritative when it
   lands. **Addition: `looks_triggerable` (≈15 lines), in `namespace.py`.**
4. **Connection on first use.** Nothing in the layer connects lazily
   (construction and connection were one step, per scan). **Addition:
   `connect_on_demand` preprocessor (done, ~80 lines of logic).** Must be
   installed *last* on `RE.preprocessors` (review finding P1).

## What must be amended (and when)

- **`CaGenericDetector` save flags → per-run configuration (phase 2).**
  `save_nonscalar_data` / `save_control_only` are constructor arguments,
  so a long-lived detector cannot switch native saving per run. The
  preamble needs `configure_saving(enabled, control_only)` that
  (re)creates or reuses the `localsavingpath` / `save` children. With
  composition (item 1) those children *already exist* as `CaSettable`
  settables on the served set — the amendment is to make
  `NonScalarSaveSupport` use them rather than its own `epics_signal_rw`
  pair. Deletes code rather than adding it.
- **`ShotController` → an ophyd-async `Device` (phase 4).** The RunEngine
  calls `pause()` / `resume()` on every object it has *seen in a message*;
  a plain Python object driven through `abs_set` on its setters is never
  seen. Making it a `Device` with the setters as children lets the plan
  `stage` it and gives `Pausable` (STANDBY on pause, re-arm on resume) for
  free; `plans/pause_semantics.py` (197 lines) retires.
- **`_column_headers` on `CaSettable`** is computed from the construction
  `name`, which a parent's `set_name` later changes — stale for children.
  Recompute lazily from `self.name` (5 lines), needed once the s-file
  exporter reads these runs (phase 2).
- **Free-run contributor → `Flyable` (phase 6).** `CaTimestampedReadable`
  + `FreeRunContributorSupport` + `free_run_step_scan.py` become one
  flyer over the #806 `StandardDetector`. Not before.

## What this means for phase 1

- **Delete** `devices/geecs_device.py` (both classes, `VariableMeta`) and
  its tests. It duplicated `CaGenericDetector`/`CaSnapshotReadable` with
  fewer capabilities and re-implemented two provider rules.
- **Keep** `shot_monitor.py` (a move), `preprocessors.py`, and the
  hardware-derived rules — served set, inferred types, protocol-name
  collision, triggerable classification — relocated into `namespace.py`
  and the two constructor amendments.
- **`namespace.py` becomes**: roster from the DB (reusing
  `GeecsDbServedSetProvider` and the scalar-policy provider instead of
  re-stating their rules), one `CaGenericDetector` or `CaSnapshotReadable`
  per device built from its served/subscribed variables, settable children
  attached, catalog pseudo/confirm entries registered, name lookups and
  `export_into`. Target ≈ 200 lines of logic; no device behaviour.
- **Reuse ledger for the phase-1 PR** (every new symbol → reuses / replaces
  / new-because): to be written into the PR body and checked by the
  redundancy lens of the `/land` review brief.

## Open for Sam

- Attaching settable children to a `CaGenericDetector` puts a camera's 40
  settables on the noun. They cost nothing until touched, but they appear
  in the queue server's device tree (`:depth=`) and in tab completion. Is
  that the desired shape, or should only the catalog's scan variables be
  attached (smaller tree, but `bps.mv(cam.exposure, …)` then needs a
  catalog entry)?
- Per-device typing: is `datatype=None` (infer) acceptable as the default
  for every non-numeric variable, or should the DB `variabletype` vocabulary
  be completed so the type is always declared?
