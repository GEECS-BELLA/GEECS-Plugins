# Phase 1 — devices as long-lived nouns, connected on first use

Status: **built and accepted on hardware 2026-09-09** (branch
`phase/01-device-namespace`, PR into `feature/native-bluesky-plans`).
Additive: nothing in the existing per-scan construction path changes in this
phase; the namespace is built *beside* it and the existing plans keep running.

## What the hardware run taught (2026-09-09, worker box, live gateway)

Three things the mock could not show, each now a rule in the code:

1. **The gateway does not serve every DB variable.** Its served set is
   subscribed (`get='yes'`) ∪ settable (+ `acq_timestamp`, `CONNECTED`), the
   same rule the unserved-variables preflight uses. A root device connects
   every child, so the namespace builds children **only for served
   variables** (`namespace.served_variable_names`; `include_unserved=True`
   for offline tooling).
2. **Many DB rows have no `variabletype`, and the PVs behind them are enums,
   strings and char-array paths.** The Amp4 camera's `Analysis`, `save`,
   `trigger`, `localsavingpath`, … all failed a guessed `float`, and one
   failed child fails the device. Rule: declare `float` only for
   numeric-typed rows or rows with a numeric hint (`tolerance`/`min`/`max`
   — `U_S1H:Current` has no type but a tolerance); everything else is
   `datatype=None`, the CA backend's inferred converter.
3. **GEECS variable names collide with Bluesky protocol names.** The camera
   has a settable enum literally called `trigger` (external trigger on/off);
   bound verbatim it overwrote `trigger()`. Any variable whose attribute
   collides with a class attribute (`trigger`, `name`, `read`, `set`, …)
   binds with a trailing underscore (`trigger_`); GEECS-name lookups are
   unaffected. Checked against the triggered class so the name is the same
   on every device.

And one design question the DB cannot answer today — **which devices are
shot-triggered**: `acq_timestamp` is generated inside LabVIEW and is not a
DB variable (yet). The gateway's PV contract says the PV exists on every
device but only acquirers push it, so a live probe misclassifies an idle
camera. Agreed shortcut (Sam): a device whose devicetype variables mention a
trigger is an acquirer, **unless** its devicetype is a trigger *source*
(`TRIGGER_SOURCE_DEVICETYPES`: DG645, DG535, Highland DDG, TDK-Lambda). A DB
`acq_timestamp` row wins outright once it exists; `DeviceRoster.triggered`
overrides per device. Checked live against all 105 Undulator devices: the
43 pushing `acq_timestamp` were all classified triggerable; the 10 extra
classifications were idle acquirers (cameras, ICT scopes, DAQ pads, FROG).
An optional startup probe logs any device the rule and the gateway disagree
on, so the exclusion list cannot rot silently.

**Acceptance run** (`tests/test_namespace_hardware.py`, integration-marked):
namespace of 105 devices from the live DB (53 triggerable); HTU-NoGas armed
and disarmed through the existing `ShotController`; stock `bp.count` — 3
shots on `UC_Amp4_IR_input`, `acq_timestamp` advancing at exactly 1 Hz,
the 9 subscribed columns + the shot stamp; stock `bp.list_scan` over
`U_S1H.Current` −1 → +1 A in 0.5 A steps — readbacks −0.99984, −0.49966,
0.00008, 0.50008, 0.99989 A (tolerance 0.05 from the DB); setpoint
restored to its pre-scan readback in the finalize. Both runs `success`;
run metadata carries the stock `motors`, `detectors`, `plan_pattern`.

Known naming wart for phase 2: a numeric settable with a DB tolerance
becomes a `CaMotor`, whose readback attribute is `position`, so the
camera's `exposure` column reads `UC_Amp4_IR_input-exposure-position`.
Harmless, but event-key naming should be settled when the preamble starts
writing s-files from these runs.

## What exists today (why this phase is needed)

Every scan constructs fresh device objects from the request's save sets and
axes (`plans/scan_request_plan.py` → `_DeferredConnectFactories` →
`GeecsSession.detector/motor/...`), connects them in the preamble
(`_connect_in_batches` → `ensure_connected`, 20 s per batch) and disconnects
them in the finalize. The ophyd class is chosen by the device's *role in the
request* (detector / contributor / snapshot / motor / settable), not by the
device. The queue-server namespace holds **no devices** (`startup.py`
exports `RE`, the plans and two functions), so no stock plan can be given a
device by name, and `user_group_permissions.yaml` gives operators
`allowed_devices: []`.

Stock plans take devices as arguments (`count(detectors)`,
`scan(detectors, motor, ...)`) and stage them. For those to run under the
queue server, devices must be **nouns in the worker namespace**, resolvable
by name (`U_S1H`, `U_S1H.current`) and connected without a preamble.

## Design

### `GeecsDevice` — one ophyd-async device per GEECS device

Built from the GEECS DB roster (`GeecsDb.get_experiment_devices`,
`get_experiment_device_types`, `get_experiment_device_variables`,
`get_subscribed_variables` — the same batch queries GeecsPvaGateway uses).
Children, one per scalar variable, attribute-named with `safe_name(var)`:

| variable meta | child |
|---|---|
| not settable, scalar | `epics_signal_r(dtype, ca_pv(exp, dev, var))` |
| settable, `tolerance` in DB **or** catalog `kind: motor` | `CaMotor(dev, var, tolerance=…)` — Movable with readback convergence |
| settable, otherwise | `CaSettable(dev, var)` — Movable, `:SP` put |
| `variabletype` image / 1darray / 2darray | **not a CA child** (non-scalar; served over PVA — #806) |
| device has `acq_timestamp` | device is **Triggerable**: the shot monitor + `trigger()` from `CaTriggerable`, extracted into a mixin so the class is reused, not re-implemented |
| every device | non-readable `connected_status` child (`Device:CONNECTED`) |

`dtype`: `numeric` → `float`; `string`/`path`/`choice` → `str`; unknown →
`float` (the existing `CaSettable` default). Recorded as a decision to
revisit if a variable type surprises us.

**What `read()` returns — selection through `configure`.** A GEECS save set
names the variables to log; a stock plan just says `count([UC_Amp4Input])`.
The device therefore *selects* which children it reads: default = the DB
"subscribed" (`get='yes'`) list (what GEECS itself logs), else all scalar
variables; a plan changes it with the stock `bps.configure(dev,
variables=[...])` (the Bluesky `Configurable` convention — `configure`
returns `(old, new)` and the selection is reported in
`read_configuration`, so every descriptor records what was logged and why).
Phase 2's preamble issues that `configure` from `md["geecs"]["capture"]`.
`stage()` starts caching the selected signals (monitor-backed reads);
`unstage()` stops.

Naming: attribute/namespace names are `safe_name(...)` of the GEECS names
(the same normalisation PV components use); the device keeps the original
GEECS device and variable names for PV minting and column headers. A
collision between two GEECS devices normalising to one name fails the
namespace build loudly.

### `GeecsNamespace`

`GeecsNamespace.from_experiment(experiment, *, resolver=None)` → the roster
from the DB; `GeecsNamespace.from_roster(...)` for tests / offline. Holds
`devices: dict[str, GeecsDevice]` keyed by namespace name, plus
`by_geecs_name`, `resolve("Device:Variable")` → the child object, and
`export_into(namespace_dict)` for the startup profile (`__all__` grows by
the device names). A DB failure at `environment open` **raises** — a worker
whose device roster is silently empty would fail every plan with
"unknown device", which is worse than a loud startup failure.

### Lazy connection: `connect_on_demand` preprocessor

Installed once on the RunEngine (`RE.preprocessors.append(...)`). A
`plan_mutator` that, on the first message in a run touching a namespace
object (`stage`, `set`, `trigger`, `read`, `configure`, `locate`, …),
yields `ensure_connected(obj, mock=…, timeout=…)` before it. Connecting is
**message-level** — the same `ensure_connected` stub the current preamble
uses — so the standing rule "a lazy connect inside the RE loop deadlocks"
(which is about *blocking* `run_coroutine_threadsafe` calls from plan
code) is respected. `ensure_connected` is idempotent (cached connect task),
so re-touching a device costs nothing.

Connected devices **stay connected** (no disconnect on `unstage`). This is
what every EPICS deployment does; channel count is bounded by what plans
actually use. Revisit with a TTL only if the gateway shows strain.

### Queue server

`startup.py` builds the namespace after the session and exports the
devices; `user_group_permissions.yaml` operator `allowed_devices` becomes
`":?.*:depth=3"` so `U_S1H.current` is addressable in plan arguments.
Plan-argument resolution of dotted sub-device names is stock queueserver
behaviour (`profile_ops`, `:depth=`).

## Acceptance for this phase

- Unit: namespace from a fake roster (mock), attribute naming, `resolve`,
  selection/`configure`, Triggerable only for `acq_timestamp` devices,
  collision detection, DB-failure raise.
- Mock RE: stock `bp.count([dev], num=3)` and `bp.scan([dev], dev2.var,
  …)` run against namespace devices with **no GEECS preamble**, connected
  by the preprocessor, shots paced by `ca_mock_helpers.start_pacer`.
- Existing suite unchanged (740 green at the branch point).
- Hardware (headless from the Mac via `GeecsSession.RE`): `bp.count` on
  the amp4in cameras and `bp.scan` over `U_S1H.current` −1 → 1 A / 0.5 A
  with the connect-on-demand preprocessor, HTU-NoGas trigger profile
  armed by hand for this phase (the preamble is phase 2).

## Decisions taken here (record in #807 when the phase lands)

- Settable children reuse `CaSettable`/`CaMotor` rather than a bare
  `SignalRW(read_pv, write_pv)`: the gateway acks a `:SP` put before the
  hardware moves, so move-complete needs the readback convergence those
  classes already implement.
- Selection via `configure`, not via passing individual signals as
  detectors: keeps `detectors=[device]` (how humans, save sets and the
  OSPREY panel name things) and records the selection in the descriptor.
- Keep-connected after first use; no TTL.
- Loud failure on an unreachable DB at environment open.
