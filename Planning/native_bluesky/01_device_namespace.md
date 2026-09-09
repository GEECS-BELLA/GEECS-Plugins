# Phase 1 — devices as long-lived nouns, connected on first use

Status: **built; hardware-accepted 2026-09-09 in its first form, then
re-cut to compose the existing device layer** (see `01a_device_layer_audit.md`
for why). Branch `phase/01-device-namespace`, PR into
`feature/native-bluesky-plans`. Additive: the per-scan construction path is
untouched; the namespace is built beside it and the existing plans keep
running.

## Why this phase exists

Stock plans take devices as arguments (`count(detectors)`,
`scan(detectors, motor, ...)`) and stage them. For those to run under the
queue server, devices must be **nouns in the worker namespace**, resolvable
by name (`U_S1H`, `U_S1H.Current`) and connected without a preamble. Today
every scan constructs fresh role-specific objects from the request and the
startup namespace holds no devices at all.

## Design — compose, do not invent

The device layer under `geecs_bluesky/devices/ca` already owns every solved
problem (the audit's table). The namespace only decides **which existing
class each GEECS device is** and **what hangs off it**:

| GEECS device … | becomes | because |
|---|---|---|
| acquires per shot (`looks_triggerable`) | `CaGenericDetector(device, subscribed_vars, datatypes=…)` | shot monitor + `trigger()`, shot-ID columns, save controls, asset docs — all already there |
| anything else | `CaSnapshotReadable(device, subscribed_vars, datatypes=…)` | one sample per row |
| each served **settable** variable | attached child: `CaMotor` if the DB gives a tolerance (readback convergence), else `CaSettable` | `bps.mv(U_S1H.Current, 0.5)` moves with GEECS semantics; ophyd-async registers and names a child attached after construction |

Rules the namespace applies come from their existing homes, never
restated:

- **Which variables exist as children** — the gateway's served set
  (subscribed ∪ settable), from `db_runtime.GeecsDbServedSetProvider`. A
  root device connects every child, so an unserved child would make it
  unconnectable (found on hardware).
- **Which variables `read()` returns** — the DB subscribed list, from
  `db_runtime.GeecsDbScalarPolicy` (what a `db_scalars` save-set entry
  logs). A subscribed settable's Movable child is registered as a readable
  (`add_readables`), so its column is `U_S1H-Current-position`.
- **Every variable's CA type** — `geecs_core.db.variable_types.effective_vartype`
  (moved there from the CA gateway in this PR; canonical source is
  `devicetype_variable.choice_id` → the `choice` table): `numeric` → float,
  `string`/`path` → str (a path PV is a char array; ophyd-async's `str`
  reads it as a long string), an option list → enum read as `str` with the
  choices as metadata, `image`/`1darray` → not a child. No inference, no
  guessing; the declared type is the type the gateway served the PV with.
- **Triggerable or not** — `acq_timestamp` is generated inside LabVIEW and
  is not a DB row yet, so: trigger-named devicetype variables minus the
  trigger-source devicetypes (DG645, DG535, Highland DDG, TDK-Lambda);
  a DB `acq_timestamp` row wins; `DeviceRoster.triggered` overrides per
  device. Checked against all 105 Undulator devices: the 43 pushing
  `acq_timestamp` all classified; the 10 extras were idle acquirers.

Naming: the namespace and attribute names keep the GEECS spelling when it
is an identifier (`U_S1H`, `Current`), else `safe_name`; a settable whose
name collides with a Bluesky/ophyd attribute (`trigger` — the Amp4
camera's external-trigger enum) binds as `trigger_`. Lookups
(`namespace.variable`, `namespace.resolve("U_S1H:current")`) accept either
spelling case-insensitively.

## Lazy connection — `connect_on_demand`

Installed **outermost** on the RunEngine (`install_connect_on_demand`,
which removes any earlier instance and re-appends itself; the RE composes
preprocessors first-appended-innermost, so anything appended later —
`SupplementalData`, the phase-2 preamble — would otherwise inject messages
it never sees). Before the first message touching a namespace object it
yields `ensure_connected(obj)`: message-level, the same stub the current
preamble uses, so the "no blocking connect inside the RE loop" rule holds.
Two message shapes carry devices: `msg.obj` (`stage`, `set`, `trigger`,
`read`, …) and `declare_stream`'s `msg.args`, where the RunEngine
*describes* the devices before any `read` (found by the P1 test). Stock
plans stage the **root** ancestor of every device, so a scan over
`U_S1H.Current` connects all of `U_S1H`'s served children; a bare
`bps.mv(U_S1H.Current, …)` connects only the child. Connected devices stay
connected.

## Queue server

The startup profile builds the namespace from the DB at `environment open`
(`QS_DEVICE_NAMESPACE=off` skips it — hermetic tests, a box without DB
reach), exports the devices into the namespace and `__all__`, and installs
`connect_on_demand` last. `user_group_permissions.yaml` gives operators
`allowed_devices: ":?.*:depth=3"` so `U_S1H.Current` is addressable in a
plan argument. Clutter in the device tree is a permissions/UI concern
(`:depth=`), not a reason to shape the device model.

## Hardware acceptance (2026-09-09, worker box, live gateway)

`tests/test_namespace_hardware.py` (hardware-marked; `-m integration` does
not select it): namespace of 105 devices (53 triggerable); HTU-NoGas armed
and disarmed through the existing `ShotController`; stock `bp.count` — 3
shots on `UC_Amp4_IR_input`, `acq_timestamp` advancing at exactly 1 Hz, the
9 subscribed columns + the shot stamp; stock `bp.list_scan` over
`U_S1H.Current` −1 → +1 A in 0.5 A steps — readbacks −0.99984, −0.49966,
0.00008, 0.50008, 0.99989 A (DB tolerance 0.05); setpoint restored in the
finalize. Both runs `success`; run metadata carries the stock `motors`,
`detectors`, `plan_pattern`. **The re-cut (composition) form has not yet
been re-run on hardware** — owed before merge; expected identical numbers.

Three defects the mock could not show, each now a rule above: the served
set, DB-derived types, the `trigger` name collision.

## Reuse ledger (for the PR body)

| new symbol | reuses | replaces | new because |
|---|---|---|---|
| `namespace.GeecsNamespace` | `CaGenericDetector`, `CaSnapshotReadable`, `CaMotor`, `CaSettable`, `GeecsDbServedSetProvider`, `GeecsDbScalarPolicy`, `GeecsDbDeviceTypes`, `effective_vartype`, `safe_name` | (phase 3) per-scan device factories in `session.py`, `_DeferredConnectFactories`, `_build_request_detectors` | devices as nouns did not exist |
| `namespace.looks_triggerable` | — | — | the DB cannot answer it yet (LabVIEW-internal `acq_timestamp`) |
| `namespace.identifier_name` | `safe_name` | — | attribute spelling for plan arguments |
| `preprocessors.connect_on_demand` | `ensure_connected` (the preamble's stub) | (phase 2) `_connect_in_batches` in the preamble | connection was never lazy |
| `devices/ca/shot_monitor.py` | moved verbatim from `triggerable.py` | the same code in `triggerable.py` (deleted there) | one implementation for two hosts |
| `geecs_core.db.variable_types` | moved verbatim from `geecs_ca_gateway.config` | the gateway's copy (deleted; PVA gateway re-pointed) | three packages need it |
| `datatypes=` on the readables | — | — | the served set is not all floats |

Deleted in this PR: `devices/geecs_device.py` (the first attempt's parallel
device) and its tests.

## Decisions taken here

- Compose the existing classes; attach **every** served settable (not only
  catalog scan variables): the noun is the whole device, phase 3's
  retirement of `CaActionSignalFactory` needs every settable reachable, and
  tree clutter is handled by `:depth=` permissions.
- Types from the DB only; `choices`/`choice_id` is canonical; an option
  list is always an enum (Sam). No inference fallback.
- Motor vs settable from the DB tolerance (every catalog `kind: motor`
  target that exists in the DB carries one); catalog `confirm` and
  `pseudo` entries become namespace nouns in phase 3 with the axis
  expansion.
- Keep-connected after first use; no TTL.
- Loud failure on an unreachable DB at environment open.
