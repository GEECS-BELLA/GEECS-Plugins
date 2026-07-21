# GEECS Gateway — client orientation

The **GeecsCAGateway** is the GEECS access layer: a caproto Channel Access
soft-IOC that mirrors GEECS devices as EPICS PVs, so any EPICS-ecosystem
consumer — Phoebus displays, an Archiver Appliance, ophyd-async / Bluesky — can
talk to GEECS the same way it talks to any IOC, without growing its own bespoke
bridge.

```
GEECS device  --TCP push stream-->  readback PV   (caget / camonitor)
GEECS device  <--blocking UDP set--  setpoint PV  (caput …:SP)
```

This page is a client-facing orientation: enough to connect a display or a
consumer and read the values correctly. It is **not** the normative contract.
The authoritative, test-pinned detail lives in the package alongside the code:

- **`GeecsCAGateway/PV_CONTRACT.md`** — the normative API contract (naming,
  types, timestamps, alarms, collision and failure semantics; every claim
  pinned by a test).
- **`GeecsCAGateway/DEPLOYMENT.md`** — launch/CLI, config resolution, and the
  client-side addressing recipe (including Windows).

These files stay in the package and are not published on this site; treat them
as the source of truth if anything here and there ever disagree.

## PV naming

A PV name is the GEECS names joined by `:`, one component per level:

```
[experiment:]device:variable          readback
[experiment:]device:variable:SP       setpoint (only when the variable is settable)
[experiment:]device:connected         per-device liveness
```

Example: `undulator:u_s1h:current` and its setpoint `undulator:u_s1h:current:SP`.
Every name component is lowercase — any casing of a GEECS name resolves to
the same PV; the only uppercase in the namespace are the fixed structural
literals `:SP` and `.DESC`.

**Within a component**, only `[A-Za-z0-9_]` survive: dots, spaces, dashes,
parentheses — any run of other characters — collapse to a single underscore
(`Trigger.Source` → `trigger_source`, `Jet X pos` → `jet_x_pos`). The dot
collapse matters because EPICS reads `.` as the record/field separator.

The mapping is **lossy** — `Trigger.Source` and `Trigger Source` both normalize
to `trigger_source` — so never reverse-engineer a GEECS name from a PV string.
The gateway holds the authoritative reverse map (`PV → (device, variable,
kind)`) in its manifest, and a genuine collision is a startup error, never a
silent clobber.

## Readbacks vs setpoints

- **Readbacks are client-read-only**, driven purely by the device's TCP push
  stream (~1–5 Hz). `caget` serves the last cached value; `camonitor` posts once
  per changed frame. A write to a readback PV fails cleanly at the client.
- **Setpoints (`…:SP`)** forward the value to the device over GEECS's *blocking*
  UDP set before storing it locally, so **put-completion means GEECS
  convergence** — `caput -c` / ophyd-async `set().wait()` block for the physical
  move (30 s default budget). A failed set raises and leaves the `:SP` PV
  unchanged. Read state from the readback PV; the `:SP` value is the last
  *commanded* value, not the device readback.

Long GEECS save paths exceed the 40-char EPICS string limit, so path-typed
variables are served as char-array (long-string) PVs — read/write them with
`caget -S` / `caput -S` (ophyd-async handles this natively).

## Liveness — the `CONNECTED` PV

Every device has one `[experiment:]device:connected` PV: an enum
(`Disconnected` / `Connected`), `MAJOR_ALARM` while the device's TCP
subscription is down. **Prefer it as the liveness signal** over inferring
liveness from data — the gateway serves a device's data PVs whether or not the
device is up, and a merely *quiet* device (idling between triggers) is not a
dead one.

When a device's subscription drops, its readbacks go `INVALID` / `COMM` and hold
their last value — so `INVALID` on a readback means "not live", not "bad value".
Recovery is automatic on the next live frame.

## Frame-ordering guarantee

All data variables of a push frame are posted to their PVs **before** that
frame's timestamp variable(s). So a monitor callback on a device's
`acq_timestamp` (or `systimestamp`) observes a **completed frame** — every data
PV of that frame already holds the frame's values. Clients that need
whole-frame consistency should trigger/latch on the timestamp PV, not on a data
PV. (Cross-*device* correlation is out of scope for the gateway — that is
Bluesky's job.)

The timestamp variables are themselves float readback PVs carrying the **raw
LabVIEW-epoch** value; the same raw value is stamped on saved external assets
(images), which is how saved files tie back to acquisition. A non-positive
`acq_timestamp` means "no acquisition yet" (the `0.0` pre-acquisition
placeholder), never a shot at the epoch.

## Curated alarm limits (gateway 0.7.0)

Value-based alarms are an **optional curated overlay** from a MySQL
`ca_alarm_limits` table keyed by `(experiment, device, variable)`. Only non-null
thresholds apply, and only to served numeric readbacks; a crossed limit sets the
configured severity (`MINOR` / `MAJOR` / `INVALID`) with status `LOW` / `LOLO` /
`HIGH` / `HIHI`. The table is optional — if it is absent, the IOC starts with no
value alarms.

Database `min` / `max` remain **display** limits only (a UX hint), never
gateway-enforced control limits: GEECS stays the authority on valid set values,
and a faithful-but-out-of-range readback (notably a `NaN` from a failed online
analysis) is reported, not clamped.

## Derived numeric channels (gateway 0.8.0)

The gateway can expose additional **read-only float PVs** computed from one
source device's numeric push-frame values — for example a Convectron pressure
derived from a vacuum-gauge analog input. The v1 schema is deliberately narrow:
one output PV, one arithmetic expression (numeric operators and a small `math`
whitelist — no arbitrary Python), and **all inputs must come from the same
source device**, so a client latching on that device's timestamp observes raw
and derived values from the same completed frame.

Derived PVs carry honest alarm states, distinct from raw readbacks:

| Condition | Derived PV severity / status |
|---|---|
| Never computed, missing/empty or non-numeric input | `INVALID` / `UDF` |
| Expression runtime failure (e.g. division by zero) | `INVALID` / `CALC` |
| Source device subscription dropped | `INVALID` / `COMM` |
| Recovery after any failure | next successful computation writes `NO_ALARM` |

The declaration shape is the `derived_channels` config kind — see the
[Schema reference](../geecs_schemas/schema_reference.md) for its fields, and
`GeecsCAGateway/PV_CONTRACT.md` §3 for the normative behavior.

## Reading more

For anything load-bearing — the full type-mapping table, enum resolution rules,
UDP/TCP failure semantics, and the complete alarm policy — read
`GeecsCAGateway/PV_CONTRACT.md` in the source tree. For getting a client
connected (addressing, environment, Windows), read
`GeecsCAGateway/DEPLOYMENT.md`.
