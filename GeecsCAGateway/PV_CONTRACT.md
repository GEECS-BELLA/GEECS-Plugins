# GeecsCAGateway — PV Contract

This is the **normative contract** between the gateway (PV producer) and every
Channel Access client — GeecsBluesky's CA-backed ophyd-async devices, Phoebus
displays, the planned Archiver Appliance, ad-hoc `caget`/`camonitor` use.
Clients (including GeecsBluesky) *consume* this contract; the gateway owns it.
If gateway behavior and this document disagree, one of them has a bug — fix it
here or there, never by letting them drift silently.

Every claim below is traceable to code, and every stated example is pinned by a
test — see the [test map](#pinned-by-test-map) at the end. Items marked
**(PR #452)** describe behavior that lands with the `fix/strict-shot-plausibles`
branch (gateway 0.5.2), which merges before this document does.

Code anchors: `geecs_ca_gateway/pv_naming.py`, `config.py`, `channels.py`,
`gateway.py`, `transport/`. Design rationale lives in `DESIGN.md`; operations
in `DEPLOYMENT.md`.

---

## 1. PV naming

### Namespace

```
[Experiment:]Device:Variable          readback
[Experiment:]Device:Variable:SP       setpoint (only when the variable is settable)
[Experiment:]Device:CONNECTED         per-device status
[Experiment:]CAGateway:<SUFFIX>       gateway self-diagnostics
```

Example: `Undulator:U_S1H:Current` and `Undulator:U_S1H:Current:SP`.

- `:` is the **reserved namespace separator**, applied only between components
  (`pv_naming.pv_name`). It never appears inside a component.
- The experiment prefix is optional at the model level (`DeviceSpec.experiment`);
  the production entry point (`python -m geecs_ca_gateway --experiment NAME`)
  always sets it, so deployed PVs are three-component. Falsy parts (`None`/`""`)
  simply drop out of the join.
- The device component defaults to the GEECS device name
  (`DeviceSpec.pv_prefix`); the variable component defaults to the GEECS
  variable name (`VariableSpec.pv_suffix`). Both can be overridden per-spec
  (`prefix=` / `pv=`) for curation.

### Component normalization (`pv_naming.normalize_component`)

Within a single component, only `[A-Za-z0-9_]` survives. The exact
transformation is:

1. Leading/trailing whitespace is stripped.
2. Every **run** of characters outside `[A-Za-z0-9_]` — spaces, dots, dashes,
   parentheses, slashes, anything — collapses to a **single** underscore
   (regex `[^A-Za-z0-9_]+` → `_`).
3. Leading and trailing underscores produced by step 2 are stripped.

Examples (each pinned by a test):

| GEECS name           | PV component     | Why it matters |
|----------------------|------------------|----------------|
| `Trigger.Source`     | `Trigger_Source` | **The dot is critical**: EPICS parses `.` as the record/field separator, so `Dev:Trigger.Source` would read as field `.Source` of record `Dev:Trigger` |
| `Jet X pos`          | `Jet_X_pos`      | spaces |
| `Beam-Current (A)`   | `Beam_Current_A` | run of ` (` collapses to one `_`; trailing `)` stripped |
| `  padded  name  `   | `padded_name`    | whitespace + run collapsing |

This policy lives in **`geecs_ca_gateway.pv_naming`** — the one module both the
gateway (producer) and GeecsBluesky's CA devices (consumer) import — so the two
sides can never drift. `geecs_ca_gateway.naming` is a thin re-export.

### The mapping is lossy — use the manifest

`Trigger.Source` and `Trigger Source` normalize to the same PV component.
**Never reverse-engineer a GEECS name from a PV string.** The gateway holds the
authoritative bidirectional map in `GeecsCaGateway.manifest`
(`PV → (device, geecs_var, kind)` where kind is `"readback"`, `"setpoint"`, or
`"status"`).

### Setpoint PVs — `:SP`

A variable whose DB `set` flag is `yes` gets a companion setpoint PV at the
literal suffix `:SP` appended to the full readback name. Non-settable variables
(including the intrinsic timestamp variables) have **no** `:SP` PV.

### Per-device status PV — `CONNECTED`

Every device gets exactly one status PV: `[Experiment:]Device:CONNECTED`.

- Type: enum with `enum_strings = ["Disconnected", "Connected"]`
  (index 0 = Disconnected, 1 = Connected).
- Severity: `NO_ALARM` while connected; **`MAJOR_ALARM`** with status `COMM`
  while the device's TCP subscription is down.
- Client-read-only.

### Gateway self-diagnostics — the reserved `CAGateway` namespace

`[Experiment:]CAGateway:{UPTIME, HEARTBEAT, DEVICES_CONNECTED, VERSION}`
(devIocStats-style; updated by a 5 s status loop; `UPTIME` in seconds,
`VERSION` is the installed package version). The `CAGateway` device component
is **reserved**: a real GEECS device whose PVs would land there trips the
collision guard at startup rather than silently clobbering the status PVs.

**`[Experiment:]CAGateway:RESTART`** is the one *client-writable* PV in the
namespace (the devIocStats `SYSRESET` pattern): an enum with states
`["Idle", "Restart"]`. Writing `Restart` (label or index 1) makes the gateway
shut down cleanly and exit with the restart code (86); under the shipped
systemd unit that is an automatic relaunch, which — because the served set
rebuilds live from the GEECS database at startup — is also the **DB-resync
mechanism** after a device/get-list edit. Writing `Idle`/0 is a no-op.
Expect a few seconds of CA disconnect; monitors reconnect automatically.
Outside systemd (a foreground run) the process simply exits.

### Long-string (path) PVs

EPICS `DBR_STRING` caps at 40 characters, and GEECS save paths routinely exceed
that. Path-typed variables are therefore served as **char-array PVs**
(`ChannelChar`, capacity 512, UTF-8) per the standard EPICS long-string
convention (areaDetector `FilePath` does the same). Clients should read/write
them as long strings (`caget -S`, ophyd-async handles it natively). Plain
`string` variables stay native 40-char string PVs.

---

## 2. Readbacks vs setpoints

```
GEECS device  --TCP push stream (~1–5 Hz)-->  readback PV   (caget / camonitor)
GEECS device  <--blocking UDP set-----------  setpoint PV   (caput :SP)
```

### Readbacks

- Driven exclusively by the device's TCP subscription stream; the gateway fans
  each push frame into the device's readback PVs.
- **Client-read-only** (CA access rights = READ). A `caput` to a readback fails
  cleanly at the client. This is load-bearing: a client write that *stuck*
  would poison the change-suppression cache and freeze the PV at the client's
  value until the hardware next changed (a real, shipped bug — fixed in 0.5.0).
- `caget` serves the cached last value; `camonitor` receives a post per changed
  frame. There is no polling of the device.

### Setpoints

- A `caput` to `…:SP` is forwarded to the device over GEECS's **blocking UDP
  set** *before* the value is stored locally. GEECS sets do not return until
  the device reports convergence (or an error), so **CA put-completion means
  GEECS convergence** — `caput -c` / ophyd-async `set().wait()` block for the
  physical move.
- If the GEECS set fails (rejection, error, timeout), the setter raises, the
  value is **not** stored, and the CA put fails. Correct put semantics: a
  failed put leaves the setpoint PV unchanged.
- **Set timeout: 30 s default, configurable** (`GeecsCaGateway(set_timeout_s=…)`).
  This deliberately matches `CaMotor._DEFAULT_MOVE_TIMEOUT` (30 s) in
  GeecsBluesky — "a slow axis is not a dead one". A legitimate 10–30 s stage
  move must not be failed mid-flight. Gets keep the transport's short 10 s
  default: a read that takes 10 s *is* a dead device.
- The setpoint PV reflects the last *successfully forwarded* put, not the
  device readback. Read state from the readback PV; the `:SP` value is the
  commanded value.
- Write safety: GEECS enforces DB value limits server-side and returns an error
  the setter propagates, so an out-of-range `caput` fails correctly. The
  DB-derived limits on the PV are **display** limits (a UX hint), not
  gateway-enforced control limits — see §4.

---

## 3. Timestamps

### PVs carry GEECS acquisition time, not gateway receive time

Every readback post from a push frame is stamped with a wall-clock timestamp
extracted from the frame via the device's **timestamp ladder**
(`DeviceSpec.timestamp_vars`, default `["acq_timestamp", "systimestamp"]`):

- `acq_timestamp` — true shot time; present only on triggered devices.
  Preferred.
- `systimestamp` — present on every device. Fallback.

Both are LabVIEW-epoch (1904-01-01) seconds; the gateway converts to Unix epoch
by subtracting `2_082_844_800`. A missing rung falls through to the next; a
non-positive (pre-1970, i.e. implausible) result is rejected and also falls
through. If no rung yields a plausible time, caproto defaults to receive time.

### The timestamp variables are also PVs — carrying the RAW LabVIEW value

`Device:acq_timestamp` and `Device:systimestamp` exist as float readback PVs on
every device, holding the **raw LabVIEW-epoch value** (not Unix-converted).
That is deliberate: the same raw value is stamped on saved external assets
(images), so these PVs are the per-device acquisition/synchronicity signal.
They are read-only and have no `:SP`.

### The `0.0` pre-acquisition placeholder

Every float readback channel (including `acq_timestamp`) initializes to `0.0`
before its first update. Clients must treat a **non-positive** `acq_timestamp`
as "no acquisition yet", never as a shot at the epoch. GeecsBluesky's
`CaAcqTimestampReadable` reads non-positive as `None` for exactly this reason.
(Non-triggered devices never push `acq_timestamp`, so their PV stays `0.0`
forever — normal, not an error.)

### Frame-completeness ordering guarantee (PR #452)

**All data variables of a push frame are posted to their PVs before the frame's
timestamp variable(s).** Each PV write is an `await`; without this ordering, a
frame that happened to list `acq_timestamp` first could complete a strict-mode
Bluesky `CaTriggerable.trigger()` on the new shot id while the data PVs still
held the previous frame — pairing shot N's id with shot N−1's values.

Contract for clients: **a monitor callback on `acq_timestamp` (or
`systimestamp`) observes the completed frame** — every data PV of that frame
already holds the frame's values. Among the data variables themselves, device
payload order is preserved (the reorder is a stable sort on
"is-a-timestamp-variable").

Clients must therefore trigger/latch on the timestamp PVs, not on a data PV,
when they need whole-frame consistency. (Cross-*device* correlation remains out
of scope — that is Bluesky's job, see DESIGN.md "scope boundaries".)

---

## 4. Type mapping

### DB `variabletype` → PV dtype

| DB `variabletype` | dtype    | caproto channel | Notes |
|-------------------|----------|-----------------|-------|
| `numeric`         | `float`  | `ChannelDouble` | precision + EGU from DB |
| `string`          | `string` | `ChannelString` | native 40-char string |
| `path`            | `path`   | `ChannelChar` (512, UTF-8) | long-string convention, §1 |
| `choice`          | `enum`   | `ChannelEnum`   | options from the `choice` table |
| `image`, `1darray`| —        | **skipped**     | not scalar CA data |
| *(absent/unknown)*| `float`  | `ChannelDouble` | default |

Resolution quirks (all DB-driven, all pinned by tests):

- The `choice` table's low IDs double as **type descriptors**: when the
  `choices` column is a bare descriptor word (`numeric`, `string`, `path`,
  `image`, `1darray`), it is the *authoritative* type — even when
  `variabletype` says `choice`. (`variabletype='choice'` + `choices='image'`
  is an image variable streaming raw bytes, not a one-option enum.)
- A blank `variabletype` with a real comma-separated option list infers `enum`.
- A `choice` with no options, more than **16** options, or any option label
  longer than **26** characters cannot be a CA enum (`DBR_ENUM` limits) and
  degrades to a plain string PV — the option text still round-trips verbatim;
  only the dropdown is lost.
- `int` exists as a dtype (served as `ChannelInteger`) but is reachable only
  via an explicit `dtypes=` override — the DB never produces it.
- Metadata: DB `units` → EGU; DB `min`/`max` → **display** limits only, never
  enforced control limits. caproto enforces control limits on write and would
  reject faithful-but-out-of-range readbacks — notably `NaN` from a failed
  online analysis, which the contract requires be *reported*, not clamped.
  GEECS remains the authority on valid set values (§2).

### Enum resolution — readback (string → index), **(PR #452)** order

GEECS speaks the option *string* on the wire; CA speaks the option *index*.
Readback resolution (`channels.enum_index`) is, in order:

1. **Exact label match** — wire text equals a choice verbatim.
2. **Numeric-value match** — if the wire text parses as a number, compare by
   *value* against every choice that itself parses as a number: `"2.000000"`
   matches the label `"2"`. Devices stream numeric option values in arbitrary
   float formatting (DG645-style trigger configs have labels like
   `["1", "2", "5"]`), so a numeric label is matched by **value**, never
   mistaken for an index — treating `"2.0"` as index 2 would silently select
   `"5"`.
3. **Index fallback** — *only* when no choice is numeric (e.g. labels
   `["Off", "On"]` with wire value `"1"`), a numeric wire value is interpreted
   as the option index.

If numeric labels exist but nothing value-matches, the update is **skipped with
a warning** (returns `None`) rather than guessed. Unresolvable values never
move the PV.

Setpoint direction (`enum_geecs_value`): a CA put by index or by label maps to
the GEECS option string, which is what the UDP set carries.

### Verbatim wire-text passthrough

String, path, **and enum** variables reach the gateway as the **exact raw wire
text** — they are excluded from the stream's numeric coercion, which is lossy
for text: `'007'` → `7` → `"7"`, `'1.10'` → `"1.1"`, `'1e5'` → `"100000.0"`.
Contract: **`'007'` stays `'007'`** on a string PV, and a numeric-looking enum
label like `"1.0"` survives verbatim for `enum_index`. Numeric variables keep
the historical coercion (int when integral and undotted, else float).

### Float deadband policy — default 0.0

- The per-variable monitor deadband defaults to **0.0**: every frame whose
  value *changed* posts; only **exact repeats** (and NaN→NaN) are suppressed,
  so a static device costs no CA/archiver traffic while every real change is
  visible. At the ~1–5 Hz GEECS stream rate, bandwidth is a non-issue.
- **The DB `tolerance` field is a *set-convergence* criterion, never a monitor
  deadband.** Through 0.5.0 it was wired as the deadband and suppressed real
  sub-tolerance motion from every value monitor — including recorded scan rows
  and s-files (observed live on a magnet PSU: it moved, the readback froze).
  Fixed in 0.5.1; do not reintroduce. Archive-volume control belongs in an
  archive event mask (MDEL/ADEL split) or the Archiver Appliance's sampling
  policies — see DESIGN.md.
- After a reconnect the suppression cache is cleared, so the **first frame
  always posts** even if unchanged (this also clears INVALID severity).
- A non-zero deadband remains available per-variable (`VariableSpec.deadband`)
  as explicit curation; nothing DB-driven sets one.

---

## 5. Alarm / severity policy

What severity means **today**:

| Condition | PV | Severity / status |
|---|---|---|
| Device TCP subscription live, no configured value alarm active | readbacks | `NO_ALARM` |
| Device TCP subscription live, configured scalar limit crossed | that numeric readback | configured severity (`MINOR_ALARM`, `MAJOR_ALARM`, or `INVALID_ALARM`) with status `LOW`, `LOLO`, `HIGH`, or `HIHI` |
| Device TCP subscription dropped / unreachable | all of that device's **readback** PVs | **`INVALID_ALARM`**, status `COMM` — data is stale, last value retained |
| Same event | that device's `CONNECTED` PV | value `Disconnected`, **`MAJOR_ALARM`**, status `COMM` |
| Recovery (first live frame / reconnect) | both | automatic return to live-value severity (`NO_ALARM` unless a configured value alarm is active) |

So: **INVALID on a readback means "not live", not "bad value"** — the held
value is the last known good one. `CONNECTED` is the explicit liveness signal
for Phoebus/alarm layers; prefer it over inferring liveness from data severity.

Value-based alarms are an explicit curated overlay from the optional
`ca_alarm_limits` MySQL table, keyed by `(experiment, device, variable)`.
Only non-null threshold columns apply. The gateway validates rows at startup,
attaches them only to served numeric readbacks, and treats the table as optional:
if it is absent during rollout, the IOC starts with no value alarms. DB `min` /
`max` remain display metadata and are **never** interpreted as alarm limits.

**Not yet implemented** (be honest with your displays):

- No enum/string bad-state alarms yet; v1 value alarms are scalar limits only.
- No archive deadbands (MDEL/ADEL split) — planned with the Archiver Appliance
  (DESIGN.md "Honest gaps").
- A device merely going *quiet* raises no alarm and does not age severity —
  GEECS devices are legitimately silent for seconds (waiting on triggers, slow
  online analysis). Only an actual socket drop marks INVALID. The known gap —
  hard power-off with the socket left open (no TCP FIN) — is deferred to TCP
  keepalive.
- Setpoint PVs carry no alarm semantics; a failed put raises at the client
  instead.

---

## 6. Collision rules

At `pvdb` build time (startup), every PV registers in the manifest:

- **Exact duplicates are tolerated**: the same `(device, geecs_var, kind)`
  registering the same PV twice is a no-op. The GEECS DB genuinely lists some
  variables more than once (real case: `U_GhostFilters`), and
  `from_db_metadata` also dedupes rows defensively.
- **Genuine collisions raise**: a *different* source mapping to an existing PV
  name (e.g. `Trigger.Source` and `Trigger Source` on one device, both
  normalizing to `Trigger_Source`) raises `ValueError` at startup. Never a
  silent clobber; the gateway refuses to start until the overlay renames one.
- **The status namespace is reserved**: per-device `…:CONNECTED` and the
  `[Experiment:]CAGateway:*` diagnostics go through the same guard, so a GEECS
  variable named `CONNECTED` or a device named `CAGateway` is a startup error,
  not a shadowed status PV.

---

## 7. Failure semantics

### Per-device startup fault tolerance

- One device failing its UDP bind at `connect()` (e.g. an unroutable IP making
  local-interface detection fall back to `""` → `EADDRNOTAVAIL` on PPP/VPN
  links) is **logged loudly and skipped**; the other N−1 devices start
  normally. The skipped device's readbacks still stream over TCP; only its
  setpoint writes fail — with **`GeecsConnectionError`** ("bind failed at
  startup") — until the gateway restarts.
- Devices unreachable over TCP at startup are also non-fatal: the per-device
  supervisor retries with exponential backoff (0.5 s → 30 s cap by default),
  and the PVs sit at INVALID / `CONNECTED=Disconnected` until the device
  appears. Down/up transitions are logged once per episode, not per attempt.
- Whole-experiment config build (`from_geecs_experiment`) skips (with a
  warning) any device whose spec cannot be built from the DB.

### UDP reply correlation — and the uncorrelatable ACK

The GEECS set/get exchange is command → ACK (same port) → exe reply (port+1).
Exchanges on one device are serialized by a lock, but a **late** exe reply from
a timed-out exchange can arrive during the next one. Exe replies name their
variable (field 2, either bare `Var` or the hardware's command echo
`getVar`/`setVar` — both accepted), so each exchange only accepts replies
matching its own variable; everything else — including malformed replies with
fewer than four `>>` fields, which carry no attributable variable — is logged
at warning level and **discarded**, and a genuinely lost reply surfaces as an
exe timeout rather than a mis-attributed value.

**Documented limitation**: the ACK is a bare `accepted`/`ok` token with no
identifying field and *cannot* be correlated. A stale positive ACK from an
abandoned exchange is indistinguishable from the real one — and harmless, since
the real ACK is then dropped as stale and the exe stage still correlates by
variable. A stale *negative* ACK could spuriously fail an exchange that would
have succeeded; at GEECS's one-command-at-a-time pace this is accepted risk.

### TCP push-frame parsing — anchored on subscribed names

Push frames are `Dev>>shot>>var1 nval,val1 nvar,var2 nval,val2 nvar…`. Values
may contain commas (paths), `>>`, even newlines — so the frame is **not**
comma-tokenized. The parser anchors on the *subscribed variable names* (known
from the DB, never containing commas) followed by the literal ` nval,` token,
taking the value non-greedily up to the ` nvar` at a genuine pair boundary;
longer names are tried first so a name that is a prefix of another cannot
shadow it. Unsubscribed pairs in the frame are skipped cleanly.

**Documented residual ambiguity**, inherent to the wire format: a value that
itself contains the full boundary text `` nvar,<comma-free-text> nval,`` is
indistinguishable from a real pair boundary and will be truncated there. (The
legacy GEECS-PythonAPI parser truncates at the *first* ` nvar` regardless, so
this parser is strictly more tolerant.)

### Stream-value fan-out failures

A frame value that cannot be coerced to its PV's dtype (a DB type mismatch), or
an enum value that does not resolve, is skipped with a **once-per-variable**
warning (not ~5 Hz spam); the PV keeps its previous value. A callback exception
never kills the listener.

---

## Pinned-by test map

All in `GeecsCAGateway/tests/` unless noted. Rows marked *(PR #452)* land with
that branch and are part of this contract's target behavior.

| Contract claim | Pinned by |
|---|---|
| Component normalization (dot, spaces, runs, strip) | `test_naming.py::test_dot_becomes_underscore`, `::test_spaces_collapse_to_underscores`, `::test_mixed_bad_chars_collapse` |
| `[Experiment:]Device:Variable` assembly, prefix optional, overrides | `test_naming.py::test_pv_name_for_with_experiment_prefix`, `::test_pv_name_for_without_experiment`, `::test_variable_spec_explicit_pv_wins`, `::test_device_prefix_defaults_to_name`; `test_pv_contract.py::test_pv_name_drops_falsy_parts_and_normalizes` |
| Manifest as authoritative map; `:SP` only when settable | `test_gateway.py::test_experiment_prefix_and_manifest`, `::test_pvdb_contains_readback_and_setpoint` |
| Genuine collision raises; exact duplicates tolerated | `test_gateway.py::test_pv_name_collision_raises`; `test_config_from_db.py::test_from_db_metadata_dedupes_duplicate_variables` |
| Reserved `CAGateway`/status namespace guarded | `test_gateway.py::test_pvdb_has_connected_and_gateway_status_pvs`; `test_pv_contract.py::test_status_pv_namespace_is_collision_guarded` |
| `CAGateway:RESTART` triggers clean shutdown; `Idle` is a no-op; exit code 86 | `test_pv_contract.py::test_restart_pv_requests_clean_shutdown`; `test_gateway.py::test_run_returns_true_on_restart_request`; `test_entrypoint.py::test_main_exits_with_restart_code` |
| Readbacks client-read-only; setpoints writable | `test_gateway.py::test_readback_channels_deny_client_writes` |
| Stream → readback; caput `:SP` → GEECS → readback | `test_gateway.py::test_stream_updates_readback`, `::test_setpoint_write_reaches_geecs` |
| Failed GEECS set ⇒ CA put fails, value not stored | `test_pv_contract.py::test_setpoint_put_failure_leaves_value_unstored` |
| 30 s configurable set budget; 10 s get budget | `test_gateway.py::test_setpoint_write_uses_move_budget_timeout`, `::test_set_timeout_is_configurable`, `::test_get_uses_standard_exe_timeout` |
| Timestamp ladder, LabVIEW→Unix, implausible rejected | `test_gateway.py::test_extract_timestamp_converts_labview_to_unix`, `::test_extract_timestamp_ladder_prefers_first_present`, `::test_extract_timestamp_none_when_absent_or_implausible`; `test_config_from_db.py::test_timestamp_ladder_default_prefers_acq_then_sys` |
| PV stamped with device time; timestamp PVs carry raw LabVIEW value | `test_gateway.py::test_pv_timestamp_from_systimestamp`, `::test_timestamp_vars_exposed_as_pvs_with_raw_value` |
| `0.0` pre-acquisition placeholder | `test_pv_contract.py::test_float_readback_initializes_to_zero_placeholder` |
| Frame ordering: data before timestamps *(PR #452)* | `test_gateway.py::test_callback_posts_timestamp_variables_last` |
| DB type mapping, descriptors, enum degradation, blank-type inference | `test_config_from_db.py::test_from_db_metadata_maps_variable_types`, `::test_choice_pointing_at_type_descriptor_is_skipped`, `::test_blank_variabletype_inferred_from_choices`, `::test_choice_without_options_falls_back_to_string`, `::test_choice_exceeding_ca_enum_limits_falls_back_to_string` |
| Long-string path PVs (>40 chars round-trip both directions) | `test_channels.py::test_cast_path_decodes_char_arrays`, `::test_path_readback_holds_long_string`, `::test_path_setpoint_forwards_full_text` |
| Enum label↔index both directions over the gateway | `test_channels.py::test_enum_index_maps_label_to_index`, `::test_enum_geecs_value_index_and_label`; `test_gateway.py::test_enum_readback_and_setpoint` |
| Enum numeric-label value-match, no index guess *(PR #452)* | `test_channels.py::test_enum_index_numeric_labels_match_by_value_not_index`, `::test_enum_index_numeric_labels_no_value_match_is_none`, `::test_enum_index_index_fallback_only_without_numeric_labels` |
| Verbatim text passthrough (`'007'`, `'1.10'`, `'1e5'`) | `test_transport.py::TestTcpSubscriber::test_string_value_reaches_callback_verbatim`, `::TestSubscriptionFrameParsing::test_string_dtype_values_pass_through_verbatim` |
| Deadband 0.0 from DB (`tolerance` never a deadband) | `test_pv_contract.py::test_db_tolerance_is_not_used_as_monitor_deadband` |
| Zero deadband: every change posts, exact repeats suppressed | `test_pv_contract.py::test_zero_deadband_posts_changes_suppresses_exact_repeats` |
| Explicit non-zero deadband suppression | `test_gateway.py::test_deadband_suppresses_small_changes` |
| Display (not control) limits; NaN readback reported | `test_config_from_db.py::test_pvdb_built_from_db_spec_has_limits`; `test_gateway.py::test_nan_readback_accepted_despite_limits` |
| Optional `ca_alarm_limits` table; curated scalar alarms attach only to served numeric readbacks | `test_geecs_db.py::test_get_ca_alarm_limits_returns_validated_rows`, `::test_get_ca_alarm_limits_missing_table_is_fail_open`; `test_config_from_db.py::test_alarm_limits_validate_order_and_presence`, `::test_from_geecs_experiment_attaches_numeric_alarm_limits` |
| Value alarm severity/status on live readbacks; disconnect INVALID wins until next live frame | `test_gateway.py::test_value_alarm_limits_set_live_readback_severity`, `::test_invalid_liveness_overrides_value_alarm_until_live_frame` |
| INVALID on drop, auto-recovery; CONNECTED MAJOR while down | `test_gateway.py::test_reconnect_and_validity`, `::test_set_connected_updates_pv_severity_and_count` |
| Per-device bind-failure tolerance; clean caput error afterward | `test_gateway.py::test_one_device_bind_failure_does_not_abort_startup`, `::test_setpoint_write_without_udp_client_raises_cleanly` |
| UDP reply correlation, echo form, malformed/stale discard | `test_udp_reply_correlation.py` (whole module) |
| TCP parse anchoring: commas/`>>` in values, prefix names, unsubscribed pairs | `test_transport.py::TestSubscriptionFrameParsing` (whole class) |
| Uncoercible value: skip + warn once | `test_gateway.py::test_uncoercible_value_warns_once_and_skips` |
