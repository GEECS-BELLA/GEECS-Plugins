# Native EPICS shot control — design note

Status: **proposal, nothing built.** How GeecsBluesky drives the trigger
box (an SRS DG645) through its own EPICS IOC instead of the LabVIEW device
`U_DG645_ShotControl` behind the CA gateway. This is a gate for an
all-EPICS scan (native box + native cameras + native scope). It need not be
used yet; the aim is that the switch is **configuration, not development**.
Companion to `NATIVE_DEVICES.md` (cameras, the registry, one owner);
`Planning/epics_estate.md` (the "shot control" row: a `dg645` instance,
owned by the worker during a run, ported last) is the estate plan.

Cites: `file:line` against `master` here; `geecs-epics:<path>` against
the private `GEECS-BELLA/geecs-epics` `main` (cited by path only, per the
root `CLAUDE.md`). Record names are upstream `epics-modules/delaygen`'s
`drvDG645.db`, as the geecs-epics bench instance loads and audits them.
**Where a record's semantics are not proven in geecs-epics, this note says
so**: none of them is taken from memory.

## 1. What `ShotControl` needs from any trigger box (as built)

`ShotControl` (`devices/shot_control.py`) is box-agnostic already: it
replays ordered `(device, variable, value)` triples per state and knows
nothing about a DG645.

| Need | How it is done today | Cite |
|---|---|---|
| States | `OFF`, `STANDBY`, `SCAN`, `ARMED` (standing) and `SINGLESHOT` (momentary, never recorded as standing) | schema `trigger_profile.py:79-105`; `shot_control.py:224-226,321-322` |
| Writes per state | ordered `TriggerWrite{device, variable, value}` lists, values verbatim wire strings; adapted to `ShotControlWrites` | `trigger_profile.py:108-131,253-261`; `shot_control.py:129-165` |
| Transport | one cached setter per `(device, variable)`, a gateway `:SP` put with completion (`CaPutSetter`); `setter_factory` is already an injection seam | `shot_control.py:188-215`; `devices/ca/gateway_put.py:1-24` |
| Liveness | one `CONNECTED` signal per profile device, read by `liveness_gate` before the first move (fail-open, only `Disconnected` counts) | `shot_control.py:216-223,273-276`; `plans/registry.py:164-209` |
| Run bracket | strict opens `ARMED`, gated opens `OFF`, both close `STANDBY` | `plans/registry.py:648-655` |
| Fire one shot (strict) | `trigger` every detector (each baselines its stamp), then `mv(shot_control, "SINGLESHOT")` as a waited move, then await each detector | `plans/strict.py:194-223,320-327` |
| Refused fire | propagates untouched, never re-fired (re-firing could add physical shots) | `plans/strict.py:200-211` |
| Missed frame | row kept with empty columns, up to `max_refires` more shots | `plans/strict.py:397-445` |
| Throttle | optional `shot_period` sleep before the triggers | `plans/strict.py:307-318` |
| Pause | from `SCAN`/`STANDBY` (`QUIESCE_FROM`) drive `OFF`, remember the state; `ARMED` needs nothing; never raises | `shot_control.py:58-60,336-367` |
| Resume | re-assert the remembered state | `shot_control.py:369-382` |
| Gated hold | `hold_for_batch(hook)`: the pause calls the hook, the plan re-arms and drives `SCAN` itself | `shot_control.py:325-334`; `plans/gated.py` |
| Box-less | a scan-mode profile with no writes: every state a logged no-op, the strict loop skips the fire | `shot_control.py:263-271,304-313`; `plans/strict.py:305,326` |
| Learning a shot happened | **not from the box.** Each detector's own signal: `GeecsDetector` = `acq_timestamp` advancing past a baseline taken synchronously in `trigger()`; `NativeCamera` = the HDF plugin's `NumCaptured` advancing | `devices/detector.py:5,182-291,560-574`; `NATIVE_DEVICES.md:144-152` |
| Gated shot count | the shot clock: an essential `GeecsDetector`'s `acq_timestamp` | `plans/gated.py:86-122` |

The profiles in use write three LabVIEW variables of the box —
`Trigger.Source`, `Trigger.ExecuteSingleShot`, `Amplitude.Ch AB` (the gas
jet gated by the AB amplitude, `trigger_profile.py:43-48`). The sim-bench
transcript shows the HTU laser-off profile's shape: ARMED = `Amplitude.Ch
AB 4.0` then `Trigger.Source Single shot`; each fire `ExecuteSingleShot on`;
STANDBY = `0.5` then `External rising edges`
(`geecs-epics:docs/runbooks/sim-bench.md`, the acceptance transcript).

So a native box must supply only: **a put per role with an honest
completion**, **a liveness verdict**, and the same values per state. The
state machine, pause and gated hold stay as they are.

## 2. The DG645 IOC's records against those needs

Source: `geecs-epics:services/undulator/timing/dgtest1/` — a bench IOC on
the `ioc-timing` image (`delaygen.DG645` entity, `config/ioc.yaml:53`) for a
**second** DG645, *not* the shot-control box, which geecs-epics does not
touch (`geecs-epics:docs/runbooks/ioc-host.md:484-492`; readiness:
"`U_DG645_ShotControl` goes last", `geecs-epics:docs/readiness.md:30`).

| Need | Record (`<prefix>` + …) | Evidence | Proven? |
|---|---|---|---|
| Trigger source | `TriggerSourceMO` (mbbo) / `TriggerSourceMI` | `config/operator.db:155`; readback labels `Internal, Ext rising edge, Ext falling edge, SS ext rise edge, SS ext fall edge, Single shot, Line` (TSRC 0..6), `check_readbacks.py:36-37` | readback labels checked on hardware; **setpoint labels assumed equal** (same mbbo pair) — verify |
| Single-shot fire | `TriggerDelayBO` (bo) | operator list "the single shot (`TriggerDelayBO`)", `opi/README.md:44`; upstream screen button "TRG", `opi/srsDG645Trigger.bob:968-974`; the SCPI fake models `*TRG` (`sim/dg645_sim.py:182-183`) | **not proven**: what the record sends, and whether its put completion means "sent" or "fired", must be read from `drvAsynDG645` source and shown on the fake's transcript |
| AB amplitude (gas jet) | `ABOutputAmpAO` / `ABOutputAmpAI` | `config/operator.db:143`; LAMP?1 readback, `check_readbacks.py:101-104` | readback proven; write not yet exercised |
| Inhibit (alternative quiesce) | `TriggerInhibitMO` / `MI`: `Off, Triggers, AB, AB,CD, AB,CD,EF, AB,CD,EF,GH` | `config/operator.db:158`; `check_readbacks.py:38` | readback proven |
| Rate (internal) | `TriggerRateAO` | `config/operator.db:157` | readback proven |
| Status after a write | `StatusLI` (`LERR?`), `StatusSI` (`STATUS OK`) | `check_readbacks.py:122-123`; a refused delay is drained into `StatusSI` (`docs/runbooks/ioc-host.md`, phase B) | proven for a delay write |
| Readback freshness | **no poll**: a readback is the boot reading until its record processes; each setpoint re-reads its own readback after a write | `config/ioc.yaml` header; `ioc-host.md:515` | proven |
| Liveness | **no CNCT-style record identified.** A busy/absent port at boot is final: every PV INVALID, no reconnect (`ioc-host.md:517`) | — | open: candidate verdict = `IdentSI` unreadable or INVALID |
| Write access | today only the maintainer's account (`config/access.acf`, ASG OPERATOR) | — | the worker's CA user would be refused |
| Run lock | the camera chain's `RUN:Lock`/`RUN:Owner` (`NATIVE_DEVICES.md:126-143`) does not exist on this IOC | — | to add (estate plan: "owned by the worker during a run") |
| Hazards | a **delay** write can abort the delay cycle on every output (`ioc-host.md:585`); one TCP client only (`ioc-host.md:494`) | — | whether source/amplitude writes disturb a cycle: unknown |

Value vocabulary differs: the LabVIEW device says `External rising edges`
(`geecs-epics:sim/trigger_box_sim.py`, `SOURCES`), the IOC says `Ext rising
edge`. A native profile is a new file anyway, so it carries the IOC's
labels verbatim (the schema's "verbatim wire strings" rule).

## 3. The seam

| Option | Shape | For | Against |
|---|---|---|---|
| (a) Profile names PVs | `{pv: "<expt>:<dev>:TriggerSourceMO", value: …}` | no new vocabulary | record names and prefixes in configs (the prefix is derived by geecs-epics decision 5; it would drift); no check that a PV is writable or a fire; liveness has no device to judge; breaks "the device name is the one identity" |
| (b) `NativeShotControl` beside `ShotControl`, picked by a profile `kind` | a second device class | each class simple | duplicates the state/pause/hold/box-less logic (~200 lines, the parts with the subtle rules); a profile is all-native or all-gateway, so a native box plus a LabVIEW gas-jet controller cannot share a transition |
| (c) Roles + a family file | a write names a registry device and a **role**; the `dg645` family maps role → record | one state machine; per-write transport; validated against the family; mixed profiles work; matches GeecsPanels/Canopy families | needs the `dg645` family file and a registry kind |

**Recommended: (c), inside the existing `ShotControl`.** The transport is
already a seam (`setter_factory`, `shot_control.py:195-202`); what is new is
choosing it per device: a profile device that is in `native_devices.yaml`
gets a native setter and a native liveness signal, any other device keeps
the gateway `:SP` put. No new class, no profile `kind`.

What changes:

| Where | Change |
|---|---|
| **GEECS-Schemas** | `TriggerWrite` **unchanged** (`device`, `variable`, `value` are strings; for a native device `variable` reads as a role). New family file `epics/families/dg645.json` (kind `setpoint_device`, the estate plan's unbuilt `dg645` row): roles `trigger_source`, `fire` (type `command`, `confirm: hazard`), `ab_amplitude` (and the other outputs), `inhibit`, `rate`, `status`, `ident`. Registry: a `trigger_box` entry kind (or `setpoint_device` with `template: dg645`; open question 6). A validator helper: a profile write to a registry device must name a writable role of its family |
| **GeecsBluesky — device** | `ShotControl.__init__` takes the registry (or a resolver of device → family); per native device: a `NativePutSetter` (raw `caput(wait=True)` — never `signal.set()`, #868, `gateway_put.py:14-17` — then confirm the role's readback equals the value and `StatusLI` is 0), and a liveness signal from the family's `ident`/`status` roles |
| **GeecsBluesky — load** | `TriggerProfiles.from_resolver` (`plans/registry.py:105-135`) passes the registry; a native write to an unknown role, or an IOC label outside the record's choices (checked at connect, before any write), skips the profile loudly as today |
| **Worker liveness gate** | `liveness_gate` judges native box devices by the native rule — **unreadable = down**, like `down_native_cameras` (`plans/registry.py:186-188`; `devices/native.py:691`) — not the gateway's fail-open rule. `ShotControl` exposes the two sets apart (`liveness_signals` stays gateway-only) |
| **Client preflight** | `qs_client/submit_preflight.py:140,188,385` reads every profile device's gateway `CONNECTED`; for a native device that PV does not exist and the probe fails open, silently. It must skip native devices (the worker gate judges them) or probe the native verdict |
| **Strict** | unchanged: the fire stays `mv(shot_control, "SINGLESHOT")`; a refused native fire raises the same `FailedStatus` |
| **Ownership** | the box takes the run lock at the bracket's first move and releases it after `STANDBY` (the `RunLockIO` pattern, `NATIVE_DEVICES.md:126-143`), if the IOC grows those records (open question 1) |

**Does a native box need scan mode?** No. Scan mode is the LabVIEW
devices' protocol (`ScanStart`, per-shot path announce, DB start/end
values); the IOC never acknowledges `ScanStart` (`native_scope.py`,
`NOT_SCAN_MODE`). A native box may still sit under a scan-mode profile that
announces to LabVIEW devices, with these changes:

| Scan-mode code | Today | With a native box |
|---|---|---|
| `run_device_names` (`plans/scan_mode.py:66-85`) | adds every `liveness_signals` device to the `ScanStart` list | native box devices excluded (else the roster check at `:287-294` refuses the run: the box is no roster device) |
| `boundary_conflicts` (`:128-148`) | compares profile writes with the device's DB start/end rows | nothing to compare (no DB row); naturally empty — pin by a test |
| `check_gateway_host` (`:169-196`) | the fire leaves from the gateway's host | still needed for the LabVIEW devices; the native fire goes worker → IOC directly, so the box adds no constraint |
| Box-less profile | the announce is the shot | unchanged |

## 4. Shot detection in an all-native run

There is no LabVIEW `acq_timestamp` anywhere. What exists and what does not:

| Question | Today | All-native |
|---|---|---|
| Who says a shot happened | each detector, against its own baseline (§1) | each `NativeCamera`: `NumCaptured` advances; the box says nothing |
| Row stamp | each device's `acq_timestamp` column | each camera's EPICS frame stamp pair (`NDArrayEpicsTSSec/nSec`, the IOC host's clock at frame arrival), carried in the LabVIEW epoch (`NATIVE_DEVICES.md:179-193`) |
| Join key validity | GEECS hosts' clocks | IOC hosts NTP-disciplined (chrony), the worker's clock too (`geecs-epics:docs/runbooks/timing.md`) |
| The fire's own time | not recorded | **missing.** Proposal: a plan-side readable like `BinCounter` (`plans/strict.py:453-480`) stamping the fire's completion on the worker clock, one column per row |
| Box-side confirmation | gateway put completes when GEECS accepts the set | **missing**: the record set has no poll and no trigger counter is read by any record (INSR's TRIG bit is read by none that geecs-epics audits). Put completion + `StatusLI` 0 is the only signal |
| Baseline race | `GeecsDetector` baselines synchronously in `trigger()` (`detector.py:153-154`, CLAUDE.md) | `NativeCamera` on a hardware line reads its baseline in a task racing the fire, "safe in practice" because the gateway fire takes ~200 ms (`NATIVE_DEVICES.md:146-150`). A direct IOC fire is faster — **this margin shrinks**; make the native baseline synchronous before an all-native run |
| Missed frame | refire | a native camera's miss fails the step (`devices/native.py:658-674`); all-native strict has no refire until the Tiled consolidator honours the index map (`NATIVE_DEVICES.md:176-178`) |
| Gated | shot clock = a `GeecsDetector` | refused with only native essentials (`plans/gated.py:118-122`; ruling 3, `NATIVE_DEVICES.md:531-534`) |
| Scope | LabVIEW scope with `acq_timestamp` | native scope: design only (`NATIVE_DEVICES.md` §9) |
| Scalar devices | LabVIEW devices stamp each read | native motors/supplies are readbacks, no per-shot stamp — read as snapshots per row |

So the box itself adds one gap (no confirmation beyond put completion); the
others belong to the camera/scope arcs. An all-native **strict** run of
native cameras + a native box is possible after slices S1–S4 plus the
synchronous native baseline; gated waits for a native shot clock.

## 5. Cutover as configuration

What a switch is, once the code exists:

| Step | Where | Who |
|---|---|---|
| 1. The box's IOC instance (prefix `<experiment>:<device>:` by decision 5, the device's own name), access group including the worker's CA user, the run-lock records | geecs-epics `services/<experiment>/timing/<device>/` | agent prepares, maintainer starts |
| 2. Registry entry for the box (`native_devices.yaml`) | configs repo | agent PR |
| 3. A native trigger profile per operating condition (e.g. `<name>-native`), values in IOC labels | configs repo | agent PR, maintainer review |
| 4. The scan request / `experiment_defaults.trigger_profile` names the native profile | configs repo or per scan | maintainer |
| 5. The LabVIEW device **stopped in Master Control**, the gateway's `connected` for it a clean 0, then the IOC started under the `connected` watcher | lab | maintainer |
| 6. The device disabled in the GEECS DB, gateways restarted, archive set re-derived (the `NATIVE_DEVICES.md` §2 migration); configs naming its LabVIEW variables (scan variables over its delays, actions, presets) moved or retired | DB + configs | maintainer |

**One owner.** The DG645 serves one TCP client with no independent
sessions (`ioc-host.md:494-505`): the LabVIEW device and the IOC must
never both connect. The guard is procedural (step 5, IOC stopped **before**
LabVIEW restarts; the watcher is a backstop with ~30 s latency) plus, once
step 6 is done, the namespace's existing refusal of a registry name that
is also a roster device (`NATIVE_DEVICES.md:82-85`). Until step 6 the
profiles naming the LabVIEW device still load; the liveness gate refuses
them only because `CONNECTED` reads `Disconnected` — acceptable for a
bench period, not as the end state.

**Same prefix, two servers.** The gateway serves the LabVIEW device under
the same `<experiment>:<device>:` prefix the IOC will use (different
suffixes). Between steps 5 and 6 both answer under one prefix; the
watcher's `connected` read must keep working through it. Name it in the
migration checklist.

**Rollback** = steps 4 → 5 reversed: name a LabVIEW profile, stop the IOC,
start the LabVIEW device, re-enable it in the DB.

## 6. Slices, tests, hardware

| Slice | Repo | Content | Size (LOC code / tests) |
|---|---|---|---|
| S0 | geecs-epics | read `drvAsynDG645`/`drvDG645.db` source: what `TriggerDelayBO` sends and when its put completes; mbbo setpoint labels; a liveness record. Prove each on the SCPI fake (`sim/dg645_sim.py`): the IOC on the fake, one put per record, the fake's `WRITE` transcript. Record on #16 | ~0 / ~80 |
| S1 | GEECS-Schemas | `dg645` family file + registry kind + the "writable role" helper + docgen | ~120 / ~150 |
| S2 | GeecsBluesky | per-device transport in `ShotControl` (`NativePutSetter`, readback + `StatusLI` confirm), native liveness in `liveness_gate`, registry-aware profile load | ~200 / ~300 |
| S3 | GeecsBluesky | scan-mode exclusion, client preflight skip/probe, fire-stamp column (if ruled in) | ~80 / ~150 |
| S4 | geecs-epics | sim acceptance: the box IOC on the SCPI fake beside the camera fake; `examples/sim_worker_count.py` with a native profile; the fake's transcript = ARMED writes, N fires, STANDBY writes; `--fault` paths (fake refuses the fire; fake port closed → liveness refusal before the claim) | ~120 / script |
| S5 | GeecsBluesky | synchronous baseline for `NativeCamera` on a hardware line (§4) | ~60 / ~100 |
| S6 | lab | hardware, below | — |

**Tests (GeecsBluesky, hermetic, mock CA):** a mixed profile (native box +
gateway device) writes in declared order across both transports; a native
write whose readback disagrees fails the move naming the role; `StatusLI`
non-zero fails it; pause from `SCAN` drives native `OFF` and resume restores;
`hold_for_batch` unchanged; unreadable native liveness refuses before the
claim (no folder); a refused native fire is not re-fired; a scan-mode run
leaves the native box out of `ScanStart`; an unknown role or out-of-choices
label skips the profile at load. Each test proven to bite (break the code,
watch it fail).

**Sims:** `geecs-epics:sim/dg645_sim.py` is the instrument's SCPI port —
the native path (IOC → fake). `geecs-epics:sim/trigger_box_sim.py` is the
LabVIEW device's *gateway* surface — the current path; it stays the
regression for profiles that keep the gateway.

**Hardware protocol** (`geecs-epics:docs/runbooks/hardware-tests.md` is the
owner; this is the order for this device):

1. **Laser off**; nothing downstream that an extra edge can hurt. The
   maintainer's hand-off is the gate.
2. Prove first on the bench DG645 (not the shot-control box) if the
   maintainer rules its outputs safe to fire (open question 5); the
   shot-control box goes last.
3. LabVIEW device stopped in Master Control; the gateway's `connected`
   reads 0; a queries-only snapshot (`dg645_snapshot.py`) before.
4. DG645 warm-up per the maintainer's procedure before any timing check.
5. IOC started under the watcher. **The maintainer runs every CA put**
   (agents are denied puts in auto mode): the agent hands over a prepared
   list — ARMED's writes, one fire, readbacks, STANDBY's writes — and
   reads back.
6. One strict `count` of 10 through the worker with a line-triggered
   camera: 10 fires, 10 frames, rows joined by stamp; a pause mid-run
   drives `OFF` and resumes.
7. IOC stopped, snapshot after equals before except the intended settings;
   LabVIEW restarted only after the IOC is down.

## 7. Open questions for the maintainer (ranked)

1. **Who may write the box during a run?** The IOC's access group holds
   only your account today. Add the worker's CA user, and a run lock
   (`RUN:Lock`/`RUN:Owner` as on the cameras) so panels cannot write
   mid-scan — yes/no?
2. **The fire record.** `TriggerDelayBO` is labelled "TRG" upstream; S0
   will prove from source and the fake what it sends and what its put
   completion means. Is put completion + `StatusLI` 0 enough confirmation
   of a shot, or do you want a box-side counter (INSR polled) first?
3. **A per-row fire stamp** (worker clock at fire completion) for runs with
   no LabVIEW stamp — worth a column, or rely on the cameras' EPICS stamps?
4. **Profile values:** IOC labels verbatim (recommended) or LabVIEW labels
   translated by the family?
5. **Bench box for the fire test:** may the second DG645's outputs take
   single shots with the laser off, or does the first fire wait for the
   shot-control box itself?
6. **Registry kind:** a dedicated `trigger_box` kind, or `setpoint_device`
   with `template: dg645` (the estate plan's row)?
7. **Native box under a scan-mode profile** (LabVIEW devices announced, box
   native): needed, or refuse the combination for v1?
8. **Delay scans:** scan variables over the box's delays go through the
   gateway today; after cutover they need the native setpoint device
   (`NATIVE_DEVICES.md` §8). Same arc or later?
