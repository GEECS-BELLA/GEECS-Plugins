# Clean-room rebuild: GeecsBluesky as a native Bluesky application

**Status: proposed direction, not started.** Written 2026-09-09 at the end of
the session that built #809, as a handoff to a fresh session. Read this
before `00_overview.md`, because it supersedes that document's phase plan.

Sam's framing, which is the point of the whole document:

> Let's start with the assumption that we *only* have our CA gateway, PVA
> gateway and bluesky/ophyd devices. How would we go from there? Blow up
> everything except the bluesky/ophyd fundamentals and build a solid RE
> that achieves what we want. […] Don't try to hold on to *anything* if it
> doesn't slot in completely cleanly.

And the constraint on that:

> holding on to the requisite GEECS things, like DB as source of truth,
> s-files etc.

**The governing fact, and the reason this is worth doing now** (Sam,
2026-09-09):

> me and my team are the only users of all of this code. My goal is to make
> it 'great' before trying to deploy at other facilities so that I don't
> have to deal with these issues of backward compatibility. We are in a
> unique 'clean slate' phase of development.

There is no external user, no deprecation cycle and no migration burden.
Backward compatibility is **not** a design input. Anything in this
repository may be deleted outright rather than adapted, and the cost of a
wrong turn is a rewrite, not a broken facility. Design for the shape that
is right in five years, not the one that is reachable in small steps.

---

## 1. How to use this document

1. Read §3. It is the evidence, and it is the only part that is hard-won.
   A fresh session given only the target architecture will re-litigate the
   design; a fresh session given the failure catalogue will not.
2. Treat §7 as a contract: anything marked **ASSUMED** must be verified
   before it is designed on. The previous session asserted API details from
   memory more than once and had to retract them publicly.
3. §8 has the sequencing question. It is the one real decision, and it is
   Sam's.

---

## 2. Where things actually stand (verified 2026-09-09)

| thing | state |
|---|---|
| `feature/native-bluesky-plans` | integration branch off master; **#808 merged** into it 2026-09-09 (device namespace, phase 1) |
| #809 `phase/02-preamble-preprocessor` | **OPEN, 13 commits, GeecsBluesky 0.79.0, CI green, not merged.** This is the code this document argues is scaffolding |
| #806 image writing | **OPEN, not started.** File plugin in GeecsPvaGateway + stock ophyd-async detectors; capture daemon retired |
| #807 | the plan of record; its six-then-three phase plan is what this document supersedes |
| the worker (`geecs-gw`) | on **master**. Nothing from #809 is deployed, so none of its open defects is a live hazard |
| hardware acceptance | scans **63 and 64** on 26_0909 via #809's preprocessor: stock `bp.count` as a noscan, stock `bp.list_scan` sweeping `U_S1H:Current` −1→+1 A at 0.5 A. Both in the Tiled catalog with s-files. Scans 56–62 are disposable artifacts of a `tiled=False` run |

**What survives from the work so far, unconditionally:** the device
namespace (#808). Every device of the experiment as a long-lived
ophyd-async noun built from the DB roster, lazily connected. That is
exactly the foundation the design below needs, and it is already merged.

---

## 3. What we learned, and why it changes the plan

#809 put the GEECS scan preamble into a RunEngine preprocessor so that a
stock `bluesky.plans` verb could run a full GEECS scan. It works, on
hardware. It was then reviewed adversarially twice.

**Sixteen findings in the first pass, five more in the verification pass.**
Classified by cause, not by severity:

### Artifacts of describing one scan twice (12 of 21)

The plan says what to read and where to move. The ScanRequest says the same
things through a save set and an axis list. The preprocessor's real job had
become reconciling the two, and each of these findings is a missing
reconciliation rule:

- the plan need not read the devices the save set enables saving on, so
  saved frames had no event row to join to
- the request's shot count and the plan's point count could disagree, and
  ScanInfo recorded the request's
- the request's positions and the plan's positions could disagree
- a first attempt to check that compared membership, not the sequence
- background telemetry was prepared, connected and advertised in the start
  document, but a stock plan reads only its own detectors, so no telemetry
  column was ever emitted
- once injected, the telemetry group was read **unstaged**, which
  `GeecsBluesky/CLAUDE.md` already documents as the 0.7 s per row regression
  that made 1 Hz scans run at 0.5 Hz
- the scan motors were missing from the s-file header map
- the namespace contributed headers for settable children no event carries
- long-lived namespace devices kept the previous run's saving mode, save
  path and asset definitions, so an unrelated later plan emitted asset
  documents pointing into the previous scan's folder
- and still did, on any pre-claim failure after the save set was applied
- scan axes silently lost `confirm:` and `kind: motor` topology, because
  the namespace builds children from the DB and the catalog describes them
  separately
- `kind: motor` disagreement between the two doors is still unresolved and
  was waived

### Real defects, independent of the architecture (9 of 21)

Save-on ordering versus arming, the missing refire on the stock door, the
refire then recursing through the mutator and firing extra shots, the
stream-blind hooks, `install` re-deriving `connect_on_demand`'s kwargs, the
missing `scan.log`, the missing eager save-off, the raw request riding into
every start document, and a set of test-quality items.

### The conclusion

Twelve of twenty-one findings have one cause. Fixing them individually
produces a reconciliation engine: a growing set of rules asserting that two
descriptions of the same scan agree. The parity test added in #809 exists
precisely to detect divergence between the two doors, which is a useful
test and also an admission.

The native answer is that there is only one description. Sam:

> save sets are really just readbacks […] and the scan variables just the
> settables with aliases. […] If you encounter a weird hack or workaround
> to accommodate save sets we should think, "what is a save set doing and
> how does Bluesky solve the same problem?"

---

## 4. The target architecture

Assume only the two gateways and ophyd-async.

**One namespace of devices, from the DB.** Anything that produces
non-scalar data is a `StandardDetector` whose data logic writes through the
PVA gateway file plugin (#806). Everything else is a `StandardReadable`
over CA gateway PVs. Settables are children of their parent device. A
device knows its own topology: a motor is a motor because the DB gives it a
tolerance, not because a scan said so. **This layer is #808 and already
exists**; what changes is that cameras become detectors.

**The trigger box is a flyer.** At HTU the box is the master clock. The
Bluesky expression for "hardware fires, detectors collect N frames" is
`prepare(TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE, …))` then
kickoff / complete / collect, with a `StandardFlyer` wrapping a
`FlyerController`. Strict single-shot and free-run are then the same
implementation with different counts: strict is one event per step,
free-run is continuous. Today they are two subsystems, and free-run is
already slated for deletion.

**Everything persistent is a callback.** ScanInfo, the s-file, the Tiled
catalog entry and the legacy column headers are all functions of the
document stream. None belongs in a plan or a preprocessor. The s-file and
catalog already work this way; ScanInfo does not.

**The scan folder is a `PathProvider`.** ophyd-async ships
`YMDPathProvider` and `AutoIncrementingPathProvider`, which between them
are close to the `scans/YY_MMDD/ScanNNN/` convention. The day-scoped claim
protocol is the part that stays ours.

**Save sets and scan variables move to the client.** "Record amp4in" is a
genuinely useful operator preset. It is a named list that a client expands
into a plan's `detectors` argument before submission. It should never reach
the worker as an instruction. This is what #807 phase C already said; #809
did that job in the wrong place.

---

## 5. The mapping

| GEECS today | Native replacement |
|---|---|
| save set, as a device list | the plan's `detectors` argument; a client-side preset |
| save set `synchronous` flag | the `Triggerable` protocol |
| `save_nonscalar_data`, `localsavingpath`, `save` | the detector's data logic, opened and closed per run |
| save set explicit scalar list | the device's own readables |
| save-set rituals, setup/closeout | plan stubs and `finalize_wrapper` (#647) |
| `background_telemetry` | monitors, or simply more detectors in the list |
| scan variable alias | the namespace attribute (`U_S1H.current`) |
| `kind: motor`, `confirm:`, pseudo | the device class, chosen once at namespace build |
| trigger profile states | a `FlyerController` |
| strict single shot | fly, one event per step |
| free run | fly, continuous |
| Gate-2 save windowing | the detector's open/close window |
| `acq_timestamp` as the shot join key | StreamDatum indices |
| `shot_id`, `shot_offset`, `bin_number` | `seq_num` and per-stream indices |
| scan number and folder | a `PathProvider` plus our claim protocol |
| ScanInfo ini | a start-document callback |
| s-file, Tiled catalog | callbacks (already true) |
| capture daemon | deleted by #806 |
| `ScanRequest` as a worker instruction | a client-side template that expands into a plan call |
| the funnel, named plans, the #809 preprocessor | deleted |

Almost every hard problem of the last two weeks is one row of this table.
Orphan frames, save windowing, the refire, the shot join, bin numbers: all
of them are the detector and flyer contract, hand-rolled because our
cameras are not detectors yet.

---

## 6. What stays GEECS

Five things have no native home, and none of them is in the scan path:

1. **The DB as the source of truth** for the device roster, types,
   tolerances and subscribed variables. The namespace builder owns this.
2. **Day-scoped scan numbering** with a multi-writer claim protocol. A
   custom `PathProvider`.
3. **The s-file format** and its legacy `Device Variable` column headers.
   A callback, plus header metadata on the devices.
4. **ScanInfo ini.** A start-document callback.
5. **PV naming and the served-set rules.** Already owned by the two
   gateways and `geecs_core`.

`ScanRequest` and its JSON Schema also survive, as the **client-side**
record of intent the GUI needs. What dies is its role as a worker-side
execution instruction.

---

## 7. Verified versus assumed

**Verified 2026-09-09 against the installed environment** (ophyd-async
**0.19.3**, bluesky **1.15.0**):

- `StandardDetector`, `StandardFlyer`, `FlyerController`, `TriggerInfo`,
  `DetectorTrigger`, `DetectorAcquireLogic`, `DetectorDataLogic`
- `PathProvider`, `StaticPathProvider`, `AutoIncrementingPathProvider`,
  `AutoMaxIncrementingPathProvider`, `YMDPathProvider`,
  `AutoIncrementFilenameProvider`, `UUIDFilenameProvider`
- `TriggerInfo` fields: `trigger`, `livetime`, `deadtime`,
  `exposures_per_collection`, `collections_per_event`, `number_of_events`,
  `exposure_timeout`
- `DetectorTrigger`: `INTERNAL`, `EXTERNAL_EDGE`, `EXTERNAL_LEVEL`
- `bps.prepare`, `kickoff`, `complete`, `collect`,
  `collect_while_completing`, `declare_stream`
- **Naming has moved**: the writer base is not `DetectorWriter` in this
  version, and the flyer's controller is `FlyerController`, not
  `TriggerLogic`. Older docs and blog posts will disagree.

**ASSUMED, must be verified before designing on it:**

- that the PVA gateway can be made to deliver **indexed, edge-triggered**
  frames with a stable frame counter. The whole fly design rests on this
  and it is the substance of #806
- that `StandardDetector`'s acquire/data split can be satisfied by a GEECS
  camera server without an areaDetector IOC
- that a step scan with N shots per point composes cleanly as fly-per-step
  in 0.19.3
- that Tiled and the s-file exporter handle StreamResource/StreamDatum
  from these detectors as well as they handle the current asset documents

---

## 8. Sequencing: the decision for Sam

**#806 comes first.** Every row of the table that hurts most lives in the
detector contract, and a rebuild that starts before cameras are detectors
will re-derive the same workarounds we are trying to delete. This is a
reversal of the current ordering, in which #806 sits behind the plan work.

Three options, in the order I would recommend them:

1. **#806, then rebuild.** Cameras become `StandardDetector`s, the capture
   daemon dies, then the flyer and the plan layer follow. Slowest to first
   visible change, cleanest result, and it is the only order in which the
   twelve duplication findings actually disappear rather than move.
2. **Rebuild the plan layer first against today's devices**, accepting that
   the fly design waits. Faster feedback for Sam, at the cost of building
   the per-shot trigger machinery a second time and deleting it later.
3. **Land #809 and continue incrementally.** Recommended against, and the
   clean slate removes its only argument. Incrementalism buys backward
   compatibility, which is worth nothing here, and it pays for it with a
   reconciliation engine the target design deletes.

**On #809 itself:** do not merge it. Leave it open as the evidence, or
close it with a pointer to this document. Its two open P1 defects need no
fix if it is not shipping, and nothing from it is deployed. Parts worth
salvaging by hand: the document-parity test, `GeecsNamespace.select`'s
role assertions, `plan_session.py`, and the shared
`fire_and_await_shot` if per-shot triggering survives at all.

**What keeps working while this happens.** Answered: Sam's team is the
only user, so the answer is "whatever they choose to keep working that
week." The rebuild does **not** need to run beside the funnel, and the
old doors do not need a deprecation path. Keep the lab scannable between
sessions and nothing more. In particular, do not spend design effort on
dual-door parity — the parity test in #809 exists only because two doors
had to coexist, and in the target there is one.

**What the clean slate unlocks, and should be used for:**

- delete free-run, the funnel, the named plans and the capture daemon
  **eagerly**, as soon as each has a replacement, rather than in a phased
  retirement
- change the event schema, the s-file columns and the ScanRequest schema
  freely where the native shape is better; there is no reader to break
  that the team does not own
- renumber, rename and restructure the package. A **total refactor of
  GeecsBluesky**, or replacing it with a new package, is fully on the
  table and is probably cheaper than incremental deletion
- treat "we already built it" as carrying no weight. The only question is
  whether a thing is right

---

## 9. Standing constraints (Sam, 2026-09-08/09)

- Prefer a native Bluesky or ophyd-async mechanism wherever one plausibly
  exists. Examine carefully before patching. Never close off adding a stock
  plan or a Bluesky feature later.
- Added lines must be justified. Never a second copy of a solved problem.
  Where a better copy replaces an old one, the old one goes in the same
  change.
- **New, and the reason for this document:** do not hold on to anything
  that does not slot in completely cleanly.
- **Clean slate.** Sam's team is the only user. No backward compatibility,
  no deprecation cycles, no migration paths. Deleting is cheaper than
  adapting, and "we already built it" is not an argument.
- Phase PRs land into `feature/native-bluesky-plans`. Each gets an
  adversarial review pass using the `/land` three-lens brief, with every
  finding dispositioned, before Sam reviews. Master merges are
  maintainer-only.
- Hardware is available: trigger profile **HTU-NoGas**, scannable
  `U_S1H:current` −1 → +1 A in 0.5 A steps, save set **amp4in**, restore
  the setpoint. `ssh geecs-gw`; the worker checkout may be changed freely.
- Run **one** test suite at a time on the Mac, unbuffered
  (`python -u -m pytest -v`). A backgrounded pytest writing to a file looks
  stalled for minutes because stdout is block-buffered; that cost an hour
  in the last session and produced a retracted claim.

---

## 10. Open questions for Sam

1. Sequencing: option 1, 2 or 3 in §8. This is the only substantial one.
2. Refactor GeecsBluesky in place, or start a new package and let the old
   one die? The mapping is identical either way; the difference is whether
   the package keeps its history. The clean slate makes the second viable.
3. How much lab downtime is acceptable between working states? That now
   sets the pace, in place of any compatibility constraint.
4. Two small carry-overs from the last session, unrelated to this
   direction: write `Amplitude.Ch AB: 0.5` explicitly in every state of
   `HTU-NoGas` so "no gas" stops being order-dependent, and add a check
   that all profiles in an experiment manage the same variable set.
