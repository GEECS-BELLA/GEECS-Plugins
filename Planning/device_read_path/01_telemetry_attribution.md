# Phase 4 design note — timestamp-only telemetry shot attribution

**Status: DECIDED (Sam, 2026-07-13) and implemented in 0.35.0.**

- **D1 → (a)**: async devices carry **no** derived labels and no
  `systimestamp` column either — Sam's refinement: `systimestamp` advances
  with the device loop even when nothing is acquiring, so it can be
  meaningless as an alignment signal.  The raw
  `telemetry_<device>-acq_timestamp` column (0.0 placeholder for async)
  is recorded for every telemetry device; attribution beyond that is a
  downstream decision.
- **D2 → tabled**: no classification stage.  The standing design decision
  holds — anything needing strict synchronization belongs in a save set.
  Telemetry just logs `acq_timestamp`; sync context is recoverable post
  facto.
- **D3 → full companion set, under the D2 constraint**: devices whose
  cached `acq_timestamp` is positive at the quiesced t0 snapshot (i.e.
  observed to have actually fired) are seeded like contributors and emit
  `shot_id`/`shot_offset`/`valid`; everything else stays value-columns
  (+ raw acq) only.  Seeding is free — no observation window, no trigger
  authority, no flag: seeded ⇔ has-ever-fired at t0.

The original option analysis is kept below for the record.

---

## Why this matters (the cutover framing)

Background telemetry is a **parity gate for the M6 cutover**: the Bluesky
path must record at least as much context as the current best
implementations (Master Control's save-everything; the legacy scanner's
device logging) before either is retired.  Tier 2 already records every
`get='yes'` variable of every live device.  What it lacks vs "best" is
*shot attribution*: today a telemetry cell is "latest cached value at row
time" with no statement about which physical shot it belongs to.  For
triggered devices that statement is derivable — and the machinery already
exists.

## Mechanism (verified in the read-path audit, link 5)

- Every gateway-served device gets `acq_timestamp` and `systimestamp` PVs
  regardless of DB content.  A triggered device pushes `acq_timestamp` per
  shot; a non-triggered device leaves it at the `0.0` placeholder forever
  while `systimestamp` advances.  **Triggered-ness is therefore observable
  — no config flag, no DB toggle**:
  - `acq_timestamp > 0` and advancing in lockstep with the trigger
    (advance rate ≈ rep rate) ⇒ triggered;
  - pinned at `0.0` with a live `systimestamp` ⇒ async;
  - advancing but *not* in lockstep ⇒ independently-clocked (treat as
    async — the quiesced-window check below catches these).
- Per-sample attribution for triggered telemetry = the existing
  contributor recipe minus the grace wait (telemetry must never gate a
  shot): include the device's own `acq_timestamp` in its signal set, seed
  a `ShotIdTracker` at the t0-sync stage (cache reads only — cheap even at
  ~87 devices), and emit `shot_id`/`shot_offset`/`valid` companions from
  `read()` exactly as `FreeRunContributorSupport` does.
- **The one trap**: deadband/exact-repeat suppression means a *data*
  column's CA timestamp is "time of last change", not "time of this
  sample".  Attribution must key on the `acq_timestamp` PV (which reposts
  every frame), never on data-column reading metadata.

## Decisions needed (owner)

**D1 — what do async devices' telemetry rows say?**
  a. Nothing new (today's behavior): value columns only; offline analysis
     joins by the run's wall clock if it cares.
  b. Add a `systimestamp` column per async telemetry device: the device's
     own last-update time rides into every row, enabling offline
     nearest-shot joins *with the clock-skew caveat made explicit*.
  c. Nearest-shot `shot_id` labeling from `systimestamp` at read time:
     maximum convenience, but manufactures precision from skewed clocks.
  **Recommendation: (b)** — record the honest raw material, never a
  derived label the data can't support.  (a) loses information we get for
  free; (c) violates the "truthfully labeled" schema philosophy.

**D2 — when does classification happen?**
  a. Passively within each scan: classify from what `acq_timestamp` did
     during the scan itself; per-sample labels only become trustworthy
     after the first few rows.
  b. An explicit pre-claim observation window: watch all candidates'
     `acq_timestamp` during one armed window and one quiesced window
     (both primitives exist: the staleness probe and
     `geecs_confirm_quiescent`).  Deterministic, but costs seconds per
     scan and needs trigger authority before the run.
  c. (b) once per session/experiment, cached, with re-observation on
     demand — devices do not change triggered-ness scan to scan.
  **Recommendation: (c)** — first scan of a session pays the observation
  window; subsequent scans reuse the classification; an operator action
  (or an experiment change) invalidates the cache.

**D3 — which companions do triggered telemetry devices get?**
  The full contributor set (`-acq_timestamp`, `-shot_id`, `-shot_offset`,
  `-valid`) or a minimal `-acq_timestamp` only (offline derivation of the
  rest)?  **Recommendation: the full set** — the tracker arithmetic is
  cheap, in-row `valid` is what makes the columns immediately usable, and
  it is exactly the schema shape consumers already understand for
  contributors.  Adds ~4 columns per *triggered* telemetry device (the
  Undulator selection is mostly async, so the growth is modest).

## What this is NOT

- Not a schema version bump: new columns follow the existing contributor
  companion pattern and the `telemetry_` prefix rule; consumers that
  ignore them keep working.  (Telemetry as Bluesky monitor streams remains
  the fully conventional endgame and IS a schema change — deferred, pairs
  with #528.)
- Not dependent on the MySQL `synchronous`-ish context: the DB can serve
  as an optional *prior* to seed D2's first classification, never as the
  source of truth.

## Verification plan (the 2-camera + DG645 rig suffices)

1. Save set = `Amp4In` only; the output camera then lands in telemetry.
2. Run a free-run NOSCAN: the output camera's telemetry columns must
   classify triggered, seed at t0, and label `shot_offset == 0` rows in
   lockstep with the reference.
3. Quiesce the trigger mid-observation-window (D2b path): the camera must
   classify async-for-this-scan and fall back to D1 behavior.
4. Any genuinely async device that is still up (gateway status PVs at
   minimum) pins the D1 column shape.

## Estimated shape

`CaTelemetryReadable` grows an optional shot-id mode (constructor flag +
the two mixins it currently omits); `build_telemetry_readables` splits the
selection by classification; the observation stage is a new pre-claim step
beside preflight.  Roughly a #541-sized PR, dominated by tests.
