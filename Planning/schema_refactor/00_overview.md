# ScanRequest schema refactor + named scan plans

Handoff design doc, 2026-09-01 (Sam + Claude session on the HTU/OSPREY integration).
Status: **Phase 1 landed** (#734, merged 2026-09-01, hardware-verified on
Undulator Scan004/Scan005 the same day); **Phase 2 amended 2026-09-01** after
reading the engine with Sam (see "Findings from reading the engine" and the
three-step Phase 2 below); **2a built** in the same session (GeecsBluesky
0.71.0 — the amendment and the code travel in one PR). Written for whoever picks this up; assumes no
context from the originating conversation.

## Priorities: the console comes first

OSPREY triggered this look, but it is the *second* client of the queueserver.
GEECS-Console was the first, is the one operators will face first, and is the
one we control; OSPREY is developed rapidly by another team, and this repo's
foundations shifting under it is expected and controlled. Emphasis order:

- Composition decisions are made on **domain-modeling grounds that serve the
  console**, not on how any particular renderer draws a form. The acceptance
  test for Phase 1 is *the console's `request_builder` and panels get
  simpler* (seven loose fields → one sub-model). A grouping that only helps
  OSPREY's form renderer and does nothing for the console is the wrong
  grouping.
- The console historically "took care of" the schema's shape in client code —
  that compensating wrapper layer is the real cost of the current
  composition, and it is code we maintain. The second client revealed the
  mess; the first client is the main beneficiary of fixing it.
- The only OSPREY-specific work (per-plan schema grafts, `PLAN_LAYOUTS`
  polish) is the optional tail of the sequencing and can slip indefinitely.

## Where this comes from

OSPREY now fronts the GEECS queueserver directly (external-worker mode,
als-apg/osprey#816 / #817): its BLUESKY panel lists the manager's `plans_allowed`
and renders a parameter form from the published ScanRequest JSON Schema
(GEECS-Schemas 0.13.0, #730; grafted OSPREY-side per GEECS-Plugins#727 item 2).
Seeing the *whole* ScanRequest rendered as one form for the first time exposed
two things:

1. the form is disorganized — a symptom of the schema's composition, not of the
   renderer;
2. every gate in both stacks (manager `user_group_permissions`, OSPREY's
   `write_tools` kill switch and approval prompts) keys on the **plan name**, so
   a 30-second background noscan and a 200-shot optimizer run are
   indistinguishable at every gate today.

## Findings from profiling the published schema

`docs/geecs_schemas/scan_request.schema.json` (generated from
`geecs_schemas.ScanRequest`): 15 top-level fields, 12 `$defs`.

- **Already well-composed**: `axes` (`ScanAxis` → `PositionRange | PositionList`),
  `optimization` (`OptimizationSpec` → `GeneratorSpec`/`EvaluatorSpec`),
  `actions` (`ActionBindings`), `submission` (`SubmissionRecord`).
- **Not composed — the "mess"**: seven loose top-level fields that are all one
  concern (capture): `shots_per_step`, `acquisition`, `save_sets`,
  `background_telemetry`, `native_image_save`, `trigger_profile`,
  `trigger_variant`.
- **Mode-dependence is invisible**: `noscan` has no meaningful `axes`;
  `optimize` effectively requires `optimization`; the schema can only express
  flat optionals, so neither validation nor the form knows which fields matter
  for which mode.
- **Request/record mixing**: `submission: SubmissionRecord` (and the
  `PreflightOutcome`/`PreflightCheckResult` defs it pulls in) are records of
  what *happened to* a request, living inside the request type.

## Proposal

### Phase 1 — decompose (schema refactor + consumer updates)

1. Extract **`CaptureSettings`** holding the seven capture fields as
   `ScanRequest.capture: CaptureSettings`. "Capture" is the concern's own
   name (this doc already called it that) and avoids the
   `acquisition`/`acquisition_settings` collision. The seven fields are
   conceptually three sub-groups — shot control (`shots_per_step` +
   `acquisition`), data logging (`save_sets` + `background_telemetry` +
   `native_image_save`), and trigger (`trigger_profile` +
   `trigger_variant`) — each arguably its own model, but all required by
   every scan, which is what makes `capture` the right umbrella. One level
   of nesting is enough for v2; the trigger pair (already coupled by a
   validator) is the one candidate for a sub-model if it earns it. Don't
   over-nest.
2. **Required: a lifting back-compat validator.** `ScanRequest` grows a
   `mode="before"` validator that accepts the flat v1 shape and lifts the
   seven fields into `capture`. This is the first mechanical
   `schema_version` migration (v1 → v2) — the marker exists for exactly
   this. Without it, Phase 1 is a lockstep wire-format break across the
   console, GEECS-MCP, the qserver worker, and every saved preset; with
   it, a rolling upgrade, and archived run metadata (immutable, flat
   forever) keeps parsing through `ScanRequest.model_validate` for free.
3. **Split request from record**: `submission: SubmissionRecord` (and the
   preflight result types) leave the request document — server-stamped
   lifecycle state, not operator input. The types stay in `geecs_schemas`.
   **Vehicle**: the funnel plan grows a separate optional `submission`
   parameter; `qs_client`'s submit path passes it alongside the request
   and the engine merges it into run metadata exactly as today. (A
   GeecsBluesky signature change riding inside the "schemas" phase —
   budget for it.)
4. Update the field-access sites: GeecsBluesky (`scan_request_runner`,
   funnel plan, `submit_preflight`, `shot_control`), GEECS-Console
   (`request_builder` + panels), GEECS-MCP, GEECS-Data-Utils
   (`tiled_catalog` — a *reader* of archived documents; it must keep
   handling the flat shape forever, which the lifting validator gives it
   if it parses via the model).
5. Regenerate the schema artifact (`python -m geecs_schemas.schema_export`;
   the no-drift CI guard keeps it honest). Bump `schema_version` to 2 and
   GEECS-Schemas' package version by a **minor** — two different dials.

This phase is worth doing regardless of Phase 2, and Phase 2 is thin only if
this lands first.

**Landed** as #734 (GEECS-Schemas 0.14.0, GeecsBluesky 0.70.0, Console 0.26.0,
MCP 0.8.0): everything above as written. **Amendment (2026-09-02):**
`trigger_variant` was then deleted outright (ScanRequest v3, TriggerProfile v2
— profile variants were never adopted; one profile file per operating
condition is the shape), so `CaptureSettings` carries six fields, not seven,
and 2b's parameter models build on that. Two things settled on the way that
the next phase inherits:

- **Versioning policy** (GEECS-Schemas `README.md`, "Versioning policy — two
  dials"): the integer `schema_version` bumps only with a lifting migration;
  additive fields ride the package version, the changelog, and the schema
  artifact's git history. No `1.1`-style markers. Phase 2 adds no document
  fields, so `schema_version` stays 2.
- **Deploy order is worker-first** whenever the queue's plan surface
  changes: a stamping client against an older worker fails queue-add
  validation (the #734 case), and a client submitting a *named plan* to a
  worker that has not registered it fails the same way (the 2b case —
  `user_group_permissions.yaml` + the startup profile's allowed list). 2a
  changes neither the funnel's public signature nor the plan list.

### Findings from reading the engine (2026-09-01, pre-Phase 2)

Phase 2 as first written assumed the wrappers could be thin because they
would "delegate to the exact same execution path". Reading GeecsBluesky with
that in mind:

- **The prologue exists twice.** `plans/scan_request_plan.py`'s plan body
  (`_scan_request_body` / `_optimize_request_body`) and the headless
  `scan_request_runner.run_scan_request` / `_run_optimize_request` each walk
  validate → resolve → construct → connect → claim over the same pure
  helpers. The queueserver migration relocated the prologue into the plan
  preamble deliberately and left the headless one for the in-process
  session; the two are kept aligned by hand ("mirrors …" comments). Three
  named plans on top of this would either pick one (drift) or add a third.
- **Caller audit of the headless path** (non-test, whole repo): the only
  caller of `run_scan_request` is `GeecsSession.run`; nothing in
  GEECS-Console, GEECS-MCP, or GEECS-DataPortal calls the session method.
  What keeps it alive: the hermetic engine suite (`test_scan_request_runner`
  alone references it ~57 times), the env-gated hardware sweep test
  (`test_scan_request_hardware.py` drives `session.run`), and its standing
  as the documented headless/scripting API (GeecsBluesky `CLAUDE.md`). It is
  a test harness and a scripting API, not a dead limb — keep the entry
  point, drop the second orchestration.
- **`scan_request_runner.py` is a grab-bag, not a runner** (~2200 lines):
  save-set merging and rituals, action-slot assembly, experiment defaults,
  validation, movable/detector construction, liveness preflight, telemetry
  readables, and the headless run functions share one module because they
  accreted during the consolidation. Distinct concerns, distinct tests.
- **Mode behaviour is scattered branches**, not one place per mode:
  noscan-as-one-step and the motor-argument choice in the spec builder,
  optimize-skips-actions in the run function, the step/optimize dispatch in
  the plan body. Fine at three modes; not at five.
- **What a plan can express.** A named plan can only assemble what
  `ScanRequest` already executes (noscan / step / optimize). Recombining
  existing capability into an operator-shaped plan is cheap after Phase 2.
  A genuinely new *execution* semantics (traversal ordering, burst, a
  different shot loop) is engine work first — enum + validator in the
  schema, a spec-builder branch or new generator in GeecsBluesky — and only
  then a cheap plan. Two-tier cost, by design.

### Phase 2 — three named plans, in three steps

Phase 2 is now three ordered steps. Step 2a is the one the engine reading
added; 2b is the original Phase 2; 2c is the OSPREY tail.

#### 2a — one preamble (no change on either door; **built**, GeecsBluesky 0.71.0)

Decided shape (the reviewer's P1/P2 on the first draft of this section
forced the choice): `GeecsSession.run` **is** `RE(geecs_scan_request_plan(...))`
on the session's own RunEngine; `run_scan_request` / `_run_optimize_request`
are deleted, not wrapped. Every known fork between the two doors became an
explicit keyword seam on the plan, so "no behaviour change" means *no change
on either door*, not "identical doors":

| Fork | Queue door | Headless door (`session.run`) |
|---|---|---|
| `failed_move_policy` | `"pause"` (plan default; decision 4) | `"raise"` (no operator to answer a pause) |
| `optimization_loader` | the worker's startup-registered loader | a loader over the caller's injected `objective`/`suggester` (+ `device_requirements`) |
| s-file export | worker stop-document callback | `session.run` exports after a saved run |
| operator abort | manager verb | `RE.abort()`, settled quietly (`last_run_aborted`), as `session.scan` |
| `should_abort` init-stage probe | never existed | **removed** — no production caller since the bridge died |

The funnel's public signature keeps `request`, `submission`, `session`,
`resolver`; the two new seams are unannotated keyword-only (RE Manager item
validation unaffected) — no client or deploy-order impact.

Acceptance as built: the hermetic runner suite keeps its *entry shape*
(fake session, no RunEngine) but its fixture was rewritten to step the
plan generator directly, running its connect/disconnect coroutines; the
hardware sweep test still runs through `session.run` unchanged; the
document-parity tests were **repurposed**, not retired — they now compare
`session.run` against the queue call shape, the structural pin that the
headless door is the plan. The port surfaced two queue-path bugs the
headless runner had masked (zero-save-set optimize crashed on the empty
merge; request-level skipped actions were not recorded) — fixed on the plan.

**Scope line held, follow-up flagged:** `GeecsSession.optimize` (the
low-level scripting twin of `session.scan`, over already-built devices)
stays; its iteration loop duplicates the plan's optimize body and it has no
caller outside tests/README. Collapsing it is a second concern — do it when
the scripting API is next touched, or delete it if nobody scripts
optimizations headless by then. The runner-module split by concern was
**not** done (PR-size discipline); the per-mode spec-builder registry stays
deferred until a fourth mode forces it.

#### 2b — the three named plans

Register in the qserver startup profile, beside the existing funnel:

- `geecs_noscan_plan` — mode pinned to noscan; params: description +
  `CaptureSettings` + `ActionBindings` (+ `background`).
- `geecs_scan_plan` — 1-D and grid as one plan (a grid is the multi-axis case
  of the same sweep; splitting them would be artificial); params: `axes` +
  shared components.
- `geecs_optimize_plan` — params: `axes`-as-bounds per current optimize
  semantics + `OptimizationSpec` + shared components.

Each is a mode-pinned parameter model **composed from the Phase-1 sub-models**,
building a ScanRequest internally and delegating to the exact same execution
path. No triplicated validation, no forked run discipline.

**Shared-execution invariant (non-negotiable)**: every named plan builds a
canonical `ScanRequest` internally and delegates to the single existing
execution path — same `run_wrapper`/`claim_scan_number`, same save-set
resolution, same trigger and data-logging architecture, same Tiled writing —
and the document recorded in run metadata is always the `ScanRequest`, so
`tiled_catalog` and every analysis-side consumer sees one shape regardless
of entry plan. The wrappers are vocabulary at the gate, invisible
downstream. Phase 2 forces **zero** console changes: the console keeps
submitting through the funnel and adopts named plans only if they earn it.

**Keep `geecs_scan_request_plan`** as the programmatic/compat entry point:
GEECS-Console and GEECS-MCP `submit_scan` keep submitting ScanRequest documents
untouched, into the same queue. The funnel retires (or stays forever as the
machine API) on usage evidence — strangler-fig, no migration cliff.

Each named plan's parameter model lives in GEECS-Schemas (pydantic-only),
its plan function in GeecsBluesky beside the funnel, its
`user_group_permissions` entry in the qserver profile. Per-plan JSON Schema
artifacts land under `docs/geecs_schemas/` through the export registry
(`schema_export.EXPORTED_SCHEMAS`, **built as 2b-i**, GEECS-Schemas 0.16.0:
one artifact per entry, the no-drift guard parametrized over the registry,
an orphan-artifact check, `SCHEMA_ARTIFACT` kept as the ScanRequest alias
the worker annotation and OSPREY point at). Each named plan's parameter
model is one registry line plus a regenerate. The Markdown reference for
the plan models (2b-ii) has docgen iterate `EXPORTED_SCHEMAS` for a
plan-models page — each plan model registered exactly once — not a change
to the config-kind registry (`SCHEMA_REGISTRY` stays the 8 versioned YAML
kinds; `scan_request` is the one model in both, pinned); mkdocs nav stays
as is.

#### 2c — OSPREY tail (optional, no deadline)

Per-plan `parameter_schemas` entries in the HTU deployment profile, optional
`PLAN_LAYOUTS` polish, and the permissions-granularity check (gate
`geecs_optimize_plan` for a test group). Detailed under "OSPREY-side impact".

### Why three plans rather than a discriminated union on `mode`

Both give per-mode field sets. Three plans additionally give:

- **Gate granularity**: `user_group_permissions` can allow noscans to a group
  while gating `optimize`; OSPREY's approval prompt can treat an optimization
  launch as the bigger decision it is. The plan name is the unit every gate
  sees — this is the safety-architecture argument and the main driver.
- **No renderer risk**: OSPREY's `schema-form.js` support for discriminated
  `oneOf` is unverified; three catalog entries with already-mode-specific
  schemas use the existing, working machinery.
- **Legible queue/history**: `geecs_optimize_plan` in the manager history says
  what happened; `geecs_scan_request_plan` × N does not.

Trade-off: mode-spanning tooling (generic validation, MCP `submit_scan`,
dry-run preview) stays simplest with one document type — which is exactly what
keeping the funnel preserves. Document API for machines, named plans for humans
and gates.

## OSPREY-side impact (near zero, by design)

- Plan catalog is downloaded from `plans_allowed` — three new entries appear
  automatically.
- Per-plan schema grafts: the deployment profile's
  `bluesky.external.parameter_schemas` maps `"<plan>.<parameter>"` → artifact
  per entry, so each new plan can publish its own composed schema
  (GEECS-Schemas exports them the same way ScanRequest is exported today).
- Optional polish: a `PLAN_LAYOUTS` entry per plan in
  `osprey/interfaces/bluesky_web/panels/bluesky/plan-presentation.js` for field
  arrangement.

## Known open items in the same territory (do not lose)

- **Draft staging bug (OSPREY)**: the panel's draft flow resolves schemas from
  the *local* plan registry only (`bluesky_bridge/draft.py::_resolve_plan_schema`)
  → `unknown plan 'geecs_scan_request_plan'` when staging a draft for a
  manager-owned plan. Same latent gap in the preview route. Diagnosed, fix
  scoped, not yet implemented — tracked in the osprey#816 orbit, not here.
- **GEECS-Plugins#727 item 3**: document-stream contract (`:5568`) for live
  scan rows in the OSPREY panel.
- ~~OWED from #730: worker restart on the qserver host~~ — closed 2026-09-01
  by the #734 verification restart (worker at GeecsBluesky 0.70.0).
- **Interim-host checkout drift**: the worker host's GEECS-Plugins checkout
  was left detached at the #734 branch tip; sync it to master before the 2a
  deploy.

## Pointers

- Schema source: `GEECS-Schemas/geecs_schemas/` (`ScanRequest`), export module
  `geecs_schemas/schema_export.py`, artifact `docs/geecs_schemas/scan_request.schema.json`.
- Funnel plans + annotations: `GeecsBluesky/geecs_bluesky/plans/scan_request_plan.py`,
  applied in `GeecsBluesky/qserver/startup/startup.py`.
- Issues/PRs: GEECS-Plugins#727 (roadmap), #730 (schema publication),
  als-apg/osprey#816/#817 (external-worker mode + schema graft).
- Deployed test bench: the HTU deployment repo (the `htu-assistant`
  checkout), BLUESKY panel on the live stack renders whatever the manager +
  published schemas say.

## Suggested sequencing for the implementer

1. ~~Phase 1 refactor PR~~ — landed as #734 (2026-09-01).
2. ~~Console acceptance check~~ — met in #734: `request_builder` builds and
   reads one `CaptureSettings` sub-model (the PR's stated acceptance test).
   The OSPREY re-vendor remains a nice-to-have, not a gate.
3. ~~Phase 2 PR~~ → now three PRs, in order:
   - ~~**2a**~~ (GeecsBluesky 0.71.0): one preamble — built as described
     above; runner module split **not** done (deferred, PR-size discipline).
   - **2b** (GEECS-Schemas minor + GeecsBluesky minor): three parameter
     models + three plans + annotations + startup registration + permissions
     + per-plan artifacts; funnel untouched; shared-execution invariant
     pinned by tests (each plan's recorded start doc is a `ScanRequest`).
   - **2c** OSPREY tail, no deadline (see above).
4. Hardware verification per PR, worker-first: 2a = one console noscan
   through the funnel (nothing should look different) **and** one headless
   `session.run` noscan from a lab-network Python session (the door that
   actually changed; the env-gated hardware sweep test is exactly that);
   2b = one queue item per named plan from the qserver CLI, start docs
   checked for the canonical `ScanRequest` shape.
