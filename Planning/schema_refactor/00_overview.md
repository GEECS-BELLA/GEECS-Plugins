# ScanRequest schema refactor + named scan plans

Handoff design doc, 2026-09-01 (Sam + Claude session on the HTU/OSPREY integration).
Status: **approved direction, implementation starting** (amended 2026-09-01 after
review with Sam: priorities, naming, migration story, invariants). Written for
whoever picks this up; assumes no context from the originating conversation.

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

### Phase 2 — three named plans (thin wrappers over the same path)

Register in the qserver startup profile, beside the existing funnel:

- `geecs_noscan_plan` — mode pinned to noscan; params: description +
  `AcquisitionSettings` + `ActionBindings` (+ `background`).
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
- **OWED from #730**: worker restart on the qserver host so `plans_allowed`
  serves the parameter annotations.

## Pointers

- Schema source: `GEECS-Schemas/geecs_schemas/` (`ScanRequest`), export module
  `geecs_schemas/schema_export.py`, artifact `docs/geecs_schemas/scan_request.schema.json`.
- Funnel plans + annotations: `GeecsBluesky/geecs_bluesky/plans/scan_request_plan.py`,
  applied in `GeecsBluesky/qserver/startup/startup.py`.
- Issues/PRs: GEECS-Plugins#727 (roadmap), #730 (schema publication),
  als-apg/osprey#816/#817 (external-worker mode + schema graft).
- Deployed test bench: the HTU deployment repo
  (`~/Desktop/Code/Github_repos/htu-assistant`), BLUESKY panel on the live
  stack renders whatever the manager + published schemas say.

## Suggested sequencing for the implementer

1. Phase 1 refactor PR (GEECS-Schemas minor bump + lifting validator;
   consumer updates across GeecsBluesky/Console/MCP/Data-Utils; regenerate
   artifact; CI guard proves no drift beyond the intended shape change).
2. **Console acceptance check**: confirm `request_builder` and the panels
   got simpler, not just different — this is the checkpoint that matters.
   (Re-vendoring the artifact into the HTU deployment repo to eyeball the
   OSPREY form is a nice-to-have, not a gate.)
3. Phase 2 PR (GeecsBluesky: three wrappers + annotations + startup
   registration; funnel untouched; shared-execution invariant pinned by
   tests).
4. Optional OSPREY tail, no deadline: publish per-plan schema artifacts;
   add the three `parameter_schemas` entries in the HTU profile; verify
   permissions granularity end-to-end (gate `geecs_optimize_plan` for a
   test group).
