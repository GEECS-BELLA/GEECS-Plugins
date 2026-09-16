# Native optimization implementation

Local branch: `codex/native-optimization`, based on
`feature/native-bluesky-plans` at `22961377`. The original uncommitted brief
in the main checkout is untouched. This branch carries an annotated copy.

The implementation covers the schema, declared analyzer scalars, native
strict worker plan, live camera measurement compiler, and scanner Optimize
mode from [the brief](11_optimization.md). GEECS-Schemas embeds GEST's actual
VOCS model; GEST 0.1 depends only on Pydantic. Every consuming Poetry lockfile
includes it. The legacy optimization schema, converter and evaluator loop
are removed. Six v1 keeper configs are validated fixtures under
`GEECS-Schemas/tests/fixtures/optimizer_configs/`.

Relative pseudo variables restore their starting positions on unstage.
The completed run records physical targets for the scanner's explicit
Set to best action. An observables-only BAX run has no arbitrary best point:
the default finish policy restores its initial positions. Seed observations
outside narrowed bounds cannot become the best point; relative-pseudo seed
reuse is refused because its zeroing frame would differ.

## Verification

- ImageAnalysis and GEECS-Schemas: **742 passed, 5 skipped, 17 deselected**.
  Executed with the existing integrated Python 3.11 environment and explicit
  worktree package paths. The scalar parity test also rejected an injected
  nonexistent declaration; the mutation was in memory, not left in source.
- GEECS-Core: **125 passed, 5 deselected**.
- GeecsPvaGateway: **51 passed**.
- GEECS-MCP: **102 passed, 1 skipped**.
- GeecsBluesky: **633 passed, 3 hardware modules skipped, 1 deselected**.
- GeecsScanner: **80 passed, 1 skipped**.
- GeecsCAGateway derived-channel compatibility: **19 passed**.
- Scanner browser demo: config/default loading, required-device locks,
  finite iteration budget, two completed iterations, live variables/outputs,
  valid-shot counts and Set to best submission. Visual spacing checked.
- All schema-consuming lockfiles checked for the GEST entry and dependency
  edge. Pre-commit includes the repository's path-scrubbing checks.

Native RunEngine coverage includes scalar and camera measurements, a dropped
frame followed by a replacement shot, failure before scan claim, immediate
and deferred pause/resume without duplicate optimizer observations, dump/seed
round trips, best fallback and relative-pseudo restoration. Xopt/BAX tests use
the real generators. The camera fixture allows 0.5 seconds for the simulated
fire, avoiding a scheduler race in its former 30 ms timeout.

An independent adversarial review found five issues which were fixed and
confirmed: partial rows displacing successful refire samples; physical motor
overlap through aliases; seed bests outside current bounds; relative seed
reference frames; and reading a full camera descriptor before prepare.
The final review of event-key encoding, JSON provenance and QueueServer
discovery reported **no surviving P1/P2 findings**.

## Still owed before rollout

No lab hardware was commanded and no deployed service or live configs tree
was changed. No PR has been opened and no branch has been pushed.

1. After the schema lands, regenerate the configs repository corpus on its
   main branch: copy the keeper configurations, retain the HiResMagCam pair
   with LEGACY headers, and resolve the ALine diagnostic prerequisite noted
   in the brief. This branch's fixtures do not constitute corpus deployment.
2. Run `bax_alignment_simulation` with the real magnets and synthetic
   observable, without beam. Verify movement/restoration, one s-file row per
   shot and the in-folder `xopt_dump.yaml`.
3. Run TopViewMax with beam: **10 iterations × 5 shots**, five valid frames
   in each iteration, plausible objective against saved images, and record
   RSS before/after. Check the live PVA timestamp join and cleanup on stop.
4. Complete the real-camera BAX alignment acceptance on an operator day.

The four review/rollout units in the original brief remain useful; this local
development branch contains their code together and is not a hardware signoff.
