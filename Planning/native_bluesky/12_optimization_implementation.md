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

## Beam-free hardware smoke test — 2026-09-15

With the operator confirming beam off and HTU-NoGas, an isolated copy of the
branch ran `bax_alignment_simulation` against the real gateway: **3 iterations
× 2 shots**, S1V bounds **−4…4**, EMQ Current_Limit.Ch1 bounds **1.2…1.7**.
Scan010 of 26_0915, UID `53a0af7b-7101-4eab-b9a3-3f8ce92f0b42`, closed successfully.

- Six primary rows, bins `1,1,2,2,3,3`, in both scalar files; three finite
  synthetic observations in the in-folder `xopt_dump.yaml`; two valid shots
  for both measurements in each iteration.
- Both original magnet setpoints were **0** and were restored to **0**;
  final readbacks were S1V **−0.00009** and EMQ **0**.
- This exercised BAX's two random cold-start observations and one model-driven
  proposal. As an observables-only problem it correctly offered no best point.
- Captured documents, restoration evidence and the hardware summary are saved
  inside Scan010 as `optimization_acceptance_{documents,restoration,summary}.json`.

The live test exposed a Tiled SQL restriction: `%` in our escaped column names
caused optimization-table creation to fail. The branch now uses one shared
worker/client codec with `~` escapes and rejects excessive or case-colliding
column names before scan claim. A hardware-free replay against **Tiled 0.2.14**
(the lab version), with isolated SQLite storage, archived all **6 primary,
3 optimization and 2 baseline** rows and the successful stop. Replaying the
original encoding reproduced the exact SQL identifier failure. The original
live Tiled entry remains partial; its captured documents and scalar/Xopt files
are intact. No additional shots were taken for the archival fix.

Follow-up verification: **11 codec/SQL archival tests passed**, **10 native
optimization tests passed**, **6 scanner optimization/boundary tests passed**.
An independent review of the codec and preclaim checks found no surviving
concrete findings.

## Still owed before rollout

No deployed service or live configs tree was changed. The hardware test used
an isolated temporary checkout on the worker host. No PR has been opened and
no branch has been pushed.

1. After the schema lands, regenerate the configs repository corpus on its
   main branch: copy the keeper configurations, retain the HiResMagCam pair
   with LEGACY headers, and resolve the ALine diagnostic prerequisite noted
   in the brief. This branch's fixtures do not constitute corpus deployment.
2. The beam-free BAX smoke test above passed. Complete any longer convergence
   acceptance separately; three iterations do not establish optimizer quality.
3. Run TopViewMax with beam: **10 iterations × 5 shots**, five valid frames
   in each iteration, plausible objective against saved images, and record
   RSS before/after. Check the live PVA timestamp join and cleanup on stop.
4. Complete the real-camera BAX alignment acceptance on an operator day.

The four review/rollout units in the original brief remain useful; this local
development branch contains their code together and is not a hardware signoff.
