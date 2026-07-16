# Cutover strategy — feat/vision-v1 as a parallel branch, gut-first

Decided 2026-07-10 (maintainer + agent discussion). This note supersedes the
incremental coexistence sequencing in `Planning/gui_retrofit/00_overview.md`
§4 steps (ii)–(iv) (that doc has since been purged — executed); step (i)
(bridge parity, PR #485) is unaffected and remains the foundation.

## Status 2026-07-13 (Sam-approved amendment)

Recording what has happened since this doc was written; the plan below is
kept as the historical record of the decision.

- **The commit/abort checkpoint resolved COMMIT** — implicitly, when the
  M5 console verification (0.6.0, full 10-item hardware checklist,
  2026-07-12) blew past condition (c). The abort option is gone; the
  greenfield direction is the direction.
- **G1 is done** (#487, 2026-07-10).
- **G2 as written never happened and is superseded**: the old GUI was not
  retrofitted to build `ScanRequest`s natively — GEECS-Console *is* the
  native-ScanRequest GUI, and GEECS-Scanner-GUI stays frozen on its
  exec_config path (`_build_exec_config`) until it is deleted
  root-and-stem at M6.
- **G3 therefore re-times to M6** (delete the bridge's exec_config twin
  path together with the old GUI), not "once G2 lands."
  *Update 2026-07-16: G3 executed early by owner decision ("we've totally
  abandoned the old gui path… We can break the legacy scanner on the dev
  branch") — the bridge's exec_config path is deleted root-and-stem
  (GeecsBluesky 0.39.0); the old GUI itself still deletes at M6, and
  `master`'s legacy line is untouched.*
- **G-actions is the only live pre-M6 engine step from this doc.**
- **Branch topology collapsed 2026-07-13** (#549): `feat/vision-v1`
  retired, the greenfield branch renamed `dev`; `master` = legacy scanner
  + living analysis line. `CONTRIBUTING.md` § Branch topology is the
  canonical copy; branch names below are historical.

## The decision

**The gut runs as a branch experiment with a clean abort.** A second
integration branch — **`feat/greenfield-epics-bluesky-gui`** (named
2026-07-10; carries the gut G1–G3 as well as the GUI work) — is cut from
`feat/vision-v1`; the gut sequence and greenfield-GUI work land THERE.
`feat/vision-v1` stays the known-good, coexistence-capable fallback.

Branch discipline (strict, or the experiment fails by sync cost):

- Syncs flow one way: master → v1 → v2.
- **The division is by path-dependence, not by date**: v1 continues to
  receive everything BOTH futures need — gateway features (alarm limits,
  derived channels, status PVs, …), schemas, GeecsBluesky engine fixes,
  the deployment recipe, hardware smokes, the s-file coverage check. An
  abort must strand nothing. Only work **exclusive to the gut/greenfield
  direction** (GUI changes, legacy-engine deletion, anything touching
  files the gut removes) lands on v2. v1→v2 merges flow routinely.
- **Commit/abort checkpoint** (time-boxed — the abort option decays):
  after (a) G1 lands on v2 with the suite green, (b) G2-retrofit runs one
  real hardware scan, and (c) a greenfield screen-map + one working screen
  exist, explicitly decide: commit (v1 freezes to pure fallback) or abort
  (delete v2, resume the coexistence plan on v1).

**`feat/vision-v1` is a fully parallel branch.** It is never tip-toed back
into master; master merges *into* it regularly, and the final merge the
other way is effectively "master becomes vision" at cutover. Consequence:
the coexistence machinery planned for M4 steps (ii)–(iv) — the
`GEECS_USE_SCAN_REQUEST` flag, dual submission paths, flag-flip default —
protected a same-branch incremental rollout that will never happen. It is
dropped. The legacy scan engine is deleted on this branch **now**, so every
subsequent build (GUI submission, editors, optimization wiring) targets one
architecture.

**The fallback story changes shape and we accept it**: instead of an env
var mid-ops-day, the deployment-time fallback is *check out master on the
control machine* (~minutes). First soak days are chosen accordingly
(low-stakes ops).

**The translation rule** (the price of the parallel branch, made explicit):
a master fix touching legacy scan behavior no longer merges — it gets a
conscious yes/no on whether the new engine needs an equivalent. Master's
active development is mostly ScanAnalysis/ImageAnalysis, which merge
cleanly; delete/modify conflicts on gutted files resolve as "keep deleted."

## The gut sequence (PRs to the v2 experiment branch)

Cut list verified by a full consumer audit 2026-07-10 (every importer of
every legacy engine module mapped; see PR discussion).

- **G1 — delete the legacy scan engine** (deletion-only; GUI stays
  bluesky-only via the existing exec_config submission):
  - Delete: `engine/scan_manager.py`, `default_scan_manager.py`,
    `data_logger.py`, `device_manager.py`, `scan_executor.py`,
    `scan_data_manager.py`, `trigger_controller.py`, `file_mover.py`,
    `lifecycle.py`, their tests, RunControl's legacy branch,
    `ScanExecutionConfig.to_device_manager_dict`.
  - **First edit `engine/__init__.py`** — it eagerly imports everything;
    pruning it is the single mandatory ordering constraint.
  - Collapse `backend_selection` to a bluesky-default stub.
  - Strip the `TYPE_CHECKING`-only `DataLogger`/`ScanDataManager` hints in
    `optimization/base_evaluator.py` / `base_optimizer.py`.
  - **Keep (not legacy, despite living in `engine/`):** `ActionManager` +
    `DeviceCommandExecutor` (ActionLibrary GUI), `sound_player`
    (action_control + multi_scanner jingles), `DatabaseDictLookup`,
    the `scan_events`/`dialog_request` shims, all of `engine/models/`
    (GUI→backend contract, also used by optimization).
- **G2 — GUI builds a `ScanRequest` natively.** `_build_scan_request`
  replaces `_build_exec_config`; no flag, one submission shape. Absorbs the
  old steps (ii)–(iv).
- **G3 — delete the bridge's exec_config path** (`_execute_scan`,
  `_build_session_devices` twin, exec_config duck-typing) once G2 lands.
  The bridge becomes a thin request adapter; the device-build twins
  reconcile here.
- **G-actions — re-home manual actions (DECIDED 2026-07-10: yes).**
  Diagnosis confirmed: every piece already exists (ActionPlan schemas,
  `compile_action_plan` plan stubs, `CaActionSignalFactory`, resolver-side
  legacy-YAML conversion) — only the interactive entry point was never
  wired. Scope: (1) `GeecsSession.run_action(name)` — resolve → compile →
  prefetch signals → `RE(stub())`, outside any scan; (2) a bluesky-native
  ActionControl adapter for the GUI, including `return_device_value`
  re-homed onto a gateway PV read (condition checks + step-list live
  gets). Once landed, **ActionManager + DeviceCommandExecutor move from
  the keep-list to the cut-list** (sequence G-actions before or with the
  PR that deletes them).

Known follow-ups that do NOT block the gut:

- **`geecs_python_api` in the OLD GUI is left untouched and dies with it
  at cutover** — its usages (ScanDevice live readback/manual-set, DB
  autocompletes, error types) are not worth replacing in code that is
  being retired. **GEECS-Console never acquires the dependency**
  (decided 2026-07-10): manual set/readback goes through gateway PVs
  (CA monitor on the readback, put to ``:SP`` riding GEECS's native
  blocking set), DB autocompletes through **`GeecsDb`** (the clean
  interface), error types from the geecs-bluesky/gateway exception tree.
  One capability note: the ScanDevice path also handled *composite*
  manual-set; GEECS-Console's manual panel is scalar-only at birth
  (composites arrive with the pseudo-variable runtime, if ever).
- Verify GeecsBluesky's s-file export covers every scan mode relied on
  (noscan/1-D/optimization/background) before G1 removes the legacy
  producer.

## M6 = a per-experiment deployment event, not a code event

Two facilities, two independent Linux centers (each hosting its own MySQL
with the identical schema). Decision: **one gateway + Tiled stack per
facility machine — separate deployments, consolidated recipe.**

- Gateway placement is dictated by physics: it speaks UDP/TCP to that
  facility's devices and reads that facility's local MySQL. No
  cross-subnet routing project.
- Fault isolation: one facility's incident/maintenance never stops the
  other's DAQ. Experiments run on different schedules; coupling their
  infrastructure means coordinating maintenance forever.
- The config chain already supports it natively (per-machine `config.ini` →
  `Configurations.INI` → local DB; per-control-machine CA `addr_list`; the
  `[Experiment:]` PV prefix keeps namespaces distinct).
- The solo-maintainer investment goes into the **recipe**: turn
  `GeecsCAGateway/DEPLOYMENT.md` into a turnkey artifact (systemd units +
  install script or compose file), versioned in git, so facility #2 is an
  hour's work and both stay in lockstep.
- Central consolidation belongs at the **monitoring layer** (a future
  archiver pulling from both CA servers), never at the access layer.
  A central Tiled is a possible later analytics consolidation; not now
  (writes fail soft, but mixed topology = two mental models).

Rollout order: cut over facility #1 (HTU/Undulator — where all hardware
verification lives), soak, then facility #2 as the first true test of the
recipe. Pre-work: one sanity pass that nothing assumes a single global
gateway host.

## Greenfield GUI — settled parameters (2026-07-10)

- **Toolkit: PySide6** (LGPL — this repo is public; Qt-official; Qt
  Designer `.ui` files keep the human-layout capability, and `.ui` XML is
  agent-editable).
- **New top-level package: `GEECS-Console` (`geecs_console`)** — decided
  2026-07-10. The control-room word for the operator front-end, distinct
  from the `geecs-bluesky` library. (`geecs-bluesky-scanner` rejected as
  too close to `geecs-bluesky`.)
- **Day-1 dependencies: PySide6 + geecs-bluesky + geecs-schemas.
  No `geecs_python_api`, ever** (decided 2026-07-10) — manual
  set/readback via gateway PVs, DB autocompletes via `GeecsDb`.

## TDMS (decided 2026-07-10)

- **On-shot TDMS: dropped.** It was a poorly-implemented Master Control
  preservation and was not in use.
- **End-of-scan TDMS conversion: possible future exporter** for LabVIEW
  tooling — naturally a post-scan Tiled→TDMS converter alongside the
  existing s-file exporter (analysis-side, needs no scanner integration).
  Not scheduled, not a gate.

## M6 gate: the validation scenario matrix

Decided: the soak criterion is **a scenario checklist, not N days** — a
list of scans to run at facility #1 under breaking conditions, each
checked off once. Draft matrix (maintainer edits):

*Modes ×* *trigger regimes*:
- noscan / 1-D step / multi-axis grid / background — each in strict
  (DG645 ARMED/SINGLESHOT) and free-run (jet bracketing) where applicable
- optimization run end-to-end (after G-actions / optimize wiring)
- a save-set-union scan (two named sets) and a composite-variable scan

*Actions*: request setup/closeout · save-set rituals · experiment-defaults
bracket · per-step actions at grid points

*Failure drills*:
- device off at preflight → operator drop + reference promotion
- device dies mid-scan → error dialog path, clean abort
- free-run trigger off → staleness dialog wording
- DB unreachable → scan proceeds with explicit-only scalars (no abort)
- Tiled down → scan proceeds (soft-fail), s-files still written
- NetApp unmounted → claim refuses gracefully, no empty folder planted
- operator abort mid-scan → STANDBY restore, save-off before trigger can
  pass edges, no orphan images
- config errors (unknown names, conflicting save-set roles, bad trigger
  variant) → fail at submit, no scan number burned

*Data integrity per mode*: s-file complete · images join event rows ·
telemetry columns present · scan.log written · ScanInfo fields correct.

## Other M6 gates

- Deployment recipe executed clean on facility #2's machine.
- **Config corpus migration: on a named branch of the configs repo**
  (GEECS-Plugins-Configs — the corpus lives there, not here), e.g.
  `schema-v1-migration`, coordinated with the cutover. Opportunistic
  migration (file converts when first edited) + a one-shot conversion
  script, landing on that branch.
- Explicitly non-blocking unless ops says otherwise: pseudo scan
  variables, `all_scalars`, TDMS outputs (see above).
