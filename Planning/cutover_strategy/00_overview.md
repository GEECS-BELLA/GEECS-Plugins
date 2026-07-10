# Cutover strategy — feat/vision-v1 as a parallel branch, gut-first

Decided 2026-07-10 (maintainer + agent discussion). This note supersedes the
incremental coexistence sequencing in `Planning/gui_retrofit/00_overview.md`
§4 steps (ii)–(iv); step (i) (bridge parity, PR #485) is unaffected and
remains the foundation.

## The decision

**The gut runs as a branch experiment with a clean abort.** A second
integration branch (`feat/vision-v2` or similar) is cut from
`feat/vision-v1`; the gut sequence and any greenfield-GUI work land THERE.
`feat/vision-v1` stays the known-good, coexistence-capable fallback.

Branch discipline (strict, or the experiment fails by sync cost):

- Syncs flow one way: master → v1 → v2. After the branch point, **v1
  receives only master syncs and critical engine fixes; all new
  development lands on v2.**
- Path-independent work stays on v1: the deployment recipe, the step-(i)
  hardware smoke, the s-file coverage check — an abort must lose nothing
  operational.
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
- **G-actions — re-home manual actions** (decision pending, maintainer
  leaning discussed): the ActionLibrary "perform action" feature is already
  dark under Bluesky (`action_control = None`). Recommended: a
  bluesky-native ActionControl running named `ActionPlan`s through the
  engine's existing `compile_action_plan` + `CaActionSignalFactory` — turns
  a feature regression into a schema consolidation. Until then the feature
  stays dark, ActionManager stays alive for the dialog's condition checks.

Known follow-ups that do NOT block the gut: `geecs_python_api` remains a
GUI dependency (ScanDevice live readback/manual-set in the main window, DB
lookups, error types) — it drops at M5 at the earliest, not here. Verify
GeecsBluesky's s-file export covers every scan mode relied on
(noscan/1-D/optimization/background) before G1 removes the legacy producer.

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

## M6 gate list (draft — to finalize with the maintainer)

- N consecutive ops days on the request path at facility #1: strict +
  free-run + DG645, at least one action-bearing scan, zero master
  fallbacks.
- Optimization verified once end-to-end (after G-actions/step-(iii)
  equivalent wiring).
- s-file outputs confirmed for every mode ScanAnalysis consumes; TDMS
  reliance explicitly confirmed dead or replaced.
- Deployment recipe executed clean on facility #2's machine.
- Config corpus migration decided (recommendation: opportunistic — a file
  migrates to new schema when first edited — plus a one-shot conversion
  script held in reserve).
- Explicitly non-blocking unless ops says otherwise: pseudo scan
  variables, `all_scalars`, legacy TDMS scan-summary output.
