# feat/vision-v1 cleanup — audit findings and decisions

Four-angle audit of GeecsBluesky + GeecsCAGateway (2026-07-10, on
feat/vision-v1 after M4 step 0 / #477): dead code, docstring bloat,
duplication, and module structure. This file records what we changed, what we
deliberately kept, and what is deferred — so a later pass doesn't re-litigate
the same calls.

Baseline numbers at audit time: geecs_bluesky 15,170 lines (34%
docstring+comment; scan_request_runner.py alone 844 docstring lines of 1,940);
geecs_ca_gateway 4,544 lines (29%, mostly load-bearing wire-protocol/DB
semantics).

## PR 1 (this change): dead code, dedup, stale references, resolver split

- **Deleted dead code**
  - `bluesky_scanner.py`: `_ScanConfig` import shim, `_CONNECT_TIMEOUT`,
    `dialog_queue`, `restore_failures` (the GUI stopped reading the latter
    two when `ScanManager` dropped them; verified no consumer).
    `last_reinit_error` KEPT — `geecs_scanner.py` reads it defensively.
  - `scan_request_runner.py`: the singular `resolve_save_set_checked` +
    `resolve_save_set_and_rituals` chain — superseded by the plural M4
    step-0 forms, no production caller (tests ported to the plural forms).
  - `utils.py`: `build_signal_attrs` — its collision-disambiguation loop was
    never wired into any detector (they all call `safe_name` directly);
    test-only. If attr-collision safety is ever wanted, re-add it *wired in*.
  - Gateway: `GeecsDb.list_devices()` (zero consumers; the config path uses
    the batched `get_experiment_devices`), `FakeGeecsDevice.fire_shot()`
    (its only consumer was the deleted direct-backend test suite).
- **Deduplicated**
  - `_build_positions` (bridge) now delegates to `session._positions` — the
    one true accidental copy of the start/end/step → linspace rule.
  - Gateway transport: the numeric-coercion idiom (`float` → int-if-whole →
    fallback str) triplicated across `tcp_subscriber`/`udp_client`/fake
    server → shared `transport/_coerce.coerce_scalar` for the two production
    copies. The fake server deliberately keeps its own copy (test/prod
    decoupling).
- **Stale references fixed** (~20 sites): every docstring cross-reference to
  the deleted direct-backend classes (`GeecsTriggerable`, `GeecsMotor`,
  `GeecsSettable`, `GeecsGenericDetector`, `GeecsTimestampedReadable`,
  `GeecsSnapshotReadable`) → the `Ca*` paths; the `devices/ca/*` module
  headers no longer define themselves as "the CA counterpart of" deleted
  classes; `step_scan.py`'s usage example no longer shows the deleted
  host/port constructor (rewritten against `GeecsSession` factories);
  `GeecsBluesky/CLAUDE.md` acquisition-mode names; gateway `CLAUDE.md`'s
  claim that GeecsBluesky imports the fake server; gateway `config.py`
  module docstring describing the shipped DB-driven config path as future
  work.
- **Split**: `ConfigResolver` + `ConfigsRepoResolver` →
  `geecs_bluesky/config_resolver.py` (~320 self-contained lines out of the
  runner; zero shared privates with execution code). Both names re-exported
  from `scan_request_runner` so the existing import surface is unchanged.

## PR 2 (planned): the docstring pass

The runner is a ~760-line module wearing an 844-line docstring coat; package
wide ~1,000 doc lines are removable with zero contract loss. Rules:

- Docstrings state the **contract** (what/args/returns/raises + non-obvious
  invariant), numpy style, concise.
- Design **why**/history/verification anecdotes live in the package
  `CLAUDE.md` or `Planning/` — most already do; the code copy shrinks to a
  one-line pointer. Every load-bearing warning ("accepted window, do not
  fix", "never gate a shot on telemetry", "do not regress to
  datatype=float") is already in `GeecsBluesky/CLAUDE.md` — cut the code
  copy, never the CLAUDE.md copy.
- Known duplication hotspots: the save-set union rule (written out 4×),
  db_runtime module docstring (≈ verbatim CLAUDE.md M3c), single_shot's
  40-line RunEngine source quote, action_compiler legacy-equivalence told
  3×, the Gate-2 orphan-frame essay in every plan file.
- Gateway: light touch only — condense `udp_client`'s thrice-stated exe-reply
  correlation rationale and `config.py`'s incident retellings; **do not cut**
  the wire-protocol/DB-inheritance semantics (see "keep" below).

## Deliberately KEPT (do not "clean up" later)

- **Gateway wire-protocol + DB-inheritance commentary**
  (`transport/udp_client.py`, `tcp_subscriber.py`, `channels.py`,
  `db/geecs_db.py`, `gateway.py` timestamp/supervisor comments): these
  encode observed real-hardware semantics that exist nowhere else — e.g.
  the reasoning that stops someone "fixing" the deliberately-loose
  comma-tolerant frame parser. High risk if cut.
- **`GeecsDb.get_scan_boundary_writes`** — reserved read-only query for a
  possible DB set-side re-enable (M3c decision); has a pinned test.
- **The fail-loud vs operator-tolerant device-build twins**
  (`_build_request_detectors` in the runner vs `_build_session_devices` /
  `_classify_device_roles` in the bridge) — deliberate parallel, reconcile
  when M4 bridge parity makes the bridge delegate, not before.
- **`session.py`** — healthiest file (21% docstring, mostly API contract);
  the eight near-identical device factories are cohesive and readable; a
  table-driven factory would hurt clarity.
- `_defaults_flag` (runner) — single caller but a well-named tolerance
  contract; inlining produces a nested-ternary mess.
- Test-only-but-public asset API (`register_geecs_handlers`,
  `supports_device_type`, `load_camera_image_from_tiled_run`,
  `build_external_asset_documents`) — documented public surface with tests.

## Deferred until M4 bridge parity (touching now is wasted work)

M4 makes `BlueskyScanner` delegate ScanRequest execution to
`run_scan_request`; the following are scheduled to shrink or die then:

- `bluesky_scanner._reinitialize_from_scan_request` (~150 lines) and the
  `_request_step` machinery — the biggest casualty.
- The runner's action-refusal cluster: `raise_if_actions_present`,
  `resolve_save_sets_checked`, `MULTI_AXIS_MESSAGE` — their only callers are
  the bridge refusals M4 removes.
- `resolve_defaults_for` — single production caller is the bridge.
- Reconciling the device-build twins (above).
- Optional bridge extractions (preflight/event-translation clusters) — clean
  seams, but not the bottleneck; avoid colliding with M4.

## Deferred: analysis/ + assets/ (WIP zone — owner decision 2026-07-10)

The post-run analysis workflow is still being designed; consolidating
scaffolding mid-design is churn. Parked until that design settles:

- `analysis/camera.py` ↔ `analysis/assets.py` share four byte-identical
  helpers (`_day_analysis_dir_from_asset`, `_iter_primary_events`,
  `_optional_int`, `_optional_str`) and parallel orchestrators → merge into
  a shared `analysis/_tiled_common.py`; camera keeps only image-specific
  parts.
- `analysis/assets.py: run_tiled_asset_analysis` — public entry point with
  zero consumers (its camera sibling is notebook-used).
- `analysis/image_analysis.py: resolve_image_analysis_config_dir` —
  test-only compat alias.
- `read_tiled_config()` duplicated between `tiled_integration.py` and
  `assets/tiled_readback.py`; `read_geecs_root_map` re-parses `[Paths]`
  keys that `data_paths` readers already return (three independent `[Paths]`
  parsers exist across the repo).
