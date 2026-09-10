# Phase 2 — the preamble and finalize as one RunEngine preprocessor

Status: **built and hardware-accepted** (PR #809, adversarially reviewed).
Branch `phase/02-preamble-preprocessor` off `feature/native-bluesky-plans`
(phase 1 merged, #808). Issue #807. What actually shipped, and the two
places it differs from this design, are in the section at the end.

## Goal

Everything the funnel plan does around the acquisition — validate, resolve,
connect, claim the scan number, write ScanInfo, wire saving, build the shot
controller, run setup actions; then save-off, disarm, closeout, disconnect —
moves into **one preprocessor installed on the RunEngine**, keyed on run
metadata. A stock `bluesky.plans` verb then gets the identical preamble with
no GEECS code in the plan:

```python
RE(bp.list_grid_scan([cam], U_S1H.current, pts, md={"geecs": request}))
```

## The seam: `open_run`, verified

`bps.open_run` yields `Msg("open_run", **md)`, so a plan's `md={"geecs": …}`
arrives as `msg.kwargs["geecs"]`, and the RunEngine reads `msg.kwargs` only
when it processes the message. A `plan_mutator` on `open_run` can therefore
**run the whole preamble first and then forward a mutated `open_run`** —
which is what makes the claim-before-the-run-opens invariant survive the
move. Probed on bluesky 1.15.0:

```
order: ['preamble ran (connects, claim)', 'open_run forwarded']
scan_number: 42 | scan_folder: /x/Scan042 | geecs block preserved | plan_name: count
```

Two mechanics the probe settled:

- **Re-entrancy.** `plan_mutator` re-processes the message the inserted plan
  yields, so re-wrapping `open_run` recurses (`ChainMap` of `ChainMap`…,
  `RecursionError`). The forwarded message's `id()` goes in a `forwarded`
  set before it is yielded — the same shape as phase 1's `seen` guard in
  `connect_on_demand`.
- **Metadata injection** is `inject_md_wrapper`'s technique:
  `msg._replace(kwargs=ChainMap(claimed, msg.kwargs))`.

### Footgun, pinned by a test

`RE(plan, geecs={...})` — per-call metadata — is **invisible to
preprocessors**: the RunEngine merges `_metadata_per_call` at `_open_run`,
it never enters a message. The request must ride in the **plan's `md=`
kwarg**. Per-call metadata would put a `geecs` block in the start document
while the preamble silently never ran, so a test pins that difference and
the docstring says it.

## Two corrections to the plan of record

1. **`SupplementalData` cannot carry GEECS telemetry — dropping it from
   phase 2.** `00_overview.md` listed "background telemetry →
   `SupplementalData`". That is wrong: today the telemetry group is appended
   to the detector list (`all_detectors = detectors + telemetry_readables`),
   so its variables are **extra columns in the `primary` stream, one value
   per row**. `SupplementalData`'s `baseline` reads once after `open_run`
   and once before `close_run` into a **separate `baseline` stream**, and
   `monitors` emit asynchronous per-signal streams. Using it would change
   the event schema, break the document-parity tests, and change what
   `background_telemetry` means. Telemetry therefore stays in the read set;
   which devices are read is a per-step/registration concern (phase 3).
   Recorded here rather than quietly; cheap to veto if the stream change is
   actually wanted.
2. **The request travels in the plan's `md=`**, not as RunEngine per-call
   metadata (above).

## Design — one preamble implementation, two callers

The preamble is **extracted, not copied**. Sequencing keeps every commit
green and never leaves two implementations alive:

| step | change | what proves it |
|---|---|---|
| A | Extract the funnel's preamble/finalize body into plan-stub functions in a new `plans/preamble.py` (pure move). `geecs_scan_request_plan` calls them. | the existing suite, unchanged — document parity, pre-claim invariants, run-wrapper pins |
| B | Add `preprocessors.geecs_preamble`, keyed on `md["geecs"]`, calling **the same functions**. | a new parity test: a stock `bp.count` / `bp.list_grid_scan` with `md={"geecs": request}` produces the same start/descriptor/event/stop documents as the funnel for the equivalent request |
| C | Startup installs it, then re-installs `connect_on_demand` last (it must stay outermost). | `test_qserver_startup.py` |

The funnel keeps calling the extracted functions until **phase 5** retires
it; there is never a second copy of the preamble.

### What moves, what waits

From the audited walkthrough (`scan_request_plan.py:439`–`:782`):

| step | destination |
|---|---|
| validate + resolve, trigger profile → `ShotController`, save sets/rituals, scalar policy, `devices_config` | **preprocessor** (pre-claim, fail-fast) |
| unserved / CONNECTED preflights, action-slot assembly, axis resolution, capture toggle + daemon heartbeat | **preprocessor** (pre-claim) |
| device construction + `_connect_in_batches`, telemetry connect, `controller.connect_setters` | **preprocessor**, but for a stock plan the devices are **namespace nouns already connected by `connect_on_demand`** — the preamble connects only what it creates itself (action signals, the controller's setters, telemetry) |
| `claim_scan_number` → `scan_number`, `scan_folder` | **preprocessor**, immediately before forwarding `open_run` |
| `_write_scan_info`, `_configure_saving`/`_configure_assets`, capture-dir mkdir | **preprocessor** (post-claim, pre-start-doc — `run_wrapper` does the mkdir from `md["capture_devices"]` today and that side effect must keep its position) |
| the `md` block (`build_step_scan_spec` + `geecs_run_wrapper`'s additions) | **preprocessor**, injected into `open_run` |
| `scan_log` attach | **preprocessor**, around the forwarded run |
| save-on windowing, per-step actions, shots per step | **phase 3** (per-step function) — the preamble only makes them available |
| pause quiescer → `Pausable` on the controller | **phase 4** |
| free-run t0 sync / reference pacing | **phase 6** |
| client-side pre-submit preflight | stays client-side |

### The finalize chain keeps its nesting

Innermost-first, unchanged in order (`orchestration.py:197`,
`run_wrapper.py:255`): save-off (innermost, so saving stops while the
trigger is still stopped) → `controller.disarm()` → closeout actions →
`_restore_movables` → stage/unstage → `_disconnect_plan` (outermost). In the
preprocessor these wrap the forwarded run instead of the inner plan; the
audit confirms every one is already a plan stub, so they compose unchanged.

### Metadata is a contract, not a convenience

The preprocessor must inject every key downstream requires, because the
audit found six consumers that read them: the s-file callback (`scan_number`,
stop `exit_status`), the Tiled→s-file export (`geecs_scalar_headers`,
`scan_number`, `scan_folder`), `tiled_schema` (`reference_device`, `motor`,
`plan_name`, `grid_shape`, `scan_axes`, `num_points`, `shots_per_step`),
`tiled_catalog` (+`save_sets`, `experiment`, `acquisition_mode`,
`num_grid_points`, `description`), the capture daemon (`capture_devices`,
`nonscalar_save_paths`), and asset readback (`scan_number`, `experiment`,
`scan_folder`). A parity test compares the **whole start document minus
`uid`/`time`** between the funnel and a stock plan — the existing
`_start_essence` helper already does exactly this for the two funnel doors,
so it is reused rather than reinvented.

Keys a stock plan cannot supply from the request alone (`motor`,
`positions`, `num_points`, `shots_per_step`, `plan_pattern`) come from the
stock plan's own metadata, which is **better** than the funnel's hand-built
versions — that is the point of the migration. The parity test therefore
compares the GEECS-owned subset exactly and asserts the stock-owned keys are
present and consistent, rather than byte-identical.

## Tests that must be re-expressed (not deleted)

Two existing tests pin the preamble's **position** rather than its effect
and will move with it:

- `test_creating_the_generator_does_no_work` — no resolution at generator
  creation. Still true; the preprocessor does nothing until `open_run`.
- `test_devices_and_claim_happen_inside_the_running_plan` — asserts
  `RE.state == "running"` at the detector factory and at the claim. Still
  true, and now also true for a stock plan; the assertion moves to the
  preprocessor's call sites.

Also: two tests monkeypatch `claim_scan_number` **by module path**
(`geecs_bluesky.plans.scan_request_plan.claim_scan_number`); the extraction
changes that path, so the patch targets move to `plans/preamble.py`.

Full list to keep green (from the audit): `test_scan_request_plan.py`
(parity `:1148/:1175/:1210`, pre-claim `:292/:335/:350/:373/:561/:1439`,
named-plan parity `:1933/:1963`, submission, telemetry, optimize, pause),
`test_run_wrapper.py`, `test_pause_checkpoints.py`, `test_operator_abort.py`,
`test_read_path_staging.py`, `test_grid_plans_message_level.py`,
`test_native_image_save.py`, `test_sfile_callback.py`,
`test_scan_request_runner.py`, `test_qserver_startup.py`,
`test_connect_on_demand.py`.

## Reuse ledger (draft — final version in the PR body)

| new symbol | reuses | replaces | new because |
|---|---|---|---|
| `plans/preamble.py` (extracted stubs) | the funnel's own body, moved verbatim | the same code in `scan_request_plan.py` | one implementation, two callers |
| `preprocessors.geecs_preamble` | the extracted stubs, `plan_mutator`, `ChainMap` md injection (`inject_md_wrapper`'s technique) | (phase 5) the funnel's role as the only preamble door | stock plans have no GEECS preamble |
| `install_geecs_preamble` | phase 1's `install_connect_on_demand` shape | — | ordering rule: preamble inside, connect-on-demand outermost |

Nothing here re-solves validation, resolution, claiming, ScanInfo, saving,
actions or the shot controller: all of it is the funnel's existing code,
moved once.

## Open for Sam

- **Telemetry** (correction 1): keep it as `primary`-stream columns, no
  schema change — or accept `SupplementalData`'s separate `baseline` stream?
  Recommendation: keep, and revisit only if the baseline stream is wanted
  for its own sake.
- **Optimize mode** builds its own `md` and its own finalize chain inline
  (`scan_request_plan.py:945`–`:1181`). Phase 2 can either bring it along or
  leave `geecs_optimize` as the one plan that keeps an in-plan preamble
  until the optimization work lands. Recommendation: leave it; it is not a
  stock plan and phase 3's registration table does not cover it.

## What shipped (and where it differs from the design above)

Steps A, B and C all landed: the preamble is extracted (`plans/preamble.py`),
the preprocessor calls it, and `qserver/startup/startup.py` installs it with
`connect_on_demand` re-appended outermost. The parity test specified above
exists — it compares every GEECS-owned start-document key exactly and the
event columns as an exact set — and it earned its place immediately by
finding four divergences during review (a missing `acq_timestamp` s-file
header, phantom headers for unread settable children, a stray raw `geecs`
key in the start document, and six missing execution keys that
`tiled_catalog`/`tiled_schema` read).

Three differences from the plan above, all forced by the same fact — the
funnel guaranteed things **by construction** that this door can only enforce
or refuse:

1. **The read set is no longer the save set.** The funnel built one device
   list and handed it to the plan; here the caller passes the detectors and
   the save set selects namespace devices. A plan that does not read every
   saved device is **refused** before the claim, because native saving on an
   unread device writes frames with no `acq_timestamp` row to join them to.
2. **Scan-axis topologies are refused, not degraded.** `build_movable`
   dispatches a catalog target onto pseudo, confirm-elsewhere and
   tolerance-checked-motor; the namespace builds children from the DB alone
   and knows none of that until phase 3. A target needing one of those
   raises rather than quietly becoming a fire-and-forget setpoint.
3. **Devices must be reset between runs.** Long-lived nouns keep whatever a
   run configured. `GeecsNamespace.reset_run_configuration()` runs before
   and after each preamble; the mixins chain through
   `devices/reset_support.py`.

On correction 1 (telemetry): confirmed, and implemented as **inserted reads
between the plan's last `read` and its `save`** — same row, same `primary`
stream, no schema change — rather than `SupplementalData`.

**Known gap, owned by phase 3:** stock plans emit no `ScanContext` columns
(`bin_number`, `scan_event_index`, `shot_index_in_bin`). Those arrive with
the per-step function, which is also when `ScanContext` retires. The parity
test asserts them as the *only* column difference, so the gap cannot widen
unnoticed.
