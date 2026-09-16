# Scan composer and the Sweep payload

**Status: design settled with the owner, 2026-09-16; foundation in progress.**
Build from `feature/native-bluesky-plans` after optimization #920, merged as
`05c0fc23`. The owner reports optimization hardware acceptance, including live
optimization runs. That acceptance does not cover the sweep plan cutover.

This brief supersedes the 2026-09-15 draft and records the follow-up design
discussion. The original mockup is a visual reference, not production code.
The newer inline mockup reflects the agreed placement and layout, but its
fixed two-axis examples are not schema limits. Neither is code to lift into
GeecsScanner.

## 1. Evidence already established

See [verified Bluesky facts](12_scan_composer/verified_bluesky_facts.md).
Those facts were established against installed Bluesky 1.15.1, not inferred
from documentation; do not re-derive them to restart the design discussion.

The current worker binds stock scan verbs with GEECS acquisition hooks, and
the scanner's fixed form exposes only some of their argument shapes. Presets
are plan calls; the client resolves device and catalog references before
submission. The replacement preserves that single-description contract.

## 2. Owner decisions (closed)

- **Operator choices: Count / Sweep / Optimize.** Keep `count` as a separate
  registered plan and UI choice. It signifies acquisition without commanded
  variable motion, preserving `Plan = count`, `ScanMode = noscan` (or
  `background` when explicitly marked). A sweep contains at least one axis;
  an empty axis list is not a count.
- **The moving predetermined plan is named `sweep`.** Every moving
  predetermined trajectory, including typed patterns, uses this plan.
  `optimize` is adaptive; `mv`, `run_action`, `measure_shot_offsets` and
  `check_shot_sync` retain their utility roles.
- **Our payload is `Sweep`, not scanspec.** Scanspec lacks arbitrary-position
  lists and logarithmic primitives, both permanent needs. Arbitrary nesting,
  masks, concatenation and fly scans are out of scope. Use familiar
  `start`/`stop`/`num` and `zip`/`product` vocabulary where it fits.
- **One to many axes, with no artificial two- or three-axis ceiling.** A
  five-variable correlated list scan is a real use case. The editor starts
  with one unconfigured axis and offers add/remove. In a correlated sweep,
  five lists of ten positions mean ten steps, not 100,000 combinations.
- **Correlated axes require equal point counts**, for ranges, lists, logs
  and mixtures. Never truncate. A grid visits all combinations and allows
  different counts. A one-axis grid is the same traversal as a single list,
  so removing axes need not force a new combination mode.
- **Spacing and relative/absolute are per axis.** Relative means offsets from
  the run's starting readback. Relative axes must return to their starting
  positions on completion or abort. Do not conflate this with a pseudo
  positioner's own relative definition or restore lifecycle.
- **Patterns are included in v1:** spiral, square spiral, Fermat spiral and
  x2x, under the Patterns tab. They are typed trajectory variants inside
  `Sweep`, not separately registered plans. The stock x2x meaning is retained:
  both axes relative, Y traverses half X's range.
- **Position lists are numeric input, not expressions.** Accept commas,
  tabs, spaces and newlines. Validate every token and reject malformed or
  non-finite values; preserve order and repetitions. Show the parsed count
  and identify bad input. No NumPy evaluator, file upload, or prior-result
  integration is required for v1.
- **All numerical expansion stays in Python.** NumPy supplies range/log
  spacing; Bluesky plan_patterns supplies traversal and pattern geometry.
  Preview and execution call the same expansion code; JavaScript only draws
  returned points. Log bounds are explicitly decade exponents.
- **Schemas stay lightweight.** Models and validation live in
  `geecs_schemas`; numerical expansion and cycler construction live in a
  hardware-free `geecs_bluesky.trajectory` module. #920 added GEST VOCS to
  the schema package; do not add NumPy, cycler or Bluesky there.
- **Use GeecsWebTheme's theme, kit, helpers and guards.** Production styles
  consume tokens. Reusable components belong in the kit with a reference-page
  example; scanner-specific layout stays in the scanner. Neither prototype's
  copied CSS nor its JavaScript trajectory mathematics is production code.
- **The composer expands inside the existing New scan panel.** No new route,
  modal or launcher-first interaction. About 80% of scans are hand-configured
  during the session. Preserve preset loading in the panel footer beside
  Save as preset and Start. Presets seed the editable form.
- **Shared capture and devices controls** belong to New scan, outside the
  Axis Sweeps / Patterns tabs. Keep Strict/Gated, trigger profile, shot count,
  period, description, device selection, saving and essential settings.
- **Fresh page: unconfigured form**, not an automatically selected preset or
  invented trajectory/devices. Sharing unsaved drafts is not required; save
  as a preset to share. Existing scan presets may be migrated or retired;
  preserve optimization configs. The owner authorizes the eventual corpus
  changes directly on the configs repository's `main`.
- **No parallel execution paths at cutover.** Delete the bound moving stock
  verbs in the same PR that registers `sweep`, and migrate the readers and
  client expansion then. Retain historical readers for old runs/items.
  `count` is deliberately retained. The old brief's “14 verbs” was not an
  accurate enumeration; derive the deletion set from the current registry.

## 3. Payload and expansion foundation (PR 1)

`Sweep` is a `SchemaModel` containing a kind-discriminated `trajectory`:

```yaml
trajectory:
  kind: axes
  combine: zip
  snake: false
  axes:
    - kind: range
      axis: example_x
      relative: true
      start: -1
      stop: 1
      num: 3
    - kind: list
      axis: example_y
      relative: false
      positions: [2, 4, 8]
```

`RangeAxis` uses `start`, `stop`, `num`; `ListAxis` uses `positions`;
`LogAxis` uses `start_exp`, `stop_exp`, `num`. Axis names are nonblank,
numeric inputs finite, counts positive integers, lists nonempty, axes unique,
zip counts equal, and snake restricted to grids. The schema cannot detect
two different aliases of one device: expansion must reject those too.

Typed variants are `spiral`, `spiral_fermat`, `spiral_square`, and `x2x`.
They carry X/Y axis references (with coordinate frames) and typed pattern
parameters. The plan name remains `sweep`. The enclosing preset remains the
versioned document; the payload does not add a duplicate version stamp.

`Sweep.axis_references()` exposes names and relative flags in authored order.
`n_steps()` provides an exact cheap count for axis sweeps, square spirals and
x2x; curved spirals return `None` until Bluesky has expanded them. Never port
the spiral loop into a schema just to count it.

`geecs_bluesky.trajectory.axis_positions` and `sweep_to_cycler` are the one
numerical implementation. The latter takes a name resolver, uses inert
movable/readable stand-ins for pattern generation, refuses aliases mapping
to one key, and returns the cycler keyed by real objects or preview names.
Relative values stay offsets in this pure layer. No namespace import, device
method, DB, gateway connection, acquisition or RunEngine is needed.

This foundation registers no new plan and changes no deployed execution path.
Parity tests compare traversal with installed Bluesky patterns, including
five axes, one-point ranges, mixed spacing, descending ranges, repeats,
unequal grids, snake order and typed patterns. Numerical overflow/underflow
must fail before a trajectory can reach a motor.

## 4. Plan and cutover (PR 2, separate hardware session)

- Register `sweep(detectors, *, sweep=<JSON>, ...)` and validate the payload
  in the worker as well as before submission. Reuse the existing acquisition
  bracket, liveness gate, failure naming, non-essential streaming, throttle
  and hooks. Delegate execution to stock `scan_nd`.
- Client expansion resolves catalog/Device:Variable references once and
  records every reference for the existing preflight. The worker maps only
  expanded bindings to namespace objects; it must not reconstruct devices
  from preset provenance or reload the catalog to reinterpret a run.
- Handle relative offsets and reset using Bluesky mechanisms; capture the
  correct frame after staging relative pseudos. Test normal completion,
  abort, mixed absolute/relative axes, and restore failures. Do not promise
  a hardware restore after a process kill or failed device communication.
- Remove all moving stock registrations and their permissions in this same
  change. Keep `count`, `optimize`, and the utility plans.
- Record the validated trajectory as execution metadata and preserve standard
  `motors`, point counts and dimensional information. Update EVENT_SCHEMA.md,
  ScanInfo's first-axis projection, scanner summaries/progress and Data Utils'
  scan classification. Preserve old plan-pattern/positional decoding for
  historical runs and queue items; new runs use the new structured contract.
- Migrate or retire affected scan presets in the configs repository as part
  of the coordinated cutover. Keep optimization configs. Do not enqueue old
  moving plan names against the new worker.

Acceptance: hermetic RunEngine tests first; then an explicit hardware session
for a 1-D range sweep, a correlated list sweep, and a two-axis snake grid.
Verify moves/restoration, shot/bin counts, s-file rows, ScanInfo keys, portal
classification, and an old scan's continued classification. Test strict and
gated coverage appropriate to the changed execution path. Keep this PR separate
from the composer UI and do not imply #920 supplies this acceptance.

## 5. Preview API (PR 3)

Add a typed trajectory service result and `POST /api/trajectory`, using the
same expansion module with inert keys. It computes without hardware, a DB,
or a worker request. Expand the scanner's allowed import seam explicitly.

Return coordinates, axis identity/frame, total steps, and useful display
information. Invalid input produces the scanner's usual structured errors.
Relative previews show offsets, not a claim about future absolute positions:
the baseline is read at execution and can change while queued.

Before serving arbitrary requests, enforce a bounded expansion/response
budget. Axis products can be checked with `n_steps()` without allocating;
curved patterns require a separate bounded-expansion strategy. Large previews
must clearly disclose any sampling; execution must never silently truncate.
This is an engineering requirement, not an operator shot cap.

## 6. Inline composer (PR 4)

Design references:

- [Original trajectory mockup](12_scan_composer/mockup.html): tone, explanatory
  text, patterns and per-axis plots. Discard verb derivation and JS geometry.
- [Accepted inline layout preview](12_scan_composer/inline_mockup.html): the
  owner liked the layout and clarified add/remove axes and five-axis lists.
  Its fixed example values and disabled add buttons are prototype limitations.

```text
New scan                 [Count | Sweep | Optimize]  [Strict | Gated]
[Axis Sweeps | Patterns]

Sweep settings                  Axes
Together / Grid                 Axis 1: variable, spacing, bounds, points
Snake                           Axis 2: variable, spacing, bounds, points
                                + Add axis

Scan summary: 24 positions x 10 shots = 240 shots
Trajectory plots
> Points as a table       > What gets submitted

Capture: shots, trigger profile, period, description
Devices and saving options
Estimate              Load preset...  Save as preset...  Start
```

Keep axis selection close to the top. Use a narrow settings column beside
the axis rows, with summary/trajectory below; stack on narrow screens.
Patterns uses pattern selection on the left and typed parameters on the
right, sharing the plots and capture/device controls. Count has no trajectory;
Optimize preserves #920's configuration, required-device and progress behavior.

One chart per axis, value against step index; an additional X-Y path for
exactly two axes. Never combine different Y scales. Show points as a table
and the submitted payload behind native disclosures. Keep the snake control
visible, disabled when it does not apply. Short essential explanations and
errors remain visible; longer explanations use an optional disclosure.

Update previews as input changes, using the server response only. Avoid
flicker and stale responses replacing newer ones. Keep the trajectory renderer
independent of form state so a later Now/queue view can reuse it. Plotting
completed points in Now is a follow-on, not part of this arc's acceptance.

The existing Now/queue organization stays for v1; a sticky running-scan strip
is a later refinement if the real screen review shows it is needed.

## 7. Delivery process

Each PR branches from and targets `feature/native-bluesky-plans` and follows
root CLAUDE.md and `/land`: scoped changes, package version bumps/changelogs,
CI-shaped checks, `scripts/commit.sh`, independent adversarial review and
hardware verification/OWED section. No direct commits to an integration
branch in this repository. The owner merges the feature-targeted PRs.

1. Models, shared numerical expansion, parity tests and this decision record.
2. Worker/client/metadata cutover, deletion of old moving verbs, its own
   hardware session and coordinated preset corpus update.
3. Hardware-free preview API and bounded response handling.
4. Kit-based inline composer and owner visual review.

The former PR 0 “submit any preset unchanged” patch is not a prerequisite:
proceed directly with the foundation now that #920 has merged. Preserve
valid preset submission when replacing the form, independent of editability.
Delete this planning directory at arc completion after transferring durable
contracts into package CLAUDE.md files and published documentation.
