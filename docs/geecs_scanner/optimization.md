# Running optimizations

The **Optimization** scan mode turns a scan into a closed feedback loop: instead
of sweeping a variable over a fixed grid, an optimization algorithm proposes
device setpoints, the scanner sets them and acquires shots, an *evaluator* turns
that data into a number (the objective), and the algorithm uses it to decide
where to look next. This page is a practical guide to configuring and running
one — what the config file means, which algorithm to pick, how the objective is
computed, and the pitfalls worth knowing before your first run.

It is operator-focused. For the underlying classes see the
[Base Optimizer](api/optimization/base_optimizer.md) and
[Base Evaluator](api/optimization/base_evaluator.md) API references; the
optimization layer wraps [Xopt](https://github.com/xopt-org/Xopt) for the
algorithms.

## How the loop works

Every optimization is assembled from four pieces, all declared in one YAML file:

| Piece | What it is | Config key |
|---|---|---|
| **VOCS** | The *problem*: which variables to tune (with bounds) and what to optimize | `vocs` |
| **Generator** | The *strategy*: the algorithm that proposes the next setpoints | `generator` |
| **Evaluator** | The *objective*: turns acquired shot data into a number (or observables) | `evaluator` |
| **Optimizer** | The glue that runs the loop and talks to the scan engine | the file as a whole |

Each scan step then runs this cycle:

1. The **generator** proposes values for the `vocs` variables (the first couple
   of steps are random, to seed the model).
2. The scan engine **sets those devices** and **acquires** the configured number
   of shots.
3. The **evaluator** reads the just-acquired bin of data and computes the
   objective `f` (and/or observables).
4. The result is fed back to the generator, which proposes the next point.

The loop repeats until you stop the scan. Optionally, at the end the scanner can
move the devices to the **best observed setpoint**.

## The optimization config file

You point the GUI's Optimization mode at a single YAML file. Here is a complete,
annotated example — a Bayesian optimization that maximizes camera counts on one
diagnostic:

```yaml
# VOCS — the optimization problem
vocs:
  variables:
    U_Hexapod:ypos: [-22.1, -21.5]    # device:variable -> [min, max]
  objectives:
    f: MINIMIZE                        # the evaluator returns "f"; MINIMIZE or MAXIMIZE
  constraints: {}                      # optional, usually empty

# Evaluator — how to turn acquired data into "f"
evaluator:
  module: geecs_scanner.optimization.evaluators.beam_sum_counts_evaluator
  class: MaxCountsEvaluator
  kwargs:
    analyzers:                         # diagnostic-driven data source
      - diagnostic: UC_TC_Output       # a diagnostic config (configs repo)
        scan:
          mode: per_bin                # average the bin, then analyze once

# Generator — the algorithm
generator:
  name: bayes_turbo_standard
```

### Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `vocs` | yes | Variables (`device:variable: [min, max]`), `objectives` (`f: MINIMIZE`/`MAXIMIZE`), optional `constraints`, optional `observables` (for BAX) |
| `evaluator` | yes | `module` + `class` of the evaluator, plus `kwargs` passed to it |
| `generator` | yes | `name` of the algorithm (see [Choosing a generator](#choosing-a-generator)) |
| `xopt_config_overrides` | no | Per-generator tuning, keyed by generator name (e.g. `beta`, BAX probe settings) |
| `seed_dump_files` | no | Paths to prior `xopt_dump.yaml` files to warm-start the model |
| `move_to_best_on_finish` | no | If `true`, move devices to the best observed setpoint when the scan ends |
| `device_requirements` | no | Usually omit — auto-generated from the evaluator's `analyzers` |

!!! tip "Variable names are GEECS `device:variable` strings"
    Use the exact `device:variable` form, e.g. `U_Hexapod:ypos` or
    `U_EMQTripletBipolar:Current_Limit.Ch1`. Names with spaces, colons, and
    dots are fine.

## Choosing a generator

Set `generator.name` to one of:

| Name | Use it when |
|---|---|
| `random` | Baseline / debugging — uniform random sampling, no model |
| `bayes_default` | Good default Bayesian optimization (Expected Improvement) for a single smooth objective |
| `bayes_ucb` | Noise-robust alternative to EI (Upper Confidence Bound). Tune exploration with `beta` (default 2.0) |
| `bayes_ucb_explore` | Pure exploration / surrogate-building (UCB with high `beta`) — maximizes model knowledge, not the objective |
| `bayes_turbo_standard` | Higher-dimensional or noisier problems — EI inside a TuRBO trust region that localizes the search |
| `bayes_turbo_ucb` | TuRBO trust region with UCB instead of EI (configurable `beta`) |
| `bayes_cheetah` | Specialized surrogate-driven BO (requires the optional `cheetah` package) |
| `multipoint_bax_alignment` / `_l2` | Beam alignment via [BAX](#bax-alignment) — observables-only, minimizes a virtual slope objective |

Per-generator knobs go under `xopt_config_overrides`, keyed by the generator
name:

```yaml
generator:
  name: bayes_ucb
xopt_config_overrides:
  bayes_ucb:
    beta: 4.0          # more exploration
```

## Defining the objective: evaluators

The evaluator is the only piece you may need to write Python for, and even then
it is small. It reads the just-acquired bin of shot data and returns a
dictionary. There are two data sources, set in the evaluator `kwargs`:

- **`analyzers`** — diagnostic-driven. Each entry names a *diagnostic* (the same
  YAML diagnostics ScanAnalysis uses) that runs against the acquired images and
  produces namespaced scalars like `UC_TopView_x_fwhm`. Entries are a bare stem
  (`- UC_TopView`) or a dict with overrides (`- {diagnostic: UC_FROG, scan: {mode: per_bin}}`).
- **`scalars`** — column names read directly from the current-bin data frame
  (s-file columns), used verbatim.

A subclass implements one or both hooks:

```python
from geecs_scanner.optimization.base_evaluator import BaseEvaluator

class MyObjective(BaseEvaluator):
    def compute_objective(self, scalars: dict, bin_number: int) -> float:
        # scalars holds analyzer outputs + raw scalar columns for this bin
        return -scalars["UC_TopView_total_counts"]   # minimize -> maximize counts
```

The built-in evaluators in `geecs_scanner.optimization.evaluators` cover the
common cases:

| Class | Objective |
|---|---|
| `MaxCountsEvaluator` | Total counts on a camera diagnostic |
| `BeamSizeEvaluator` | Beam size (FWHM) |
| `BeamPositionEvaluator` | Beam centroid position |
| `BeamPositionSimulationEvaluator` | Synthetic centroid from a ray-tracing model — no camera needed (used to exercise the loop on hardware) |
| `EBeamSourceOpt` | E-beam source optimization (mag-spec) |

!!! note "`device_requirements` is auto-generated"
    You normally do **not** write a `device_requirements` block. The optimizer
    derives it from the diagnostics in `analyzers`, so the cameras those
    diagnostics need are saved automatically. (See the gotcha below for the one
    case this doesn't cover.)

## BAX alignment

BAX (Bayesian Algorithm Execution) is a different kind of generator used for
**alignment**: rather than optimizing a measured objective directly, it builds a
model of one or more *observables* and computes a *virtual* objective from them
— here, minimizing the slope of a beam-position observable as a "measurement"
variable is swept.

The key differences from a normal optimization:

- The VOCS is **observables-only** — it declares `observables`, **not**
  `objectives`. The optimization target is the algorithm's virtual objective, so
  the generator requires zero objectives.
- The evaluator returns observables (e.g. `x_CoM`), not an `f`.
- The probe behavior is configured under `xopt_config_overrides`.

A worked BAX-alignment config (driving real correctors, with a synthetic
centroid so the full control loop can be exercised without the beam diagnostic):

```yaml
vocs:
  variables:
    U_S1V:Current: [-4, 4]                              # control (the BAX "S")
    U_EMQTripletBipolar:Current_Limit.Ch1: [1.2, 1.7]   # measurement (the BAX "M")
  observables:
    - x_CoM                                             # no `objectives:` block

evaluator:
  module: geecs_scanner.optimization.evaluators.beam_position_evaluator_simulation
  class: BeamPositionSimulationEvaluator
  kwargs:
    control_variable_name: U_S1V:Current
    measurement_variable_name: U_EMQTripletBipolar:Current_Limit.Ch1

generator:
  name: multipoint_bax_alignment_l2

xopt_config_overrides:
  multipoint_bax_alignment_l2:
    control_names: ["U_S1V:Current"]
    measurement_name: "U_EMQTripletBipolar:Current_Limit.Ch1"
    observable_names: ["x_CoM"]
    probe_nominal: 1.5
    probe_grid_absolute: [-0.25, 0.0, +0.25]
    n_control_mesh: 21
    mesh_measurement: true
    n_measurement_mesh: 5
    n_monte_carlo_samples: 32
    use_low_noise_prior: false
```

## Seeding and finishing

- **Warm-start** a run from previous data by listing prior dumps under
  `seed_dump_files`. Each scan writes an `xopt_dump.yaml` into its scan folder;
  point a new run at one (or several) to seed the model. Compatibility is
  checked: variable, objective, and observable names must match.
- **Move to best** by setting `move_to_best_on_finish: true`. At scan end the
  scanner moves devices to the best observed (feasible, non-errored) setpoint.
  For observables-only BAX runs there is no single "best objective" row, so the
  devices fall back to their initial state.

## Gotchas

!!! warning "BAX VOCS must be observables-only"
    A `multipoint_bax_alignment*` generator requires **no** `objectives:` block —
    only `observables:`. Leaving an `objectives:` block in a BAX config will fail
    at generator construction. (Conversely, every non-BAX generator requires an
    objective.)

!!! warning "Simulation / observables-only runs still need data logged"
    Evaluators read the acquired bin of shot data. If your scan has **no save
    devices**, nothing is logged and the evaluator fails with a cryptic
    `KeyError: 'Elapsed Time'`. A synthetic evaluator (e.g.
    `BeamPositionSimulationEvaluator`) does not pull in any cameras, so you must
    add the control/measurement devices to your **save elements** yourself so
    their setpoints are recorded.

!!! warning "Diagnostics must match the current schema"
    Diagnostics referenced under `analyzers` are validated against the current
    ImageAnalysis diagnostic schema (`extra="forbid"`). A stale diagnostic YAML
    — camera fields at the top level, or a missing `image_analyzer` — fails with
    a `DiagnosticAnalysisConfig` validation error. Update the diagnostic in the
    configs repo to the current `image:` / `scan:` shape.

## Where to next

- [Base Optimizer API](api/optimization/base_optimizer.md) — the wrapper class and its methods
- [Base Evaluator API](api/optimization/base_evaluator.md) — writing evaluators
- [Generator Factory API](api/optimization/generators/generator_factory.md) — the full generator registry
- [Basic Optimization Setup](examples/optimization/optimization_example.ipynb) — runnable example notebook
