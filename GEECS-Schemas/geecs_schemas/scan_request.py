"""ScanRequest — everything you submit to run one scan, in one document.

A scan request says what kind of scan to run (a sweep, a stand-still
statistics run, or an optimization), what to sweep and over which positions,
how many shots to take, what to save, how the trigger is driven, and which
action plans run around it.  Saved presets *are* scan requests; a multi-scan
queue is a list of them.  You would edit one to save a scan you run often.

Developer notes
---------------
This is the one submission object of the target architecture (vision doc
§4.1): clients build a ``ScanRequest`` and call ``session.run(request)``.
Legacy scan presets (``Scan Mode`` / ``Start`` / ``Stop`` / ``Step Size`` /
``Shot per Step`` / ``Num Shots`` / ``Devices`` / ``Info``) convert into it;
``ScanInfo`` and run metadata become projections of it.

Design decisions carried from the legacy system:

- ``shots_per_step`` is declared directly (intent), replacing the legacy
  derivation ``round(rep_rate_hz * wait_time)`` (mechanics).
- The legacy ``Background`` scan mode is a ``noscan`` with the ``background``
  flag set — it was never a distinct acquisition behaviour, only a marker in
  the scan metadata.
- ``acquisition`` defaults to ``strict`` to match the engine default
  (``GEECS_BLUESKY_ACQUISITION_MODE``).
- ``actions.per_step`` exists from day one: "actions between scan steps" is
  composition (a named plan at the step boundary), never a new plan type —
  that is this schema's acceptance test (vision doc §4.5).
- Step scans declare ``axes: [ScanAxis]`` (variable + positions per axis).
  One axis is the legacy 1-D scan; several form an outer-product grid
  (first axis outermost/slowest, last innermost/fastest). The schema is
  axes-only — no top-level ``variable``/``positions`` aliases; converters do
  the adapting. Grid *execution* lands in a later milestone; v1 has no
  traversal-ordering options (see the class docstring).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class ScanRequestMode(str, Enum):
    """What kind of scan to run.

    Attributes
    ----------
    STEP : str
        Sweep one scan variable through a list of positions, taking a batch
        of shots at each.
    NOSCAN : str
        Don't move anything — just collect shots for statistics.
    OPTIMIZE : str
        Let an optimizer choose the next settings each iteration.
    """

    STEP = "step"
    NOSCAN = "noscan"
    OPTIMIZE = "optimize"


class AcquisitionMode(str, Enum):
    """How shots are taken.

    Attributes
    ----------
    FREE_RUN : str
        The trigger free-runs at the machine repetition rate; devices are
        matched up by their timestamps afterwards.
    STRICT : str
        The scan fires each shot itself and requires every device to report
        in before the next one — slower, but nothing is ever missing.
    """

    FREE_RUN = "free_run"
    STRICT = "strict"


class PositionRange(SchemaModel):
    """Scan positions given as start / end / step size.

    The scan visits start, start±step, … up to and including the end (when
    the step divides the range evenly).  Start may be above or below end —
    the direction follows start→end and the sign of ``step`` is ignored.
    """

    start: float = Field(description="First position of the sweep.")
    end: float = Field(description="Last position of the sweep.")
    step: float = Field(
        description=(
            "Spacing between positions. Its sign is ignored — the sweep "
            "direction comes from start and end."
        )
    )

    @model_validator(mode="after")
    def _step_nonzero(self) -> "PositionRange":
        """Reject a zero step size.

        Returns
        -------
        PositionRange
            The validated model.

        Raises
        ------
        ValueError
            If ``step`` is 0.
        """
        if self.step == 0:
            raise ValueError("Position step size must not be 0.")
        return self

    def to_values(self) -> list[float]:
        """Expand the range into the explicit list of positions.

        Returns
        -------
        list of float
            Positions from ``start`` towards ``end`` inclusive (reference
            derivation; a small tolerance absorbs floating-point drift).
        """
        span = self.end - self.start
        step = abs(self.step) * (1 if span >= 0 else -1)
        count = int(abs(span) / abs(self.step) + 1e-9) + 1
        return [self.start + i * step for i in range(count)]


class PositionList(SchemaModel):
    """Scan positions given as an explicit list of values.

    Use this instead of start/end/step when the positions are irregular —
    e.g. ``values: [0.0, 0.5, 2.0, 8.0]``.
    """

    values: list[float] = Field(
        min_length=1,
        description="The exact positions to visit, in the order given.",
    )

    def to_values(self) -> list[float]:
        """Return the positions as a plain list.

        Returns
        -------
        list of float
            The listed positions, unchanged.
        """
        return list(self.values)


# Either shape works in YAML: {start, end, step} or {values: [...]}. The two
# are disjoint under extra="forbid", so smart-union resolution is unambiguous.
Positions = Union[PositionRange, PositionList]


class ScanAxis(SchemaModel):
    """One swept variable and the positions it visits.

    A step scan sweeps one or more axes.  One axis is the familiar 1-D
    scan; several axes form a grid — see :class:`ScanRequest` for how the
    axes loop together.
    """

    variable: str = Field(
        description=(
            "The friendly name of the variable this axis sweeps (from the "
            "experiment's scan-variables catalog)."
        )
    )
    positions: Positions = Field(
        description=(
            "The positions this axis visits, either as {start, end, step} "
            "or as {values: [...]}."
        )
    )


class ActionBindings(SchemaModel):
    """Which named action plans run around (and inside) the scan.

    Each slot lists plan names from the experiment's action library.  Leave
    a slot empty for "nothing".
    """

    setup: list[str] = Field(
        default_factory=list,
        description="Plans to run once before the scan starts.",
    )
    per_step: list[str] = Field(
        default_factory=list,
        description=(
            "Plans to run between scan steps — after each move, before the "
            "shots at that position."
        ),
    )
    closeout: list[str] = Field(
        default_factory=list,
        description="Plans to run once after the scan finishes (even on abort).",
    )


class EvaluatorSpec(SchemaModel):
    """Which analysis code turns raw shots into the number being optimized.

    Points at the Python evaluator class and carries its configuration.
    """

    module: str = Field(
        description=(
            "Python import path of the evaluator module, e.g. "
            "'geecs_scanner.optimization.evaluators.beam_sum_counts_evaluator'."
        )
    )
    class_name: str = Field(
        alias="class",
        description="Name of the evaluator class inside that module.",
    )
    kwargs: dict = Field(
        default_factory=dict,
        description=(
            "Settings passed to the evaluator when it is created — e.g. "
            "which diagnostics/analyzers it should read. Free-form: each "
            "evaluator documents its own options."
        ),
    )

    model_config = SchemaModel.model_config | {"populate_by_name": True}


class GeneratorSpec(SchemaModel):
    """Which optimization algorithm proposes the next settings.

    Names the generator and carries its algorithm-specific options.
    """

    name: str = Field(
        description=(
            "Name of the optimization algorithm, e.g. 'bayes_default', "
            "'random', or 'multipoint_bax_alignment_l2'."
        )
    )
    options: dict = Field(
        default_factory=dict,
        description=(
            "Algorithm-specific tuning options. Free-form: each generator "
            "documents its own options (legacy 'xopt_config_overrides')."
        ),
    )


class OptimizationSpec(SchemaModel):
    """The optimization problem: what to vary, within what limits, to improve what.

    Only used when the scan's mode is ``optimize``.  Lists the variables the
    optimizer may move (with their allowed ranges), what is being minimized
    or maximized, and which algorithm drives the search.

    Notes
    -----
    Mirrors what the legacy optimizer YAML (``BaseOptimizerConfig`` +
    Xopt VOCS) could express: objectives may be empty for BAX-style
    generators that model ``observables`` only; ``constraints`` follow the
    VOCS ``[bound_type, value]`` form.
    """

    variables: dict[str, tuple[float, float]] = Field(
        min_length=1,
        description=(
            "What the optimizer may move and how far, as "
            "'variable name: [lowest, highest]'. Names may be scan-variable "
            "names or 'Device:Variable' strings."
        ),
    )
    objectives: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "What counts as better, as 'objective name: MINIMIZE' or "
            "'objective name: MAXIMIZE'. May be empty for algorithms that "
            "only model observables (BAX)."
        ),
    )
    observables: list[str] = Field(
        default_factory=list,
        description=(
            "Extra measured quantities the algorithm should track without "
            "optimizing them, e.g. ['x_CoM']."
        ),
    )
    constraints: dict[str, tuple[str, float]] = Field(
        default_factory=dict,
        description=(
            "Hard limits on measured quantities, as "
            "'name: [LESS_THAN or GREATER_THAN, value]'. Usually empty."
        ),
    )
    evaluator: EvaluatorSpec = Field(
        description="The analysis code that scores each iteration."
    )
    generator: GeneratorSpec = Field(
        description="The algorithm that proposes the next settings."
    )
    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Stop after this many optimization iterations. Leave unset to "
            "run until stopped by hand."
        ),
    )
    seed_dump_files: list[str] = Field(
        default_factory=list,
        description=(
            "Optional earlier results (ECS dump files) used to warm-start "
            "the optimizer. Usually empty."
        ),
    )
    move_to_best_on_finish: bool = Field(
        False,
        description=(
            "After the optimization ends, drive the variables back to the "
            "best settings found."
        ),
    )

    @model_validator(mode="after")
    def _check_directions(self) -> "OptimizationSpec":
        """Normalize and validate objective directions and constraint bounds.

        Returns
        -------
        OptimizationSpec
            The validated model with directions upper-cased.

        Raises
        ------
        ValueError
            If an objective direction or constraint bound type is unknown.
        """
        normalized = {}
        for name, direction in self.objectives.items():
            upper = direction.upper()
            if upper not in ("MINIMIZE", "MAXIMIZE"):
                raise ValueError(
                    f"Objective {name!r} has direction {direction!r}; "
                    "expected 'MINIMIZE' or 'MAXIMIZE'."
                )
            normalized[name] = upper
        self.objectives = normalized
        for name, (bound, _value) in self.constraints.items():
            if bound.upper() not in ("LESS_THAN", "GREATER_THAN"):
                raise ValueError(
                    f"Constraint {name!r} has bound type {bound!r}; "
                    "expected 'LESS_THAN' or 'GREATER_THAN'."
                )
        return self


class ScanRequest(VersionedSchemaModel):
    """One complete scan, ready to submit: what to do, what to save, how to trigger.

    Fill in the mode (sweep / stand still / optimize), the axis (or axes) to
    sweep, how many shots per position, and the names of the save set,
    trigger profile, and action plans to use.  Saving a request you like
    *is* a preset.

    A step scan may sweep **one axis or several**.  With several axes the
    scan visits every combination (a grid): the *first* axis in the list is
    the outermost, slowest-changing loop and the *last* axis is the
    innermost, fastest-changing one.  ``shots_per_step`` shots are taken at
    each grid point, and each grid point is one bin in the scan data.

    Notes
    -----
    Name-valued fields (``save_set``, ``trigger_profile``, entries in
    ``actions``) are resolved against the experiment's config library at
    submission time; this model checks shape and mode consistency, not name
    existence.

    v1 grid semantics are a plain outer product in list order.  Traversal
    ordering options (snake/raster, per-axis direction) are deliberately not
    modelled yet; a future ``ordering`` field on this model is the
    anticipated extension point.
    """

    mode: ScanRequestMode = Field(
        description=(
            "What kind of scan: 'step' sweeps one or more axes, 'noscan' "
            "collects shots without moving anything, 'optimize' lets an "
            "algorithm pick the settings."
        )
    )
    axes: list[ScanAxis] = Field(
        default_factory=list,
        description=(
            "For step scans: what to sweep. One entry is a simple 1-D scan; "
            "several entries form a grid visiting every combination, with "
            "the first axis as the outermost (slowest) loop and the last as "
            "the innermost (fastest). Leave empty for noscan and optimize."
        ),
    )
    shots_per_step: int = Field(
        1,
        ge=1,
        description=(
            "How many shots to take at each scan position / grid point (or "
            "in total for a noscan)."
        ),
    )
    acquisition: AcquisitionMode = Field(
        AcquisitionMode.STRICT,
        description=(
            "'strict' fires shot by shot and guarantees every device is in "
            "every row; 'free_run' lets the trigger run at the machine rate "
            "and matches devices up by timestamp."
        ),
    )
    save_set: Optional[str] = Field(
        None,
        description=(
            "Name of the save set listing the devices this scan REQUIRES — "
            "the ones that get guarantees (completeness, dialogs, images, "
            "rituals). Unset means no required devices beyond scan "
            "bookkeeping."
        ),
    )
    background_telemetry: Optional[bool] = Field(
        None,
        description=(
            "Also log every other live experiment device as best-effort "
            "snapshot columns — the variables the GEECS experiment database "
            "marks for scan logging (MySQL table expt_device_variable, "
            "get='yes') — read from the gateway's always-on monitor cache: "
            "read-only and never waited on, so it cannot slow or stall the "
            "scan; dead devices are dropped with a log line, never a dialog "
            "or abort. Leave unset to inherit the experiment default; set "
            "true/false to override for this scan."
        ),
    )
    trigger_profile: Optional[str] = Field(
        None,
        description=(
            "Name of the trigger profile that drives the shot trigger. "
            "Unset means the scan does not manage the trigger."
        ),
    )
    trigger_variant: Optional[str] = Field(
        None,
        description=(
            "Optional variant of the trigger profile to use, e.g. "
            "'laser_off'. Leave unset for the profile's base behaviour."
        ),
    )
    actions: ActionBindings = Field(
        default_factory=ActionBindings,
        description=(
            "Named action plans to run before the scan (setup), between "
            "steps (per_step), and after it (closeout)."
        ),
    )
    description: str = Field(
        "",
        description=(
            "Free-text note about this scan; it ends up in the scan's "
            "metadata and the experiment log."
        ),
    )
    background: bool = Field(
        False,
        description=(
            "Mark this scan's data as background/calibration shots so "
            "analysis can find them later."
        ),
    )
    optimization: Optional[OptimizationSpec] = Field(
        None,
        description=(
            "The optimization problem definition. Required for (and only "
            "allowed with) mode 'optimize'."
        ),
    )

    @model_validator(mode="after")
    def _check_mode_consistency(self) -> "ScanRequest":
        """Cross-check the fields each mode requires or forbids.

        Returns
        -------
        ScanRequest
            The validated model.

        Raises
        ------
        ValueError
            If required fields for the mode are missing, or fields that
            don't apply to the mode are set.
        """
        if self.mode is ScanRequestMode.STEP:
            if not self.axes:
                raise ValueError(
                    "A 'step' scan needs at least one entry in 'axes' to say "
                    "what to sweep."
                )
            seen: set[str] = set()
            for axis in self.axes:
                if axis.variable in seen:
                    raise ValueError(
                        f"Axis variable {axis.variable!r} appears more than "
                        "once — each axis must sweep a different variable."
                    )
                seen.add(axis.variable)
        elif self.axes:
            raise ValueError(
                f"'axes' only applies to 'step' scans, not "
                f"{self.mode.value!r}. (An 'optimize' scan declares its "
                "variables inside the 'optimization' block.)"
            )
        if self.mode is ScanRequestMode.OPTIMIZE:
            if self.optimization is None:
                raise ValueError(
                    "An 'optimize' scan needs the 'optimization' block "
                    "(variables, evaluator, generator)."
                )
        elif self.optimization is not None:
            raise ValueError("'optimization' is only allowed when mode is 'optimize'.")
        if self.trigger_variant is not None and self.trigger_profile is None:
            raise ValueError("'trigger_variant' needs 'trigger_profile' to be set too.")
        return self

    def grid_shape(self) -> tuple[int, ...]:
        """Return how many positions each axis visits, outermost first.

        Returns
        -------
        tuple of int
            One count per axis, in list order (empty for noscan/optimize).
        """
        return tuple(len(axis.positions.to_values()) for axis in self.axes)

    def n_steps(self) -> int:
        """Return the total number of grid points the scan visits.

        Returns
        -------
        int
            The product of the axis lengths; 1 when there are no axes (a
            noscan is one motionless bin).
        """
        total = 1
        for count in self.grid_shape():
            total *= count
        return total
