"""ScanRequest — everything you submit to run one scan, in one document.

A scan request says what kind of scan to run (a sweep, a stand-still
statistics run, or an optimization), what to sweep and over which positions,
how many shots to take, what to save, how the trigger is driven, and which
action plans run around it.  Saved presets *are* scan requests; a multi-scan
queue is a list of them.  You would edit one to save a scan you run often.

Developer notes
---------------
This is the one submission object of the target architecture: clients
build a ``ScanRequest`` and call ``session.run(request)``.
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
  that is this schema's acceptance test.
- Step scans declare ``axes: [ScanAxis]`` (variable + positions per axis).
  One axis is the legacy 1-D scan; several form an outer-product grid
  (first axis outermost/slowest, last innermost/fastest). The schema is
  axes-only — no top-level ``variable``/``positions`` aliases; converters do
  the adapting. Grid *execution* lands in a later milestone; v1 has no
  traversal-ordering options (see the class docstring).

Schema v2 (the capture refactor):

- The seven capture-concern fields (``shots_per_step``, ``acquisition``,
  ``save_sets``, ``background_telemetry``, ``native_image_save``,
  ``trigger_profile``, ``trigger_variant``) moved into one
  :class:`CaptureSettings` sub-model at ``ScanRequest.capture``.
- ``submission`` left the request document — a :class:`SubmissionRecord` is
  server-stamped lifecycle state, not operator input; it now travels beside
  the request (a separate plan parameter) and still lands in run metadata.
- A ``mode="before"`` validator lifts the flat v1 layout into ``capture``
  (and drops a v1 ``submission`` key), so saved presets, archived run
  metadata, and stale clients keep validating forever.

Schema v3 (drop the trigger variant):

- ``trigger_variant`` left :class:`CaptureSettings`: profile variants were
  never adopted (every experiment kept one profile file per operating
  condition), and :class:`~geecs_schemas.trigger_profile.TriggerProfile`
  v2 removed them.  The same before-validator drops an unset
  ``trigger_variant`` (flat v1 or inside ``capture``) and refuses a set
  one with the remedy (name the condition's own profile); ``schema_version``
  ≤ 2 is normalized to 3.

Schema v4 (drop the native-image-save toggle):

- ``native_image_save`` left :class:`CaptureSettings`.  It existed to say
  "these cameras' frames are captured losslessly elsewhere, so skip their
  LabVIEW per-shot files", and the elsewhere was the central PVA capture
  daemon, deleted with #806.  Its implementation — the engine's
  fail-closed preflight and the per-camera resolution behind it — went
  with the daemon, leaving a field nothing read.  PNG retirement (#738)
  owns the replacement, whose preflight asks the distributed file plugin
  a different question ("armed on every camera in the save set?"), so
  nothing here was reusable.  The same before-validator drops an unset
  ``native_image_save`` and refuses a set one with the remedy;
  ``schema_version`` ≤ 3 is normalized to 4.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import Field, field_validator, model_validator

from geecs_schemas._base import (
    SchemaModel,
    VersionedSchemaModel,
    stale_schema_version,
)


class ScanRequestMode(str, Enum):
    """What kind of scan to run.

    Attributes
    ----------
    STEP : str
        Sweep one scan variable through a list of positions, taking a batch
        of shots at each.
    NOSCAN : str
        Don't move anything — just collect shots for statistics.
    """

    STEP = "step"
    NOSCAN = "noscan"


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

    def n_positions(self) -> int:
        """Count the positions WITHOUT materializing them.

        The arithmetic twin of :meth:`to_values` (same derivation, same
        tolerance).  Size guards must use this, never ``len(to_values())``
        — an agent-composed range like ``{start: 0, end: 1e15, step:
        1e-9}`` validates cleanly, and expanding it to count it is the
        crash the guard exists to prevent.

        Returns
        -------
        int
            How many positions :meth:`to_values` would return.
        """
        span = self.end - self.start
        return int(abs(span) / abs(self.step) + 1e-9) + 1

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
        return [self.start + i * step for i in range(self.n_positions())]


class PositionList(SchemaModel):
    """Scan positions given as an explicit list of values.

    Use this instead of start/end/step when the positions are irregular —
    e.g. ``values: [0.0, 0.5, 2.0, 8.0]``.
    """

    values: list[float] = Field(
        min_length=1,
        description="The exact positions to visit, in the order given.",
    )

    def n_positions(self) -> int:
        """Count the positions (the listed length).

        Returns
        -------
        int
            ``len(values)``.
        """
        return len(self.values)

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


class CaptureSettings(SchemaModel):
    """How shots are taken and what gets recorded — the capture concern.

    Every scan, whatever its mode, captures data the same way: a number of
    shots per step, an acquisition discipline, the save sets naming the
    recorded devices, the telemetry toggle, and the trigger profile
    driving the shot trigger.  This model groups those
    five settings; conceptually they are three sub-groups — shot control
    (``shots_per_step`` + ``acquisition``), data logging (``save_sets`` +
    ``background_telemetry``), and the trigger
    profile — kept one level flat here on purpose.

    Every field has a usable default, so an omitted ``capture`` block is a
    valid one-shot strict capture with no named save sets.
    """

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
    save_sets: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the save sets — reusable named device groups — "
            "recorded for this scan; devices are unioned across them. Each "
            "names the devices that get guarantees (completeness, dialogs, "
            "images, rituals). A bare string is accepted and stored as a "
            "one-element list. Empty means no required devices beyond scan "
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

    @field_validator("save_sets", mode="before")
    @classmethod
    def _coerce_save_sets(cls, value: object) -> object:
        """Coerce a bare save-set name to a single-element list.

        ``save_sets="Amp4In"`` and ``save_sets=["Amp4In"]`` both validate;
        the canonical stored form is always the list. Anything else (a real
        list, or a value of the wrong type) is passed through unchanged for
        the normal list validation to accept or reject.

        Parameters
        ----------
        value : object
            The raw ``save_sets`` value before validation.

        Returns
        -------
        object
            ``[value]`` when *value* is a string, otherwise *value* unchanged.
        """
        if isinstance(value, str):
            return [value]
        return value


class PreflightCheckResult(str, Enum):
    """How one pre-submit check ended.

    Attributes
    ----------
    PASSED : str
        The check found nothing to ask about.
    CONTINUED : str
        The check raised a question and the operator chose to continue
        anyway.
    SKIPPED : str
        The check could not run (for example the database was unreachable)
        and submission went ahead without it.
    """

    PASSED = "passed"
    CONTINUED = "continued"
    SKIPPED = "skipped"


class PreflightOutcome(SchemaModel):
    """One pre-submit check and how it ended, kept for the scan's record.

    A check that would have stopped the submission never produces one of
    these — an aborted submission is never queued, so there is nothing to
    record.
    """

    check: str = Field(
        description=(
            "Name of the pre-submit check, e.g. 'unserved_variables', "
            "'gateway_liveness', 'free_run_staleness'."
        )
    )
    result: PreflightCheckResult = Field(
        description=(
            "How the check ended: 'passed' (nothing found), 'continued' "
            "(the operator saw a warning and chose to go ahead), or "
            "'skipped' (the check could not run)."
        )
    )
    detail: str = Field(
        "",
        description=(
            "What the check found or why it was skipped, in the words the "
            "operator saw. Empty for a clean pass."
        ),
    )


class SubmissionRecord(SchemaModel):
    """Who submitted this request, when, and what the pre-submit checks said.

    Filled in by the submitting client (a front end, a script, an agent) at
    the moment the request is queued — not written by hand.  Since format
    v2 this record is **not part of the request document**: it travels
    beside the request (a separate plan parameter) — server-stamped
    lifecycle state, not operator input.  The scan engine copies it into
    the run metadata for provenance and never acts on it: a submission
    without one runs exactly the same.
    """

    client: str = Field(
        "",
        description=(
            "What submitted the request, e.g. 'geecs-console 0.21.0'. "
            "Free text, for the record only."
        ),
    )
    submitted_at: str = Field(
        "",
        description=(
            "When the request was queued, as an ISO 8601 timestamp with "
            "timezone from the submitting machine's clock, e.g. "
            "'2026-08-21T14:30:00-07:00'. Informational only."
        ),
    )
    preflight: list[PreflightOutcome] = Field(
        default_factory=list,
        description=(
            "The pre-submit checks that ran and how each ended. Empty when "
            "the client ran no checks."
        ),
    )


# The flat v1 top-level ScanRequest fields that live inside ``capture``
# since format v2 — the lifting validator's migration surface.
_V1_CAPTURE_FIELDS = (
    "shots_per_step",
    "acquisition",
    "save_sets",
    "background_telemetry",
    "trigger_profile",
)
_TRIGGER_VARIANT_REMEDY = (
    "'trigger_variant' was removed in ScanRequest format v3 (profile "
    "variants never existed in practice): save the operating condition as "
    "its own trigger profile and name it in 'trigger_profile'."
)
_NATIVE_IMAGE_SAVE_REMEDY = (
    "'native_image_save' was removed in ScanRequest format v4: the central "
    "capture daemon it switched off native saving for was deleted (#806), "
    "and nothing has read the field since. Native per-shot saving is "
    "currently decided per device by the GEECS DB (the 'save' and "
    "'localsavingpath' variables); a scan-time toggle returns with PNG "
    "retirement (#738)."
)
#: Removed fields: dropped when unset, refused when set (no overlay exists).
#: Each maps to the remedy a set value is refused with.
_REMOVED_FIELDS: dict[str, str] = {
    "trigger_variant": _TRIGGER_VARIANT_REMEDY,
    "native_image_save": _NATIVE_IMAGE_SAVE_REMEDY,
}


class ScanRequest(VersionedSchemaModel):
    """One complete scan, ready to submit: what to do, what to save, how to trigger.

    Fill in the mode (sweep / stand still), the axis (or axes) to
    sweep, how many shots per position, and the names of the save sets,
    trigger profile, and action plans to use.  Saving a request you like
    *is* a preset.

    A step scan may sweep **one axis or several**.  With several axes the
    scan visits every combination (a grid): the *first* axis in the list is
    the outermost, slowest-changing loop and the *last* axis is the
    innermost, fastest-changing one.  ``shots_per_step`` shots are taken at
    each grid point, and each grid point is one bin in the scan data.

    Notes
    -----
    Name-valued fields (``capture.save_sets``, ``capture.trigger_profile``,
    entries in ``actions``) are resolved against the experiment's config
    library at submission time; this model checks shape and mode
    consistency, not name existence.  ``capture.save_sets`` is a list — the
    engine resolves each named set and **unions** their devices into the
    recorded device set (a device in more than one set is merged), so
    operators mix and match named diagnostic groups per scan.  A bare
    string is accepted for the single-set case and stored as a one-element
    list.

    Grid semantics are a plain outer product in list order.  Traversal
    ordering options (snake/raster, per-axis direction) are deliberately not
    modelled yet; a future ``ordering`` field on this model is the
    anticipated extension point.

    Format v2 moved the capture-concern fields into ``capture`` and dropped
    ``submission``; format v3 dropped ``trigger_variant`` and format v4
    ``native_image_save`` (see the module docstring).  One before-validator
    lifts every older layout — the flat v1 fields into ``capture``, a v1
    ``submission`` key and an unset removed field dropped — so older
    documents keep validating; a declared ``schema_version`` ≤ 3 is
    normalized to 4 (even on a sparse document with nothing to lift).
    """

    schema_version: int = Field(
        4,
        description=(
            "Format version of this config file. Leave at 4 — tools update "
            "this automatically when the file format changes."
        ),
    )
    mode: ScanRequestMode = Field(
        description=(
            "What kind of scan: 'step' sweeps one or more axes, 'noscan' "
            "collects shots without moving anything."
        )
    )
    axes: list[ScanAxis] = Field(
        default_factory=list,
        description=(
            "For step scans: what to sweep. One entry is a simple 1-D scan; "
            "several entries form a grid visiting every combination, with "
            "the first axis as the outermost (slowest) loop and the last as "
            "the innermost (fastest). Leave empty for noscan."
        ),
    )
    capture: CaptureSettings = Field(
        default_factory=CaptureSettings,
        description=(
            "How shots are taken and what gets recorded: shots per step, "
            "acquisition discipline, save sets, the telemetry toggle and "
            "the trigger profile. Omit for a one-shot strict "
            "capture with no named save sets."
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

    @model_validator(mode="before")
    @classmethod
    def _lift_v1_layout(cls, data: object) -> object:
        """Lift older document layouts into the current (v3) shape.

        Applied mechanically at validation: the v1 capture fields found at
        the top level move into ``capture``, a v1 ``submission`` record is
        dropped (it left the request document — the engine's run metadata
        carries submission provenance independently), an unset removed
        field (``trigger_variant`` v3, ``native_image_save`` v4 — flat, or
        inside ``capture``) is dropped and a set one refused with its
        remedy, and a declared ``schema_version`` ≤ 3 is normalized to 4
        (a version ≥ 4 is never overwritten — a future v5 document must
        keep its stamp through this validator).  Saved
        presets, archived run-metadata documents, and stale clients
        therefore keep validating forever.  Mixing the flat fields with an
        explicit ``capture`` block is ambiguous and rejected.

        Parameters
        ----------
        data : object
            The raw input; non-mapping input passes through untouched.

        Returns
        -------
        object
            The (copied) mapping in v2 layout, or *data* unchanged.

        Raises
        ------
        ValueError
            If flat v1 capture fields and a ``capture`` block are both
            present, or if a removed field (``trigger_variant``,
            ``native_image_save``) is set.
        """
        if not isinstance(data, dict):
            return data
        if data.get("mode") == "optimize" or data.get("optimization") is not None:
            raise ValueError(
                "legacy optimization requests are retired; use the scanner's Optimize "
                "mode or submit an optimize Preset with OptimizerConfig v1"
            )
        if "optimization" in data:
            data = {key: value for key, value in data.items() if key != "optimization"}
        flat = [key for key in _V1_CAPTURE_FIELDS if key in data]
        stale_version = stale_schema_version(data, 4)
        capture = data.get("capture")
        removed_flat = [key for key in _REMOVED_FIELDS if key in data]
        removed_in_capture = (
            [key for key in _REMOVED_FIELDS if key in capture]
            if isinstance(capture, dict)
            else []
        )
        if (
            not flat
            and "submission" not in data
            and not stale_version
            and not removed_flat
            and not removed_in_capture
        ):
            return data
        lifted = dict(data)
        lifted.pop("submission", None)
        # The removed fields first, on their own terms: an unset one is
        # dropped wherever it sits; a set one gets its remedy — before the
        # flat-vs-capture check, which is about the five live fields.
        for key in removed_flat:
            if lifted.pop(key) is not None:
                raise ValueError(_REMOVED_FIELDS[key])
        if removed_in_capture:
            lifted["capture"] = dict(capture)
            for key in removed_in_capture:
                if lifted["capture"].pop(key) is not None:
                    raise ValueError(_REMOVED_FIELDS[key])
        if flat and "capture" in data:
            raise ValueError(
                f"Give capture settings either flat (v1: {flat}) or inside "
                "'capture' (v2+), not both."
            )
        if flat:
            lifted["capture"] = {key: lifted.pop(key) for key in flat}
        if stale_version:
            lifted["schema_version"] = 4
        return lifted

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
                f"'axes' only applies to 'step' scans, not {self.mode.value!r}."
            )
        return self

    def grid_shape(self) -> tuple[int, ...]:
        """Return how many positions each axis visits, outermost first.

        Returns
        -------
        tuple of int
            One count per axis, in list order (empty for noscan).
        """
        # n_positions, never len(to_values()): the shape must be computable
        # without materializing a possibly-huge range (size guards call
        # through here).
        return tuple(axis.positions.n_positions() for axis in self.axes)

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

    def planned_shots(self) -> int:
        """Return the finite shot budget without materializing axis positions.

        Both step and noscan requests use ``n_steps() × shots_per_step``;
        noscan has one motionless bin.
        """
        return self.n_steps() * int(self.capture.shots_per_step)
