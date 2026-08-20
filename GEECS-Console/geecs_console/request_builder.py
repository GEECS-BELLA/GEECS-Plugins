"""Build a validated :class:`geecs_schemas.ScanRequest` from console form state.

This is the core of the console: :class:`ConsoleFormState` mirrors the scan
form (region R3 of the screen map) as a small Pydantic model, and
:func:`build_scan_request` is the one pure function that turns it into the
schema's submission object.  Everything the GUI submits goes through here, so
the mapping (mode radios → ``ScanRequestMode`` + ``background`` flag,
start/stop/step → ``PositionRange``, the ``MAXIMUM_SCAN_SIZE`` guard) is
testable without a single widget.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from geecs_schemas import (
    AcquisitionMode,
    OptimizationSpec,
    PositionList,
    PositionRange,
    Positions,
    ScanAxis,
    ScanRequest,
    ScanRequestMode,
)

MAXIMUM_SCAN_SIZE = 1_000_000
"""Runaway-scan guard: total shots above this refuse to build a request."""


class ConsoleFormError(ValueError):
    """The form state cannot become a valid scan request."""


class ConsoleMode(str, Enum):
    """The scan-mode radio buttons of region R3.

    Attributes
    ----------
    NOSCAN : str
        Collect shots without moving anything.
    ONE_D : str
        Sweep one variable through start/stop/step (or explicit positions).
    GRID : str
        Sweep two or more variables as an outer-product grid.
    OPTIMIZATION : str
        Let an optimizer drive the settings, per a named optimizer config
        loaded into the form's ``optimization`` spec.
    BACKGROUND : str
        A noscan whose data is marked as background/calibration shots.
    """

    NOSCAN = "noscan"
    ONE_D = "1d"
    GRID = "grid"
    OPTIMIZATION = "optimization"
    BACKGROUND = "background"


class FormAxis(BaseModel):
    """One axis row of the scan form: a variable and its positions.

    Exactly one positions shape must be given: ``start``/``stop``/``step``
    together, or an explicit ``values`` list.
    """

    variable: str = Field(min_length=1)
    start: Optional[float] = None
    stop: Optional[float] = None
    step: Optional[float] = None
    values: Optional[list[float]] = None

    @model_validator(mode="after")
    def _one_positions_shape(self) -> "FormAxis":
        """Require start/stop/step together, or values, never both.

        Returns
        -------
        FormAxis
            The validated model.

        Raises
        ------
        ValueError
            If neither or both position shapes are given, or the range is
            partially filled.
        """
        range_fields = (self.start, self.stop, self.step)
        has_range = any(value is not None for value in range_fields)
        full_range = all(value is not None for value in range_fields)
        if self.values is not None:
            if has_range:
                raise ValueError(
                    f"Axis {self.variable!r}: give start/stop/step or an "
                    "explicit values list, not both."
                )
            return self
        if not full_range:
            raise ValueError(
                f"Axis {self.variable!r} needs start, stop, and step "
                "(or an explicit values list)."
            )
        return self

    def to_positions(self) -> Positions:
        """Return the schema positions object for this axis.

        Returns
        -------
        PositionRange or PositionList
            The schema-validated positions (a zero step raises here, via
            the schema's own validator).
        """
        if self.values is not None:
            return PositionList(values=self.values)
        try:
            return PositionRange(start=self.start, end=self.stop, step=self.step)
        except ValueError as exc:
            raise ConsoleFormError(f"Axis {self.variable!r}: {exc}") from exc


class ConsoleFormState(BaseModel):
    """Snapshot of the scan form — everything the operator chose on screen.

    ``optimization`` carries the *loaded* spec of the optimizer config the
    R3 combo names (the window resolves name → spec when snapshotting), so
    this model — and :func:`build_scan_request` — stays pure: no file I/O.

    ``max_iterations`` mirrors the R3 Iterations spinner (optimization mode
    only) and **owns** the submitted spec's ``max_iterations``: the builder
    writes it onto the spec, so ``0``/``None`` (the spinner's "auto")
    submits no limit even when the loaded config carries one — the window
    seeds the spinner from the config on selection, so a config's own limit
    is surfaced there rather than applied silently.
    """

    mode: ConsoleMode = ConsoleMode.ONE_D
    axes: list[FormAxis] = Field(default_factory=list)
    shots_per_step: int = Field(1, ge=1)
    save_sets: list[str] = Field(default_factory=list)
    trigger_profile: Optional[str] = None
    trigger_variant: Optional[str] = None
    acquisition: AcquisitionMode = AcquisitionMode.FREE_RUN
    description: str = ""
    background: bool = False
    optimization: Optional[OptimizationSpec] = None
    max_iterations: Optional[int] = Field(None, ge=0)


def _position_count(positions: Positions) -> int:
    """Count the positions without materializing them (guard-safe).

    Parameters
    ----------
    positions : PositionRange or PositionList
        The schema positions object.

    Returns
    -------
    int
        How many positions the axis visits (matches
        ``PositionRange.to_values``'s reference derivation).
    """
    if isinstance(positions, PositionList):
        return len(positions.values)
    span = positions.end - positions.start
    return int(abs(span) / abs(positions.step) + 1e-9) + 1


def estimate_total_shots(form: ConsoleFormState) -> int:
    """Compute the total shot count the form implies (the live R3 label).

    Parameters
    ----------
    form : ConsoleFormState
        The current form state; axes may be present for step modes.

    Returns
    -------
    int
        ``shots_per_step`` times the product of the axis position counts
        (1 for noscan/background).  Optimization mode multiplies by
        ``max_iterations`` when set — the engine's announced upper bound —
        and falls back to one iteration's worth when unset ("auto": the
        window renders that case as such rather than as a count).

    Raises
    ------
    ConsoleFormError
        If an axis's positions are invalid (e.g. zero step).
    """
    if form.mode is ConsoleMode.OPTIMIZATION and form.max_iterations:
        return form.shots_per_step * form.max_iterations
    if form.mode not in (ConsoleMode.ONE_D, ConsoleMode.GRID):
        return form.shots_per_step
    total = form.shots_per_step
    for axis in form.axes:
        total *= _position_count(axis.to_positions())
    return total


def build_scan_request(form: ConsoleFormState) -> ScanRequest:
    """Turn the console form state into a validated :class:`ScanRequest`.

    Parameters
    ----------
    form : ConsoleFormState
        The form snapshot to translate.

    Returns
    -------
    ScanRequest
        The schema-validated submission object.

    Raises
    ------
    ConsoleFormError
        On optimization mode without a loaded optimization spec, a
        missing/extra axis for the mode, or a total shot count above
        :data:`MAXIMUM_SCAN_SIZE`.
    pydantic.ValidationError
        When the schema itself rejects the assembled request (e.g. a
        trigger variant without a profile, duplicate axis variables).
    """
    # A stray spec on a non-optimize form is a bug the schema's consistency
    # validator surfaces loudly rather than being silently dropped here.
    optimization = form.optimization
    if form.mode in (ConsoleMode.ONE_D, ConsoleMode.GRID):
        if form.mode is ConsoleMode.ONE_D and len(form.axes) != 1:
            raise ConsoleFormError("A 1D scan needs exactly one axis.")
        if form.mode is ConsoleMode.GRID and len(form.axes) < 2:
            raise ConsoleFormError("A grid scan needs at least two axes.")
        mode = ScanRequestMode.STEP
        axes = [
            ScanAxis(variable=axis.variable, positions=axis.to_positions())
            for axis in form.axes
        ]
    elif form.mode is ConsoleMode.OPTIMIZATION:
        if form.optimization is None:
            raise ConsoleFormError(
                "An optimization scan needs an optimization config — pick "
                "one in the scan form."
            )
        mode = ScanRequestMode.OPTIMIZE
        axes = []
        # The form's iteration count owns the submitted spec's limit (see
        # ConsoleFormState): the spinner was seeded from the config, so
        # whatever it now says — a number or "auto" (None) — is the
        # operator's answer, not a value to merge with the config's.
        optimization = form.optimization.model_copy(
            update={"max_iterations": form.max_iterations or None}
        )
    else:
        mode = ScanRequestMode.NOSCAN
        axes = []

    total_shots = estimate_total_shots(form)
    if total_shots > MAXIMUM_SCAN_SIZE:
        raise ConsoleFormError(
            f"This scan would take {total_shots} shots — above the "
            f"{MAXIMUM_SCAN_SIZE} runaway-scan limit. Reduce the range or "
            "shots per step."
        )

    return ScanRequest(
        mode=mode,
        axes=axes,
        shots_per_step=form.shots_per_step,
        acquisition=form.acquisition,
        save_sets=list(form.save_sets),
        trigger_profile=form.trigger_profile,
        trigger_variant=form.trigger_variant,
        description=form.description,
        background=form.background or form.mode is ConsoleMode.BACKGROUND,
        optimization=optimization,
    )


def form_state_from_request(request: ScanRequest) -> ConsoleFormState:
    """Invert :func:`build_scan_request`: a saved request back into form state.

    This is the Apply half of presets (a preset IS a saved
    :class:`~geecs_schemas.ScanRequest`).  It is pure — no widgets — and
    strict: content the scan form cannot express raises instead of being
    silently dropped, so applying then re-saving a preset never loses part
    of it.

    Parameters
    ----------
    request : ScanRequest
        The saved request to translate.

    Returns
    -------
    ConsoleFormState
        Form state that :func:`build_scan_request` maps back onto an
        equivalent request.  A ``noscan`` with the ``background`` flag
        becomes :attr:`ConsoleMode.BACKGROUND` (the same folding the
        builder applies in the other direction); an ``optimize`` request
        becomes :attr:`ConsoleMode.OPTIMIZATION` carrying the request's
        optimization spec and its ``max_iterations`` (the Iterations
        spinner).

    Raises
    ------
    ConsoleFormError
        For a request carrying action bindings — inexpressible on the form.
    """
    if request.actions.setup or request.actions.per_step or request.actions.closeout:
        raise ConsoleFormError(
            "This preset carries action bindings (setup/per_step/closeout), "
            "which the console form cannot express — applying it would drop "
            "them on the next save or submit."
        )

    if request.mode is ScanRequestMode.NOSCAN:
        mode = ConsoleMode.BACKGROUND if request.background else ConsoleMode.NOSCAN
        background = False  # folded into the mode, mirroring the builder
        axes: list[FormAxis] = []
    elif request.mode is ScanRequestMode.OPTIMIZE:
        mode = ConsoleMode.OPTIMIZATION
        background = request.background
        axes = []
    else:
        mode = ConsoleMode.ONE_D if len(request.axes) == 1 else ConsoleMode.GRID
        background = request.background
        axes = []
        for axis in request.axes:
            if isinstance(axis.positions, PositionList):
                axes.append(
                    FormAxis(variable=axis.variable, values=axis.positions.to_values())
                )
            else:
                axes.append(
                    FormAxis(
                        variable=axis.variable,
                        start=axis.positions.start,
                        stop=axis.positions.end,
                        step=axis.positions.step,
                    )
                )

    return ConsoleFormState(
        mode=mode,
        axes=axes,
        shots_per_step=request.shots_per_step,
        save_sets=list(request.save_sets),
        trigger_profile=request.trigger_profile,
        trigger_variant=request.trigger_variant,
        acquisition=request.acquisition,
        description=request.description,
        background=background,
        optimization=request.optimization,
        # The spec's limit is the Iterations spinner's value (None = "auto"
        # = 0 on the widget) — the exact inverse of the builder's override.
        max_iterations=(
            request.optimization.max_iterations
            if request.optimization is not None
            else None
        ),
    )
