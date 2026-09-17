"""The service's answers, as Pydantic models.

Every route returns one of these; every field the page reads is named
here once.  ``state`` fields carry the kit's status words
(``geecs_web_theme.STATES``) so a chip in the page is a direct render of
the answer, never a second mapping.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class StatusOut(BaseModel):
    """One poll of the manager, plus the readiness verdict over it."""

    connected: bool
    re_state: Optional[str] = None
    manager_state: Optional[str] = None
    worker_exists: bool = False
    worker_environment_state: Optional[str] = None
    items_in_queue: int = 0
    running_item_uid: Optional[str] = None
    detail: str = ""
    readiness: str = Field(description="The readiness verdict's state word")
    readiness_detail: str = ""
    experiment: str
    identity: str = Field(description="What this process submits as")


class QueueRow(BaseModel):
    """One queue or history item, summarized for a table row."""

    state: str = Field(description="A kit status word")
    word: str = Field(description="The word the chip shows (stopped, paused…)")
    plan: str
    summary: str
    user: str = ""
    detail: str = ""
    item_uid: Optional[str] = None
    position: Optional[int] = None
    scan_numbers: list[int] = Field(default_factory=list)
    run_uids: list[str] = Field(
        default_factory=list, description="The runs' uids — the portal's run pages"
    )
    planned_shots: Optional[int] = None


class QueueOut(BaseModel):
    """The running item, what waits behind it, and what finished."""

    running: Optional[QueueRow] = None
    waiting: list[QueueRow] = Field(default_factory=list)
    finished: list[QueueRow] = Field(default_factory=list)
    summary: str


class ConfigListOut(BaseModel):
    """The names of one config kind in the experiment's tree."""

    kind: str
    names: list[str]
    experiment: str
    unavailable: dict[str, str] = Field(default_factory=dict)


class ScanVariableOut(BaseModel):
    """One catalog entry as the variable picker lists it."""

    name: str
    kind: str
    target: Optional[str] = None
    scannable: bool = True
    reason: Optional[str] = Field(
        default=None, description="Why it is not scannable today, when it is not"
    )


class PreflightQuestionOut(BaseModel):
    """One question the operator answers before the scan is queued."""

    check: str
    title: str
    message: str
    continue_label: str = "Continue"
    abort_label: str = "Abort"


class PreflightOutcomeOut(BaseModel):
    """One check already decided (``passed`` / ``skipped``)."""

    check: str
    result: str
    detail: str = ""


class PlanCallOut(BaseModel):
    """The expanded plan call — what the queue would receive."""

    name: str
    args: list[Any]
    kwargs: dict[str, Any]
    references: list[str]


class PreflightOut(BaseModel):
    """What ``POST /api/preflight`` answers: refusal, or questions + outcomes."""

    refusal: Optional[str] = None
    questions: list[PreflightQuestionOut] = Field(default_factory=list)
    outcomes: list[PreflightOutcomeOut] = Field(default_factory=list)
    plan: Optional[PlanCallOut] = None
    summary: str = ""
    planned_shots: Optional[int] = None


class SubmitIn(BaseModel):
    """The body of ``POST /api/submit``."""

    preset: dict[str, Any] = Field(description="A geecs_schemas.Preset document")
    acknowledged: list[str] = Field(
        default_factory=list,
        description="The `check` names of the preflight questions the operator ticked",
    )
    operator: Optional[str] = Field(default=None, description="Who pressed Start")
    clear_pending: bool = Field(
        default=False,
        description="Remove a failed item sitting at the front of the queue first",
    )


class SubmitOut(BaseModel):
    """A queued submission."""

    item_uid: Optional[str] = None
    message: str = ""
    submitted_as: str
    planned_shots: Optional[int] = None
    summary: str = ""


class VerbIn(BaseModel):
    """The body of the stop/pause/resume verbs."""

    force: bool = Field(
        default=False, description="Act on another operator's item (recorded)"
    )
    operator: Optional[str] = None


class VerbOut(BaseModel):
    """What a verb answered."""

    ok: bool
    message: str = ""


class ProgressOut(BaseModel):
    """The latest-run picture from the document and console streams."""

    available: bool
    detail: str = ""
    scan_number: Optional[int] = None
    plan_name: Optional[str] = None
    planned_total: Optional[int] = None
    shots_done: int = 0
    state: Optional[str] = Field(
        default=None, description="running / paused / done / aborted from the documents"
    )
    exit_status: Optional[str] = None
    paused_reason: Optional[str] = None
    updated_at: Optional[float] = None
    scan_folder: Optional[str] = Field(
        default=None, description="The run's folder as the start document names it"
    )
    day: Optional[str] = Field(
        default=None,
        description="The run's day, ISO (from the start document's scan_tag)",
    )


class ConsoleLine(BaseModel):
    """One line of the manager's console-output stream."""

    seq: int
    text: str
    at: float


class HealthOut(BaseModel):
    """Liveness + the manager probe + version (the fleet-map health check)."""

    ok: bool
    version: str
    manager: bool
    readiness: str
    experiment: str


class MoveIn(BaseModel):
    """The body of ``POST /api/move``: one manual move as an ``mv`` queue item."""

    variable: str = Field(
        description="A scan-variable catalog name, a Device:Variable, or a device"
    )
    value: float
    operator: Optional[str] = None


class ItemOut(BaseModel):
    """A queued non-scan item: a move, an action, a calibration."""

    item_uid: Optional[str] = None
    message: str = ""
    submitted_as: str
    plan: str
    summary: str = ""
    reference: Optional[str] = Field(
        default=None, description="For a move: the device reference the item carries"
    )


class ActionOut(BaseModel):
    """One action plan as the picklist shows it."""

    name: str
    description: str = ""
    steps: int = Field(description="Concrete steps after nested runs are inlined")
    nested: list[str] = Field(
        default_factory=list, description="Plans this one runs by name"
    )
    problem: Optional[str] = Field(
        default=None, description="Why it cannot run (unknown nested plan, a loop)"
    )


class ActionStepOut(BaseModel):
    """One flattened step of an action plan."""

    do: str
    device: Optional[str] = None
    variable: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    seconds: Optional[float] = None
    wait: Optional[bool] = None
    from_plan: Optional[str] = Field(
        default=None, description="The nested plan this step was inlined from"
    )
    text: str


class ActionDetailOut(BaseModel):
    """``GET /api/actions/{name}``: the preview — what running it would do, in order."""

    name: str
    description: str = ""
    steps: list[ActionStepOut]
    writes: int = Field(description="How many steps write to hardware")


class CalibrationDeviceOut(BaseModel):
    """One device's stored drain offset."""

    name: str
    offset_s: float
    scatter_s: Optional[float] = None
    shots: Optional[int] = None
    geecs_device: Optional[str] = None


class CalibrationOut(BaseModel):
    """``GET /api/calibration``: the stored shot offsets, summarized."""

    stored: bool
    path: str = ""
    detail: str = ""
    reference: Optional[str] = None
    measured_at: Optional[str] = None
    trigger_profile: Optional[str] = None
    trigger_rate_hz: Optional[float] = None
    description: str = ""
    devices: list[CalibrationDeviceOut] = Field(default_factory=list)
    max_offset_s: Optional[float] = None
    max_offset_device: Optional[str] = None


class CalibrationIn(BaseModel):
    """The body of the two calibration verbs."""

    devices: list[str] = Field(
        description="The triggered devices to check or measure (at least two)"
    )
    trigger_profile: Optional[str] = None
    tolerance_s: Optional[float] = Field(
        default=None, description="check only: widest accepted disagreement"
    )
    shots: Optional[int] = Field(default=None, description="measure only")
    write: bool = Field(
        default=False, description="measure only: store the result in the configs tree"
    )
    operator: Optional[str] = None


class SavePresetIn(BaseModel):
    """The body of ``POST /api/configs/presets/{name}``."""

    preset: dict[str, Any] = Field(description="A geecs_schemas.Preset document")
    overwrite: bool = False


class SavePresetOut(BaseModel):
    """A written preset."""

    name: str
    path: str
    message: str


class ScanLogOut(BaseModel):
    """A chunk of the running (or last) scan's ``scan.log``."""

    available: bool
    folder: Optional[str] = None
    offset: int = 0
    lines: list[str] = Field(default_factory=list)
    more: bool = Field(
        default=False, description="The file holds more beyond this chunk"
    )
    detail: str = ""
    scan_number: Optional[int] = None


class SettableOut(BaseModel):
    """One numeric settable as the movable panel lists it."""

    name: str = Field(
        description="The canonical Device:Variable — what a request stores"
    )
    device: str
    variable: str
    alias: str = Field(
        default="", description="The DB's curated short name; empty when none"
    )
    units: str = ""
    min: Optional[float] = None
    max: Optional[float] = None


class SettablesOut(BaseModel):
    """Every numeric settable of the experiment, aliased ones first."""

    items: list[SettableOut] = Field(default_factory=list)
    source: str = Field(description="db or demo")
    detail: str = Field(default="", description="Why the list is empty, when it is")


class ReadbackOut(BaseModel):
    """One reading of a device variable over the gateway."""

    variable: str
    pv: str = ""
    ok: bool = False
    value: Optional[float] = None
    units: str = ""
    timestamp: Optional[float] = Field(
        default=None, description="The reading's own stamp, epoch seconds"
    )
    age_s: Optional[float] = Field(
        default=None, description="Now minus the reading's stamp"
    )
    detail: str = ""


class OptimizationOut(BaseModel):
    """Latest adaptive iteration; missing or nonfinite values are null."""

    run_uid: Optional[str] = None
    config: Optional[str] = None
    iteration: int = 0
    max_iterations: Optional[int] = None
    proposal: dict[str, Optional[float]] = Field(default_factory=dict)
    measured: dict[str, Optional[float]] = Field(default_factory=dict)
    outputs: dict[str, Optional[float]] = Field(default_factory=dict)
    valid_shots: dict[str, Optional[float]] = Field(default_factory=dict)
    best: dict[str, Optional[float]] = Field(default_factory=dict)
    best_moves: dict[str, Optional[float]] = Field(default_factory=dict)
    objectives: list[str] = Field(default_factory=list)
    finished: bool = False
    scan_number: Optional[int] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    expires_at: Optional[float] = None
    exit_status: Optional[str] = None
    invalidated_reason: Optional[str] = None
    expired: bool = False


class SetBestIn(BaseModel):
    """Identify the observed run whose best physical settings should be applied."""

    run_uid: str
    operator: Optional[str] = None


class OptimizerConfigOut(BaseModel):
    """Config document, required device names and run defaults for the form."""

    name: str
    document: dict[str, Any]  # The registered OptimizerConfig's JSON representation.
    required_devices: list[str]
    shots_per_step: int
    max_iterations: Optional[int] = None
