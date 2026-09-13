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
