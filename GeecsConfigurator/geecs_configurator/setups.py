"""Setups on the surface: capture from live readbacks, the diff against them, the apply through the scanner.

All three are the scanner's, reached over HTTP (``apply.ScannerClient``);
this module adds no rule of its own:

- **Capture reads, never writes.** It asks the experiment's scanner for one
  readback per ``Device:Variable`` (``GET /api/readback``, the path the
  scanner's own diff reads; a variable that is not a number is read again
  as a word) and answers a draft ``Setup``: one ``set`` step per value read,
  ``captured`` naming the operator, the time and the readbacks. Nothing is
  written, to a device or to the configs repo: the draft is saved through
  the one save route (``PUT /api/documents/setup/<name>.yaml`` with
  ``create``), which makes the operator the author, judges the tier and
  refuses a path this box does not own.
- **The diff and the apply are relayed**, status and body as the scanner
  answered them: ``GET /api/setups/{name}/diff`` and
  ``POST /api/setups/{name}/apply``. The apply's ``acknowledged`` goes as
  the page sent it; the scanner asks its questions (the setup's writes,
  each hazard confirm, every check bound to the setup's digest) and this
  service never answers one. An experiment this box does not own is never
  reached (403), as no save of it is made here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from geecs_schemas import Setup
from pydantic import BaseModel, Field, ValidationError

from geecs_configurator.apply import ScannerClient

router = APIRouter(prefix="/api")

#: The most variables one capture reads (each is one readback through the scanner).
MAX_CAPTURE = 200


class CaptureIn(BaseModel):
    """What to capture: the new setup's name, its description, who captures it, and the variables to read."""

    name: str
    description: str = ""
    operator: str
    variables: list[str] = Field(
        description="Device:Variable names, in the order the steps take"
    )


class Unread(BaseModel):
    """A variable the capture could not read, and why."""

    variable: str
    detail: str


class Captured(BaseModel):
    """A draft setup from live readbacks: not saved, never written to a device."""

    id: str = Field(description="The document id a save creates: <name>.yaml")
    document: dict[str, Any]  # Any: the setup as served (Setup.model_dump(mode="json"))
    unread: list[Unread] = []


def _scanner(request: Request, experiment: str | None) -> tuple[str, ScannerClient]:
    """The experiment's scanner (the served one by default); 503 when none is configured."""
    exp = experiment or request.app.state.settings.experiment
    scanner = request.app.state.consumers.scanners.get(exp)
    if scanner is None:
        raise HTTPException(503, f"no scanner configured for {exp or 'this experiment'} (GEECS_CONFIGURATOR_SCANNER_URL)")  # fmt: skip
    return exp, scanner


def _owned(request: Request, experiment: str) -> None:
    """403 for an experiment another services box owns (no check with the commit path off: nothing is owned)."""
    commit = request.app.state.commit
    if commit is not None and experiment not in commit.ownership.experiments:
        raise HTTPException(403, f"{experiment} is not this services box's: its setups are applied from the box that owns it")  # fmt: skip


def _read(
    scanner: ScannerClient, variable: str
) -> tuple[Any, str]:  # Any: a number or a word
    """One readback through the scanner: ``(value, "")``, or ``(None, why not)``."""
    try:
        status, body = scanner.readback(variable)
        if status == 200 and not body.get("ok") and "not a number" in (body.get("detail") or ""):  # fmt: skip
            status, body = scanner.readback(variable, word=True)
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"the scanner did not answer: {' '.join(str(exc).split())[:200]}"
    if status != 200:
        error = body.get("error") if isinstance(body, dict) else None
        return None, (error or {}).get("message") or f"the scanner answered {status}"
    if not body.get("ok"):
        return None, body.get("detail") or "no reading"
    return (body["text"] if body.get("text") is not None else body["value"]), ""


def capture(
    scanner: ScannerClient, body: CaptureIn, *, now: datetime | None = None
) -> Captured:
    """Read each variable through the scanner and draft the setup; raise ``ValueError`` when nothing can be drafted."""
    steps, readbacks, unread = [], [], []
    for name in body.variables:
        device, sep, variable = name.strip().partition(":")
        if not (sep and device and variable):
            unread.append(Unread(variable=name, detail="not a Device:Variable name"))
            continue
        value, why = _read(scanner, f"{device}:{variable}")
        if why:
            unread.append(Unread(variable=name, detail=why))
            continue
        steps.append({"do": "set", "device": device, "variable": variable, "value": value})  # fmt: skip
        readbacks.append({"device": device, "variable": variable})
    if not steps:
        raise ValueError("no readback answered: nothing to capture")
    at = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    draft = {"name": body.name, "description": body.description, "steps": steps, "captured": {"by": body.operator.strip(), "at": at, "readbacks": readbacks}}  # fmt: skip
    setup = Setup.model_validate(draft)
    return Captured(id=f"{setup.name}.yaml", document=setup.model_dump(mode="json"), unread=unread)  # fmt: skip


@router.post("/setups/capture", response_model=Captured)
def capture_route(request: Request, body: CaptureIn, experiment: str | None = None) -> Captured:  # fmt: skip
    """A draft setup from the live readbacks of ``variables`` (nothing saved; nothing written to a device)."""
    if not body.operator.strip():
        raise HTTPException(422, "the operator is required: they are the setup's capturer and its commit's author")  # fmt: skip
    if not body.variables:
        raise HTTPException(422, "choose the variables to capture")
    if len(body.variables) > MAX_CAPTURE:
        raise HTTPException(422, f"at most {MAX_CAPTURE} variables per capture")
    _, scanner = _scanner(request, experiment)
    try:
        return capture(scanner, body)
    except ValidationError as exc:
        raise HTTPException(422, [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]) from exc  # fmt: skip
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _relay(
    scanner: ScannerClient, method: str, path: str, body: Any = None
) -> JSONResponse:  # Any: the page's JSON
    try:
        status, out = scanner.relay(method, path, body)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"the scanner did not answer: {' '.join(str(exc).split())[:200]}") from exc  # fmt: skip
    return JSONResponse(out, status_code=status)


@router.get("/settables")
def settables(request: Request, experiment: str | None = None) -> JSONResponse:
    """The experiment's numeric settables, the scanner's list (the capture's picker)."""
    return _relay(_scanner(request, experiment)[1], "GET", "/api/settables")


@router.get("/setups/{name}/diff")
def setup_diff(request: Request, name: str, experiment: str | None = None) -> JSONResponse:  # fmt: skip
    """The committed setup's writes against the live readbacks: the scanner's diff, as it answered."""
    return _relay(_scanner(request, experiment)[1], "GET", f"/api/setups/{_quoted(name)}/diff")  # fmt: skip


class ApplyIn(BaseModel):
    """The operator, and the checks they acknowledged in the page's dialog (sent on as given)."""

    operator: str = ""
    acknowledged: list[str] = []


@router.post("/setups/{name}/apply")
def setup_apply(request: Request, name: str, body: ApplyIn, experiment: str | None = None) -> JSONResponse:  # fmt: skip
    """Queue the committed setup through the scanner: its questions come back (409) until the operator acknowledged each."""
    exp, scanner = _scanner(request, experiment)
    _owned(request, exp)
    return _relay(scanner, "POST", f"/api/setups/{_quoted(name)}/apply", body.model_dump())  # fmt: skip


def _quoted(name: str) -> str:
    from urllib.parse import quote

    return quote(name, safe="")
