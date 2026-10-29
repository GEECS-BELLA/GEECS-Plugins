"""The experiment's home, the Control Room: the top strip, the two device lists, the launcher.

The lists sort the module's words; they judge nothing. **Not live** holds
every device whose state is not ``live``; **In alarm** every device whose
state, or any of whose roles' verdicts, is ``error``: the module's word for
an EPICS severity of major or invalid (except a never-processed record,
``unknown``, and a silent device, ``not answering``, both "Not live"), and
for a device with a tripped fault bit. A row's cause is the module's own
evidence: the heartbeat's word and the roles that are not ``live``, by word.
A device whose detail the module did not give is ``unread``: its roles'
alarms are unknown, so the alarm list is "none" only when ``unread`` is
empty (a ``live`` state says nothing about an alarmed setpoint). The strip reads the scanner's
API, its run word the manager's (``re_state``, as the scanner's own page
has it); "unreachable" there is a state, not an error page.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import yaml
from fastapi import APIRouter, Request
from geecs_schemas import (
    HOME_LAYOUT_FILENAME,
    PANEL_DEVICES_FILENAME,
    SCREENS_DIRNAME,
    HomeLayout,
    PanelDevices,
    load_families,
)
from pydantic import ValidationError

from geecs_control_room.module_client import ModuleClient, ModuleUnavailable
from geecs_control_room.screens import KIT_STATE
from geecs_control_room.settings import Settings

#: The cause's words, most telling first; ``live`` is never a cause.
_WORDS = ("error", "not answering", "unreachable", "stale", "off", "unknown")
#: Role names shown per word before "+N".
_NAMED = 3
_LINK_TITLES = {"scanner": "Scanner", "portal": "Data Portal", "logbook": "Logbook", "config_editor": "Config editor"}  # fmt: skip
#: Each of the strip's two scanner reads, made at once: inside the page's 3 s timeout,
#: yet longer than one of the queue client's 2 s manager round trips.
SCANNER_TIMEOUT_S = 2.5
#: The manager's words as the strip's kit states (others: unknown).
_RUN_STATE = {
    "running": "running",
    "paused": "paused",
    "pausing": "paused",
    "idle": "ok",
}


def _role(entry: dict[str, Any]) -> str:
    name = (
        entry.get("label") or entry["role"]
    )  # the family's label reads better than the role
    return f"{name}[{entry['axis']}]" if entry.get("axis") else name


def cause(state: str, detail: Optional[dict[str, Any]], only: tuple[str, ...] = _WORDS) -> str:  # fmt: skip
    """The module's evidence for ``state``: the heartbeat's word, then the roles not live, by word."""
    if detail is None:
        return f"{state}; the module gave no detail for it"
    parts = []
    if detail.get("heartbeat") not in (None, "live") and detail["heartbeat"] in only:
        parts.append(f"heartbeat {detail['heartbeat']}")
    by_word: dict[str, list[str]] = {}
    for entry in detail.get("roles", []):
        by_word.setdefault(entry.get("verdict") or "", []).append(_role(entry))
    for word in only:
        if names := by_word.get(word):
            more = f" +{len(names) - _NAMED}" if len(names) > _NAMED else ""
            parts.append(f"{word}: {', '.join(names[:_NAMED])}{more}")
    if parts:
        return "; ".join(parts)
    # a tripped fault bit makes a device error with every role live; the route does not name it yet
    fault = " (a fault bit, perhaps: the device route does not name one yet)" if state == "error" else ""  # fmt: skip
    return f"the module's state is {state}; it names no role{fault}"


def sort_devices(
    roster: list[dict[str, Any]],
    details: dict[str, Optional[dict[str, Any]]],
    journal: list[dict[str, Any]],
    iocs: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Sort the roster into ``not_live`` and ``in_alarm`` (a device may sit in both); ``unread`` names the devices without detail."""
    last = {row["device"]: row for row in journal}  # oldest first: the newest wins
    lists: dict[str, list[dict[str, Any]]] = {"not_live": [], "in_alarm": []}
    # no detail: the roles' verdicts are unknown, not clear; the page must not read "none"
    lists["unread"] = [
        {"name": d["name"]} for d in roster if details.get(d["name"]) is None
    ]
    for device in roster:
        name, state = device["name"], device.get("state") or "unknown"
        detail = details.get(name)
        roles = (detail or {}).get("roles", [])
        row = {"name": name, "family": device.get("family"), "state": state}
        row |= {"kit": KIT_STATE.get(state, "unknown"), "ioc": iocs.get(name)}
        row["last"] = last.get(name)
        if state != "live":
            lists["not_live"].append(row | {"cause": cause(state, detail)})
        if state == "error" or any(e.get("verdict") == "error" for e in roles):
            lists["in_alarm"].append(row | {"cause": cause(state, detail, ("error",))})
    return lists


def _iocs(settings: Settings) -> dict[str, str]:
    """Each device's serving IOC, from ``panel_devices.yaml`` (none: no IOC shown)."""
    path = settings.experiment_dir / PANEL_DEVICES_FILENAME
    try:
        panel = PanelDevices.from_path(path) if path.is_file() else PanelDevices()
    except (ValidationError, yaml.YAMLError, OSError):
        return {}
    return {name: d.ioc for name, d in panel.devices.items() if d.ioc}


async def device_lists(settings: Settings, module: ModuleClient) -> dict[str, Any]:
    """The roster sorted into the two lists, or why it could not be read (never an empty list then)."""
    try:
        status, body = await module.devices()
    except ModuleUnavailable as exc:
        return {"ok": False, "detail": f"the module did not answer: {exc}"}
    if status != 200:
        return {"ok": False, "detail": f"the module answered {status}: {body.get('detail', body)}"}  # fmt: skip
    roster = body.get("devices", [])

    async def detail(device: dict[str, Any]) -> Optional[dict[str, Any]]:
        # a live DB device's roles carry no module verdict: its detail says nothing the roster does not
        if device.get("source") == "gateway" and device.get("state") == "live":
            return {"roles": []}
        try:
            got, answer = await module.device(device["name"])
        except ModuleUnavailable:
            return None
        return answer if got == 200 else None

    async def rows() -> Optional[list[dict[str, Any]]]:
        try:
            got, journal = await module.journal(200)
        except ModuleUnavailable:
            return None
        return journal if got == 200 else None

    # the details and the journal at once: the whole read is two module reads long
    *answers, journal = await asyncio.gather(*(detail(d) for d in roster), rows())  # fmt: skip
    lists = sort_devices(roster, dict(zip([d["name"] for d in roster], answers)), journal or [], _iocs(settings))  # fmt: skip
    devices = [{"name": d["name"], "family": d.get("family")} for d in roster]
    return {"ok": True, "devices": devices, "journal": journal is not None} | lists


async def strip(scanner: Optional[httpx.AsyncClient]) -> dict[str, Any]:
    """The run state and Scan#/Shot# from the scanner's API; unreachable is a state."""
    if scanner is None:
        return {"scanner": "not configured", "run": "unknown", "kit": "unknown"}
    try:
        answers = await asyncio.gather(scanner.get("/api/status"), scanner.get("/api/progress"))  # fmt: skip
        status, progress = (r.raise_for_status().json() for r in answers)
    except (httpx.HTTPError, ValueError):
        return {"scanner": "unreachable", "run": "unknown", "kit": "unknown"}
    if not status.get("connected"):
        run = "queue server unreachable"
    else:
        run = (
            status.get("re_state") or "unknown"
        )  # the manager's word; progress lags a pause
    keep = ("scan_number", "shots_done", "planned_total", "state")
    return {"scanner": "ok", "run": run, "kit": _RUN_STATE.get(run, "unknown")} | {
        k: progress.get(k) for k in keep
    }


@dataclass
class Entry:
    """One launcher entry: where it opens, or why it cannot (``href`` None)."""

    title: str
    href: Optional[str] = None
    reason: str = ""
    families: list[str] = field(default_factory=list)


@dataclass
class Launcher:
    """The launcher as ``home.yaml`` makes it, or why it could not be read."""

    areas: list[tuple[str, list[Entry]]] = field(default_factory=list)
    device_pages: list[Entry] = field(default_factory=list)
    links: list[Entry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    state: str = "ok"  # ok | empty (no file) | error (it does not validate)


def launcher(settings: Settings) -> Launcher:
    """Read ``<experiment>/home.yaml``; a missing screen, family or link setting is a disabled entry."""
    folder = settings.experiment_dir
    path = folder / HOME_LAYOUT_FILENAME
    if not path.is_file():
        return Launcher(problems=[f"no {HOME_LAYOUT_FILENAME} in {settings.experiment}"], state="empty")  # fmt: skip
    try:
        layout = HomeLayout.from_path(path)
    except (ValidationError, yaml.YAMLError, OSError) as exc:
        return Launcher(problems=[str(exc)], state="error")

    def screen(name: str) -> Entry:
        if (folder / SCREENS_DIRNAME / f"{name}.yaml").is_file():
            return Entry(name, f"/screens/{name}")
        return Entry(name, reason=f"no {SCREENS_DIRNAME}/{name}.yaml")

    installed = load_families()
    pages = []
    for page in layout.device_pages:
        unknown = [f for f in page.families if f not in installed]
        why = f"no family {', '.join(unknown)} installed" if unknown else ""
        pages.append(Entry(page.title, None if unknown else "", why, page.families))
    links = [
        Entry(
            _LINK_TITLES.get(word, word),
            url,
            "" if url else f"not configured: GEECS_CONTROL_ROOM_{word.upper()}_URL",
        )  # fmt: skip
        for word in layout.links
        for url in [settings.links.get(word)]
    ]
    areas = [(a.title, [screen(n) for n in a.screens]) for a in layout.areas]
    return Launcher(areas, pages, links)


def router(settings: Settings, scanner: Optional[httpx.AsyncClient]) -> APIRouter:
    """The home page and the two JSON reads it polls."""
    r = APIRouter()

    @r.get("/")
    async def home(request: Request) -> Any:
        context = {"launcher": launcher(settings)}
        return request.app.state.templates.TemplateResponse(request, "home.html", context)  # fmt: skip

    @r.get("/api/home/devices")
    async def devices(request: Request) -> dict[str, Any]:
        return await device_lists(settings, request.app.state.module)

    @r.get("/api/home/strip")
    async def run_strip() -> dict[str, Any]:
        return await strip(scanner)

    return r
