"""Device pages: one page per kind of hardware, any device of that kind (``GeecsControlRoom/CLAUDE.md`` § "The device pages").

``/device/<kind>`` is the picker: the registry's devices of that kind
(``native_devices.yaml`` + ``panel_devices.yaml``, a device's kind from its
panel entry's family, else its native ``kind``) in a select, each with the
module's roster state (``/api/device/<kind>``, polled by the page), and a
free-text name for a device the files do not list. ``/device/<kind>/<name>``
resolves the device with one read of the module's device route and draws
the kind's page from the roles that route serves (already resolved there); a device the page cannot
draw is a kit pane that says why (not on the roster, wrong kind, not a
native device: ``denied``; the module silent: ``stale``, and the page polls
until it answers), never a blank page and never a 500. PV names, verdicts
and values come from the module and Hare in the browser.

Every page draws a rail of controls and a readings strip from a **panel**:
the family's ``panel``, else the automatic one (``auto_panel``: every
writable role in role order, every monitored readback a reading), which is
how a DB device (kind ``settables``: on the module's roster with ``source:
gateway``, its family the GEECS DB device type, no family file) and a
family not yet given a panel are complete with no template of their own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from geecs_schemas import AUTO_GROUP, FamilyFile, load_families
from geecs_web_theme.web import root_of
from pydantic import ValidationError

from geecs_control_room.module_client import GATEWAY, ModuleClient, ModuleUnavailable
from geecs_control_room.screens import KIT_STATE, GroupKey, registry, role_spec
from geecs_control_room.settings import Settings
from geecs_control_room.traces import trace_roles


@dataclass(frozen=True)
class Kind:
    """A page kind: the family ``kind`` its devices' families carry (``None``: the DB devices), and the template that draws them."""

    family_kind: Optional[str]
    template: str


#: The kind word of the GEECS DB devices the module serves through the gateway (no family file).
DB_KIND = "settables"

#: The URL's kind word → the family kind → the page. A new family of a kind needs no code.
KINDS = {
    "camera": Kind("aravis_camera", "device_camera.html"),
    "trace": Kind("array_detector", "device_trace.html"),
    "motor": Kind("epics_motor", "device.html"),
    "supply": Kind("setpoint_device", "device.html"),
    DB_KIND: Kind(None, "device.html"),
}
_WORD = {k.family_kind: word for word, k in KINDS.items() if k.family_kind}


def kind_of(family: Optional[str]) -> Optional[str]:
    """The kind word of an installed family's devices; ``None`` for no such family (a DB device type)."""
    found = load_families().get(family or "")
    return _WORD.get(found.kind) if found else None


#: The word for a registry device the module's roster does not list.
NOT_ON_ROSTER = "not on the roster"


@dataclass(frozen=True)
class Catalogue:
    """The registry's devices (``name``, ``family``, ``kind``), the files' problems, and whether the files could be read at all."""

    rows: list[dict[str, Any]]
    problem: Optional[str] = None
    unreadable: bool = False


def catalogue(settings: Settings) -> Catalogue:
    """Every registry device by name (``kind``: its page's kind word, ``None`` for no page), and the files' problems.

    The module's join rule (GeecsCanopy ``native.py``): a device both files name
    with different families is no kind's; it is left out and named in the problem.
    A family that is not installed, or whose kind no page draws, is kind ``None``.
    """
    try:
        panel, native = registry(settings.experiment_dir)
    except (ValidationError, yaml.YAMLError, OSError) as exc:
        return Catalogue([], str(exc), unreadable=True)
    families = load_families()
    rows, problems = [], []
    for name in sorted({*native.devices, *panel.devices}):
        entry, panel_entry = native.devices.get(name), panel.devices.get(name)
        template = entry.family_template if entry else panel_entry.family
        if panel_entry and entry and panel_entry.family != template:
            problems.append(f"{name}: native_devices.yaml names {template}, panel_devices.yaml {panel_entry.family}; not listed")  # fmt: skip
            continue
        family = families.get(template)
        rows.append({"name": name, "family": template, "kind": _WORD.get(family.kind) if family else None})  # fmt: skip
    return Catalogue(rows, "; ".join(problems) or None)


def db_devices(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The roster's DB devices (``source: gateway``) by name, as registry rows: kind ``settables``, the family their DB device type."""
    found = [d for d in roster if d.get("source") == GATEWAY]
    return [{"name": d["name"], "family": d.get("family") or "", "kind": DB_KIND} for d in sorted(found, key=lambda d: d["name"])]  # fmt: skip


async def roster_of(module: ModuleClient) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:  # fmt: skip
    """The module's roster, or ``None`` and why it could not be read."""
    try:
        status, body = await module.devices()
    except ModuleUnavailable as exc:
        return None, f"the module did not answer: {exc}"
    if status == 200 and isinstance(body, dict) and body.get("devices") is not None:
        return body["devices"], None
    return None, f"the module answered {status}"


def roster_states(roster: list[dict[str, Any]]) -> dict[str, str]:
    """The module's roster as each device's state word (none given: ``unknown``)."""
    return {d["name"]: d.get("state") or "unknown" for d in roster}


def state_word(states: dict[str, str], name: str) -> dict[str, str]:
    """A registry device's word from the roster (``NOT_ON_ROSTER`` when it lacks it) and its kit state: the pickers' and the directory's one rule."""
    word = states.get(name, NOT_ON_ROSTER)
    return {"state": word, "kit": KIT_STATE.get(word, "unknown")}


def registered(settings: Settings, kind: str) -> tuple[list[dict[str, str]], Optional[str]]:  # fmt: skip
    """The registry's devices of ``kind`` by name (``name``, ``family``), and the files' problems (``catalogue``'s)."""
    found = catalogue(settings)
    return [{"name": r["name"], "family": r["family"]} for r in found.rows if r["kind"] == kind], found.problem  # fmt: skip


async def rows_of(settings: Settings, module: ModuleClient, kind: str) -> tuple[list[dict[str, str]], Optional[str]]:  # fmt: skip
    """The picker's devices of ``kind``: the registry's (``registered``), or for ``settables`` the roster's DB devices (the module silent: none, and why)."""
    if kind != DB_KIND:
        return registered(settings, kind)
    roster, why = await roster_of(module)
    return _picker_rows(roster), why


def _picker_rows(roster: Optional[list[dict[str, Any]]]) -> list[dict[str, str]]:
    """The roster's DB devices as picker rows (``name``, ``family``)."""
    return [
        {"name": r["name"], "family": r["family"]} for r in db_devices(roster or [])
    ]


async def listing(settings: Settings, module: ModuleClient, kind: str) -> dict[str, Any]:  # fmt: skip
    """The picker's rows, each with the module's roster state (``unknown`` while the module is silent)."""
    if kind == DB_KIND:
        roster, why = await roster_of(module)
        rows, problem = _picker_rows(roster), None
    else:
        rows, problem = registered(settings, kind)
        roster, why = await roster_of(module)
    states = roster_states(roster or [])
    for row in rows:
        row |= {"state": "unknown", "kit": "unknown"} if why else state_word(states, row["name"])  # fmt: skip
    return {
        "kind": kind,
        "devices": rows,
        "roster": why is None,
        "detail": why or problem,
    }


def tables(body: dict[str, Any], family: Optional[FamilyFile]) -> list[tuple[Optional[str], list[dict[str, Any]]]]:  # fmt: skip
    """The role rows per axis, from the module's device route (its roles are already resolved); ``level`` and units from the family (a DB device has none: every role operator)."""
    by_name = {r.name: r for r in family.roles} if family else {}
    axes = (body.get("axes") or []) if family and family.per_axis else [None]
    out = []
    for axis in axes:
        rows = []
        for entry in body.get("roles") or []:
            if entry.get("axis") != axis:
                continue
            role = by_name.get(entry["role"])
            level = role.level if role else "operator"
            label = entry.get("label") or entry["role"]
            row = {"name": entry["role"], "label": label, "type": entry.get("type")}
            rows.append(
                row
                | {"level": level, "units": role.units if role else entry.get("units")}
            )
        out.append((axis, rows))
    return out


#: The automatic layout puts the writable short roles (bits, enums) in a grid group of their own from this many.
GRID_MIN = 4
_SHORT = ("bit", "enum")


def auto_panel(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The panel of a device whose family has none: every writable role (the module's ``writePv``) in role order, every monitored read-only role a reading.

    One ``Settings`` group; when ``GRID_MIN`` or more of the writable roles
    are short (bits, enums), they go in a ``Switches`` grid group after the
    rest (all of them in one grid group when there is no rest). Nothing hidden.
    """
    writable = [e for e in entries if e.get("writePv")]
    short = [e["role"] for e in writable if e.get("type") in _SHORT]
    rest = [e["role"] for e in writable if e.get("type") not in _SHORT]
    if len(short) < GRID_MIN:
        rail = [{"title": "Settings", "roles": [e["role"] for e in writable], "layout": "fields"}]  # fmt: skip
    elif rest:
        rail = [{"title": "Settings", "roles": rest, "layout": "fields"}, {"title": "Switches", "roles": short, "layout": "grid"}]  # fmt: skip
    else:
        rail = [{"title": "Settings", "roles": short, "layout": "grid"}]
    readings = [e["role"] for e in entries if not e.get("writePv") and e.get("read") == "monitor"]  # fmt: skip
    return {"rail": [g for g in rail if g["roles"]], "readings": readings, "buttons": {}, "tweak": {}}  # fmt: skip


def drawn_panel(panel: dict[str, Any], served: set[str]) -> dict[str, Any]:
    """The panel as this device draws it: only the roles it serves (``omit``), a group with none left out, a tweak only with all four of its roles."""
    tweak = {k: h for k, h in panel.get("tweak", {}).items() if {k, *h.values()} <= served}  # fmt: skip
    rail = [g | {"roles": [r for r in g["roles"] if r in served]} for g in panel.get("rail", [])]  # fmt: skip
    return {
        "rail": [g for g in rail if g["roles"]],
        "readings": [r for r in panel.get("readings", []) if r in served],
        "buttons": {
            r: pair for r, pair in panel.get("buttons", {}).items() if r in served
        },  # fmt: skip
        "tweak": tweak,
        "inside": {
            r for h in tweak.values() for r in h.values()
        },  # drawn with their setpoint
    }


def specs(entries: list[dict[str, Any]], family: Optional[FamilyFile], units: dict[str, Optional[str]]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Each served role as the rail's control macro reads it: its type and label, writable, whole, text, readback, units, and a setpoint's done flag and tolerance."""
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        name = e["role"]
        role = family.role(name) if family and name in family.role_names() else None
        spec: dict[str, Any] = {"name": name, "type": e.get("type"), "label": e.get("label") or name}  # fmt: skip
        spec |= {"writable": bool(e.get("writePv")), "units": units.get(name)}
        spec |= role_spec(role, name)
        if (
            role is None
        ):  # a DB device's text and tolerance: its DB row's, through the route
            spec |= {"text": e.get("text"), "tolerance": e.get("tolerance")}
        out[name] = spec
    return out


#: Roles a kind's page draws itself, never on its rail, whatever the family's panel says: the camera ROI (one
#: draft of four fields applied behind its dialog, since it changes what every scan with the camera saves).
OWN_ROLES = {"camera": frozenset({"min_x", "min_y", "size_x", "size_y"})}


def page_panel(body: dict[str, Any], family: Optional[FamilyFile], rows: list[dict[str, Any]], kind: Optional[str] = None, axis: Optional[str] = None) -> dict[str, Any]:  # fmt: skip
    """What the page draws above the role tables: the panel (the family's, else automatic) for this device's roles on ``axis`` (default: its first), less the kind's own (``OWN_ROLES``), and each role's control spec."""
    axes = body.get("axes") or [None]
    entries = [e for e in body.get("roles") or [] if e.get("axis") == ((axis or axes[0]) if family and family.per_axis else None)]  # fmt: skip
    authored: Optional[dict[str, Any]] = family.panel.model_dump(mode="json") if family and family.panel else None  # fmt: skip
    panel = drawn_panel(authored or auto_panel(entries), {e["role"] for e in entries} - OWN_ROLES.get(kind or "", frozenset()))  # fmt: skip
    units = {r["name"]: r.get("units") for r in rows}
    return {"panel": panel, "specs": specs(entries, family, units), "auto": authored is None}  # fmt: skip


#: A screen's group widget's read: one page draw, never a 2 s poll, so it waits past a slow module reload tick.
EMBED_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Unread:
    """Why the module's answer for a device is no device a page can draw: the page's status, its pane, and whether asking again may help."""

    status: int
    state: str
    title: str
    detail: str
    again: bool = False


def unread(status: int, body: Any, name: str) -> Optional[Unread]:
    """The module's answer for ``name`` read as the device pages read it; ``None`` for a device."""
    why = body.get("detail") if isinstance(body, dict) else None
    if status == 404:
        return Unread(404, "denied", f"{name} is not on the module's roster", "Choose a device above, or check the name.")  # fmt: skip
    if status in (401, 403):  # the read key refused: no amount of asking again helps
        return Unread(status, "denied", "The module refused this display's read", why or f"it answered {status}")  # fmt: skip
    if 400 <= status < 500 or (status == 200 and (not isinstance(body, dict) or "family" not in body)):  # fmt: skip
        return Unread(502, "error", f"The module answered {status} for {name}", why or "not a device the page can read")  # fmt: skip
    if status != 200:  # unreachable, not JSON, or a 5xx
        return Unread(503, "stale", "The module did not answer", why or f"it answered {status}", again=True)  # fmt: skip
    return None


def classify(body: dict[str, Any]) -> tuple[Optional[FamilyFile], Optional[str], Optional[str]]:  # fmt: skip
    """A device's family and page kind word, or why it has no page: the source decides first (a DB device's family is its DB device type, never looked up as a family file)."""
    if body.get("source") == GATEWAY:
        return None, DB_KIND, None
    family = load_families().get(body.get("family") or "")
    if family is None:
        return None, None, f"Its family {body.get('family')!r} is no installed family file."  # fmt: skip
    return family, _WORD.get(family.kind), None


def drawn_of(body: dict[str, Any], family: Optional[FamilyFile], word: Optional[str], axis: Optional[str] = None) -> tuple[list[tuple[Optional[str], list[dict[str, Any]]]], dict[str, Any]]:  # fmt: skip
    """The role tables and what the page draws above them (``page_panel``) on ``axis``."""
    rows = tables(body, family)
    return rows, page_panel(body, family, [r for _, axis_rows in rows for r in axis_rows], word, axis)  # fmt: skip


@dataclass(frozen=True)
class Embedded:
    """A screen's group widget as the device page draws that group: the panel (its rail the one group) and specs, or why not."""

    panel: Optional[dict[str, Any]] = None
    specs: Optional[dict[str, dict[str, Any]]] = None
    problem: Optional[str] = None


def group_panel(body: dict[str, Any], group: str, axis: Optional[str] = None) -> Embedded:  # fmt: skip
    """The device's rail group ``group`` (``settings`` on an automatic layout: every group of it) and the panel's readings, as its page draws them."""
    family, word, problem = classify(body)
    if problem:
        return Embedded(problem=problem)
    _, drawn = drawn_of(body, family, word, axis)
    auto, rail = drawn["auto"], drawn["panel"]["rail"]
    rail = rail if auto and group == AUTO_GROUP else [g for g in rail if not auto and g["title"] == group]  # fmt: skip
    if not rail:
        return Embedded(problem=f"the device as the module serves it draws no group {group!r}")  # fmt: skip
    return Embedded(drawn["panel"] | {"rail": rail}, drawn["specs"])


async def embedded(module: ModuleClient, keys: Collection[GroupKey]) -> dict[GroupKey, Embedded]:  # fmt: skip
    """Each group widget's group, one module read per device (concurrently); a device the module did not give is a problem, never a 500."""
    names = sorted({name for name, _, _ in keys})

    async def read(name: str) -> tuple[int, Any]:
        try:
            return await module.device(name, timeout=EMBED_TIMEOUT_S)
        except ModuleUnavailable as exc:
            return 0, {"detail": str(exc)}

    bodies = dict(zip(names, await asyncio.gather(*(read(n) for n in names))))
    out = {}
    for key in keys:
        status, body = bodies[key[0]]
        if why := unread(
            status, body, key[0]
        ):  # a screen has no picker: its own advice for a device off the roster
            detail = "check the screen's device name, or why the module has not loaded it" if why.status == 404 else why.detail  # fmt: skip
            out[key] = Embedded(problem=f"{why.title}: {detail}" + ("; reload the screen to draw it" if why.again else ""))  # fmt: skip
        else:
            out[key] = group_panel(body, key[2], key[1])
    return out


async def resolve(module: ModuleClient, kind: str, name: str) -> tuple[int, dict[str, Any]]:  # fmt: skip
    """One read of the module's device route: the page's status and context, or the pane that says why not."""
    try:
        status, body = await module.device(name)
    except ModuleUnavailable as exc:
        status, body = 0, {"detail": str(exc)}
    if why := unread(status, body, name):
        note = f"{why.detail}; this page asks again every few seconds." if why.again else why.detail  # fmt: skip
        return why.status, {"pane": (why.state, why.title, note)} | ({"recover": True} if why.again else {})  # fmt: skip
    family, word, problem = classify(body)
    if problem:
        return 422, {"pane": ("denied", f"{name} has no page", problem)}
    if word != kind:
        what = family.template if family else f"{name} (a GEECS DB device, {body['family'] or 'no type'})"  # fmt: skip
        return 422, {"pane": ("denied", f"Wrong kind: {what} is a {word}", ""), "other": word}  # fmt: skip
    rows, drawn = drawn_of(body, family, word)
    return 200, {"pane": None, "device": body, "family": family, "tables": rows, "trace": trace_roles(family)} | drawn  # fmt: skip


def router(settings: Settings) -> APIRouter:
    """The picker, its roster read, and the device page."""
    r = APIRouter()

    def page(request: Request, status: int, **context: Any) -> Any:
        context = {
            "kinds": list(KINDS),
            "kit_state": KIT_STATE,
            "device": None,
        } | context
        templates = request.app.state.templates
        name = (
            KINDS[context["kind"]].template
            if context["kind"] in KINDS
            else "device.html"
        )
        return templates.TemplateResponse(request, name, context, status_code=status)  # fmt: skip

    def no_kind(request: Request, kind: str) -> Any:
        pane = ("error", f"No device kind {kind!r}", "The kinds are " + ", ".join(KINDS) + ".")  # fmt: skip
        return page(request, 404, kind=kind, name=None, pane=pane)

    @r.get("/device/{kind}")
    async def picker(request: Request, kind: str, name: str = "", typed: str = "") -> Any:  # fmt: skip
        if kind not in KINDS:
            return no_kind(request, kind)
        if target := (typed.strip() or name.strip()):  # the form's Go, without script
            url = f"{root_of(request)}/device/{kind}/{quote(target, safe='')}"
            return RedirectResponse(url, status_code=303)
        rows, problem = await rows_of(settings, request.app.state.module, kind)
        return page(request, 200, kind=kind, name=None, rows=rows, problem=problem, pane=None)  # fmt: skip

    @r.get("/api/device/{kind}")
    async def roster(request: Request, kind: str) -> Any:
        if kind not in KINDS:
            return {"kind": kind, "devices": [], "roster": False, "detail": f"no kind {kind!r}"}  # fmt: skip
        return await listing(settings, request.app.state.module, kind)

    @r.get("/device/{kind}/{name}")
    async def device(request: Request, kind: str, name: str) -> Any:
        if kind not in KINDS:
            return no_kind(request, kind)
        rows, problem = await rows_of(settings, request.app.state.module, kind)
        view = {"kind": kind, "name": name, "rows": rows, "problem": problem}
        status, context = await resolve(request.app.state.module, kind, name)
        return page(request, status, **view, **context)

    return r
