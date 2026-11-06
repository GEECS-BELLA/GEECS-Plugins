"""Device pages: one page per kind of hardware, any device of that kind (``Planning/device_pages.md``).

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

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from geecs_schemas import FamilyFile, load_families
from geecs_web_theme.web import root_of
from pydantic import ValidationError

from geecs_control_room.module_client import ModuleClient, ModuleUnavailable
from geecs_control_room.screens import KIT_STATE, registry, role_spec
from geecs_control_room.settings import Settings
from geecs_control_room.traces import trace_roles


@dataclass(frozen=True)
class Kind:
    """A page kind: the family ``kind`` its devices' families carry (``None``: the DB devices), and the template that draws them."""

    family_kind: Optional[str]
    template: str


#: The kind word of the GEECS DB devices the module serves through the gateway (no family file).
DB_KIND = "settables"
#: The roster's ``source`` of a DB device.
GATEWAY = "gateway"

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


def page_panel(body: dict[str, Any], family: Optional[FamilyFile], rows: list[dict[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """What the page draws above the role tables: the panel (the family's, else automatic) for this device's roles, and each role's control spec."""
    axes = body.get("axes") or [None]
    entries = [e for e in body.get("roles") or [] if e.get("axis") == (axes[0] if family and family.per_axis else None)]  # fmt: skip
    authored: Optional[dict[str, Any]] = family.panel.model_dump(mode="json") if family and family.panel else None  # fmt: skip
    panel = drawn_panel(authored or auto_panel(entries), {e["role"] for e in entries})
    units = {r["name"]: r.get("units") for r in rows}
    return {"panel": panel, "specs": specs(entries, family, units), "auto": authored is None}  # fmt: skip


async def resolve(module: ModuleClient, kind: str, name: str) -> tuple[int, dict[str, Any]]:  # fmt: skip
    """One read of the module's device route: the page's status and context, or the pane that says why not."""
    try:
        status, body = await module.device(name)
    except ModuleUnavailable as exc:
        status, body = 0, {"detail": str(exc)}
    if status == 404:
        return 404, {"pane": ("denied", f"{name} is not on the module's roster", "Choose a device above, or check the name.")}  # fmt: skip
    why = body.get("detail") if isinstance(body, dict) else None
    if status in (401, 403):  # the read key refused: no amount of asking again helps
        return status, {"pane": ("denied", "The module refused this display's read", why or f"it answered {status}")}  # fmt: skip
    if 400 <= status < 500 or (status == 200 and (not isinstance(body, dict) or "family" not in body)):  # fmt: skip
        return 502, {"pane": ("error", f"The module answered {status} for {name}", why or "not a device the page can read")}  # fmt: skip
    if status != 200:  # unreachable, not JSON, or a 5xx: the page asks again
        note = (
            f"{why or f'it answered {status}'}; this page asks again every few seconds."
        )
        return 503, {"pane": ("stale", "The module did not answer", note), "recover": True}  # fmt: skip
    # the source decides first: a DB device's family is its DB device type, never looked up as a family file
    gateway = body.get("source") == GATEWAY
    family = None if gateway else load_families().get(body["family"] or "")
    if family is None and not gateway:
        return 422, {"pane": ("denied", f"{name} has no page", f"Its family {body['family']!r} is no installed family file.")}  # fmt: skip
    word = DB_KIND if gateway else _WORD.get(family.kind)
    if word != kind:
        what = family.template if family else f"{name} (a GEECS DB device, {body['family'] or 'no type'})"  # fmt: skip
        return 422, {"pane": ("denied", f"Wrong kind: {what} is a {word}", ""), "other": word}  # fmt: skip
    rows = tables(body, family)
    drawn = page_panel(body, family, [r for _, axis_rows in rows for r in axis_rows])
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
