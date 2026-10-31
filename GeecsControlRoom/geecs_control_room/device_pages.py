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
until it answers), never a blank page and never a 500. Read only here: PV
names, verdicts and values come from the module and Hare in the browser.
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
from geecs_control_room.screens import KIT_STATE, registry
from geecs_control_room.settings import Settings


@dataclass(frozen=True)
class Kind:
    """A page kind: the family ``kind`` its devices' families carry, and the template that draws them."""

    family_kind: str
    template: str


#: The URL's kind word → the family kind → the page. A new family of a kind needs no code.
KINDS = {
    "camera": Kind("aravis_camera", "device.html"),
    "scope": Kind("array_detector", "device.html"),
    "motor": Kind("epics_motor", "device.html"),
    "supply": Kind("setpoint_device", "device.html"),
}
_WORD = {k.family_kind: word for word, k in KINDS.items()}


def kind_of(family: Optional[str]) -> Optional[str]:
    """The kind word of an installed family's devices; ``None`` for no such family (a DB device type)."""
    found = load_families().get(family or "")
    return _WORD.get(found.kind) if found else None


def registered(settings: Settings, kind: str) -> tuple[list[dict[str, str]], Optional[str]]:  # fmt: skip
    """The registry's devices of ``kind`` by name (``name``, ``family``), and the files' problems.

    The module's join rule (GeecsCanopy ``native.py``): a device both files name
    with different families is no kind's; it is left out and named in the problem.
    """
    try:
        panel, native = registry(settings.experiment_dir)
    except (ValidationError, yaml.YAMLError, OSError) as exc:
        return [], str(exc)
    families, want = load_families(), KINDS[kind].family_kind
    rows, problems = [], []
    for name in sorted({*native.devices, *panel.devices}):
        entry, panel_entry = native.devices.get(name), panel.devices.get(name)
        template = entry.family_template if entry else panel_entry.family
        if panel_entry and entry and panel_entry.family != template:
            problems.append(f"{name}: native_devices.yaml names {template}, panel_devices.yaml {panel_entry.family}; not listed")  # fmt: skip
            continue
        family = families.get(template)
        if family and family.kind == want:
            rows.append({"name": name, "family": template})
    return rows, "; ".join(problems) or None


async def listing(settings: Settings, module: ModuleClient, kind: str) -> dict[str, Any]:  # fmt: skip
    """The picker's rows, each with the module's roster state (``unknown`` while the module is silent)."""
    rows, problem = registered(settings, kind)
    states, why = {}, None
    try:
        status, body = await module.devices()
        if status == 200 and isinstance(body, dict) and body.get("devices") is not None:
            states = {d["name"]: d.get("state") or "unknown" for d in body["devices"]}
        else:
            why = f"the module answered {status}"
    except ModuleUnavailable as exc:
        why = f"the module did not answer: {exc}"
    for row in rows:
        word = "unknown" if why else states.get(row["name"], "not on the roster")
        row |= {"state": word, "kit": KIT_STATE.get(word, "unknown")}
    return {
        "kind": kind,
        "devices": rows,
        "roster": why is None,
        "detail": why or problem,
    }


def tables(body: dict[str, Any], family: FamilyFile) -> list[tuple[Optional[str], list[dict[str, Any]]]]:  # fmt: skip
    """The role rows per axis, from the module's device route (its roles are already resolved); ``level`` and units from the family."""
    by_name = {r.name: r for r in family.roles}
    axes = (body.get("axes") or []) if family.per_axis else [None]
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
            rows.append(row | {"level": level, "units": role.units if role else None})
        out.append((axis, rows))
    return out


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
    family = load_families().get(body["family"] or "")
    if family is None:
        return 422, {"pane": ("denied", f"{name} is not a native device", f"Its family {body['family']!r} is no installed family file (a GEECS DB device type): a screen draws it.")}  # fmt: skip
    if (word := _WORD.get(family.kind)) != kind:
        return 422, {"pane": ("denied", f"Wrong kind: {family.template} is a {word}", ""), "other": word}  # fmt: skip
    return 200, {"pane": None, "device": body, "family": family, "tables": tables(body, family)}  # fmt: skip


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
        return templates.TemplateResponse(request, "device.html", context, status_code=status)  # fmt: skip

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
        rows, problem = registered(settings, kind)
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
        rows, problem = registered(settings, kind)
        view = {"kind": kind, "name": name, "rows": rows, "problem": problem}
        status, context = await resolve(request.app.state.module, kind, name)
        return page(request, status, **view, **context)

    return r
