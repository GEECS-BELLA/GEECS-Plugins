"""The one page: a day + scan form, the column pickers, the plot.

The URL names what is open — ``?day=&scan=&set=`` — and every control
lives in a named settings set (:mod:`geecs_portal.plotter.settings`), a file
the page saves on each pick, toggle or scan step. The form POSTs the
controls; the server merges them into a set, saves it, and redirects
back to the short URL. "Save as" POSTs the same form to ``/save-as``,
which writes the controls to a new name instead.

Each viewer saves into **their own set**, the one with their name: the
page asks the name once (a URL without ``set=`` shows only that question;
the browser remembers the answer and the page script sends it as
``owner`` with every save). Opening someone else's set shows it and
writes nothing; the viewer's first change copies it, with the change,
into the viewer's own set and switches to it, so a shared link never
lets one person rewrite another's plots. The owner's set changes only
by "Save as" onto its name, which asks before replacing.

With ``x_auto`` on (the default) X is the open scan's variable; a hand
pick on X saves ``x_auto`` off. The page renders the pick lists
server-side; its script fetches the figure from ``/api/figure`` and
draws it.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from geecs_data_utils.tiled_catalog import RunDetail

from geecs_portal.plotter import figures, images, scans
from geecs_portal.plotter.routes import Context, read_form
from geecs_portal.plotter.routes.images import strip_code, strip_tiles
from geecs_portal.plotter.settings import (
    CHOICES,
    DEFAULT_SET,
    MAX_FIT_ORDER,
    MAX_IMAGE_COLUMNS,
    MAX_Y_LISTS,
    NameRefused,
    PlotterSettings,
    SettingsError,
    form_value,
    merged,
    set_name,
)

#: s-file columns every scan has, never a useful default Y (nor is any
#: device's ``<device> shot #`` counter).
_BOOKKEEPING = ("Shotnumber", "Bin #")


def _opened(form: dict[str, str], name: str) -> str:
    """The page URL for *form*'s day and scan under set *name*, relative."""
    query = {k: form[k] for k in ("day", "scan") if form.get(k)}
    return "./?" + urlencode({**query, "set": name})


def register(router: APIRouter, ctx: Context) -> None:
    """Add the page routes to *router*."""

    @router.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        day: Optional[date] = None,
        scan: Optional[int] = None,
        name: Optional[str] = Query(None, alias="set"),
        here_day: Optional[date] = None,
        here_scan: Optional[int] = None,
    ) -> Response:
        """The plotter page; without a day and scan, the set's last scan.

        ``here_day`` / ``here_scan`` are the scan open when the sets list
        switched: kept when the new set has none of its own. A set that
        has never opened a scan, with nothing open, is the empty form.
        Without a set, the page asks the viewer's name (their own set).
        """
        if name is None:
            return ctx.templates.TemplateResponse(
                request,
                "index.html",
                {
                    "experiment": ctx.experiment,
                    **_page(day, scan, PlotterSettings()),
                    "ask_name": True,
                    "set": "",
                    "sets": [],
                },
            )
        if day is None and scan is None:
            here = (
                {"day": here_day.isoformat(), "scan": str(here_scan)}
                if here_day is not None and here_scan is not None
                else None
            )
            target = _last_scan(ctx, name) or (here and _opened(here, name))
            if target:
                return RedirectResponse(target, status_code=303)
        return _render(ctx, request, day, scan, name)

    @router.post("/", response_class=HTMLResponse)
    async def apply(request: Request) -> Response:
        """Save the form's controls into the open set, then reopen the page."""
        return await _save(ctx, request, as_new=False)

    @router.post("/save-as", response_class=HTMLResponse)
    async def save_as(request: Request) -> Response:
        """Save the form's controls as a new set, then open that set."""
        return await _save(ctx, request, as_new=True)


def _render(
    ctx: Context,
    request: Request,
    day: Optional[date],
    scan: Optional[int],
    name: str,
    *,
    problem: Optional[str] = None,
    save_as: str = "",
    name_error: Optional[str] = None,
    status_code: int = 200,
) -> HTMLResponse:
    """The page for *day*/*scan* under set *name*; *problem* shown if given.

    *save_as* refills the Save-as box after any refusal, so nothing typed
    is lost; *name_error* marks the box itself invalid, and only when the
    name was the reason (the kit's ``.field`` validation).
    """
    settings = PlotterSettings()
    try:
        name = set_name(name)
        settings = ctx.store.load(name)
    except SettingsError as bad:
        problem = problem or str(bad)
    page = _page(day, scan, settings)
    page.update(
        {
            "set": name,
            "sets": ctx.store.names(),
            "settings_folder": str(ctx.store.folder),
            "settings_origin": ctx.store.origin,
            "settings_error": problem,
            "save_as": save_as,
            "name_error": name_error,
        }
    )
    if day is not None and scan is not None:
        try:
            run = ctx.run(day, scan)
        except HTTPException as missing:
            page["error"] = missing.detail
        else:
            _fill(page, run, settings, ctx.experiment, ctx.logbook_send)
    return ctx.templates.TemplateResponse(
        request,
        "index.html",
        {"experiment": ctx.experiment, **page},
        status_code=status_code,
    )


async def _save(ctx: Context, request: Request, *, as_new: bool) -> Response:
    """Merge the posted controls into a set (the open one, or a new name).

    The store work runs in the threadpool, as the sync routes' does: a hung
    share stalls this request, never the server.
    """
    form = await read_form(request)
    try:
        target = await run_in_threadpool(_store, ctx, form, as_new)
    except SettingsError as bad:
        day, scan = _when(form)
        refused = isinstance(bad, NameRefused)
        return await run_in_threadpool(
            _render,
            ctx,
            request,
            day,
            scan,
            form.get("set", DEFAULT_SET),
            problem=None if refused else str(bad),
            save_as=form.get("save_as", ""),
            name_error=str(bad) if refused else None,
            status_code=422,
        )
    return RedirectResponse(_opened(form, target), status_code=303)


def _store(ctx: Context, form: dict[str, str], as_new: bool) -> str:
    """Save *form* into its set, or a new one; the name saved to."""
    source = set_name(form.get("set", DEFAULT_SET))
    target = source
    # A change saves into the viewer's own set (their name, which the page
    # sends as ``owner``), with the open set's controls copied under it when
    # that is someone else's. A change that names no owner is refused, not
    # saved where it is: that would let a first-time viewer of a shared
    # link rewrite its owner's set. Save as names its target itself.
    if not as_new:
        owner = form.get("owner", "")
        if not owner:
            raise SettingsError(
                "say who you are before changing anything: your changes save "
                "into your own set (the page asks your name)"
            )
        target = set_name(owner)
    if as_new:
        try:
            target = set_name(form.get("save_as", ""))
        except SettingsError as bad:
            raise NameRefused(str(bad)) from None
        # Replacing a saved set is a choice the page asks about first (its
        # dialog posts overwrite=true); without it a taken name is refused.
        if ctx.store.exists(target) and form.get("overwrite") != "true":
            raise NameRefused(f"a set named {target!r} is already saved")
    ctx.store.save(target, merged(ctx.store.load(source), form))
    return target


def _last_scan(ctx: Context, name: str) -> Optional[str]:
    """The page URL for the scan set *name* last opened, if it has one.

    A bad name or an unreadable set is left for the page to report.
    """
    try:
        settings = ctx.store.load(set_name(name))
    except SettingsError:
        return None
    if settings.day is None or settings.scan is None:
        return None
    last = {"day": settings.day.isoformat(), "scan": str(settings.scan)}
    return _opened(last, set_name(name))


def _when(form: dict[str, str]) -> tuple[Optional[date], Optional[int]]:
    """The form's day and scan, ``None`` where absent or malformed.

    The same rules as the set's (``settings.form_value``), forgiving: a
    page that reports a refused save still renders.
    """

    def parsed(field: str) -> object:
        try:
            return form_value(field, form[field]) if form.get(field) else None
        except SettingsError:
            return None

    day, scan = parsed("day"), parsed("scan")
    return (
        day if isinstance(day, date) else None,
        scan if isinstance(scan, int) else None,
    )


def _page(day: Optional[date], scan: Optional[int], settings: PlotterSettings) -> dict:
    """The template context before a scan is opened."""
    return {
        "day": day,
        "scan": scan,
        "error": None,
        "columns": [],
        "x": "",
        "y": "",
        "xalias": settings.x_alias,
        "yalias": settings.y_alias,
        "xauto": settings.x_auto,
        "xf": settings.x_filter,
        "yf": settings.y_filter,
        "y2": "",
        "y3": "",
        "y2alias": settings.y2_alias,
        "y3alias": settings.y3_alias,
        "y2f": settings.y2_filter,
        "y3f": settings.y3_filter,
        "ycount": settings.y_count,
        "ycounts": range(1, MAX_Y_LISTS + 1),
        "y2_kept": "",
        "y3_kept": "",
        "yfit": settings.y_fit,
        "y2fit": settings.y2_fit,
        "y3fit": settings.y3_fit,
        "yorder": settings.y_fit_order,
        "y2order": settings.y2_fit_order,
        "y3order": settings.y3_fit_order,
        "fits": figures.FITS,
        "max_fit_order": MAX_FIT_ORDER,
        # The XY plot's view, shot filters, bins and cosmetics (JSON text).
        "view": _choice(settings, "xy_view"),
        "filters": settings.shot_filters,
        "bincfg": settings.bin_config,
        "display": settings.display,
        "filter_groups": _filter_groups(settings.shot_filters),
        "uid": "",
        "slots": [],
        "caption": "",
        # The tab, and the image strip's controls. A choice a newer copy
        # saved that this one lacks shows this copy's default.
        "tab": _choice(settings, "tab"),
        "devf": settings.image_device_filter,
        "strip": _choice(settings, "image_strip"),
        "layout": _choice(settings, "image_layout"),
        "cols": settings.image_columns,
        "max_cols": MAX_IMAGE_COLUMNS,
        "cmap": settings.image_cmap,
        "cmaps": images.CMAPS,
        "zscale": _choice(settings, "image_zscale"),
        "zmin": settings.image_zmin,
        "zmax": settings.image_zmax,
        "roi": "",
        "display_query": "",
        "levels_url": "",
        "devices": [],
        "device": "",
        "device_kept": "",
        "device_problem": None,
        "tiles": [],
        "strip_code": "",
    }


def _filter_groups(text: str) -> list[dict]:
    """The shot filters' groups as the rail's chips; none for bad or empty text."""
    from geecs_portal import analysis

    try:
        filters = analysis.parse_filters(text)
    except analysis.BadParam:
        return []
    return [
        {
            "name": g.name or f"filter {i + 1}",
            "enabled": g.enabled,
            "conditions": len(g.conditions),
        }
        for i, g in enumerate(filters.groups)
    ]


def _choice(settings: PlotterSettings, key: str) -> str:
    """The stored choice, or its default when this copy does not offer it."""
    value = getattr(settings, key)
    return value if value in CHOICES[key] else PlotterSettings.model_fields[key].default


def _fill(
    page: dict,
    run: RunDetail,
    settings: PlotterSettings,
    experiment: str = "",
    logbook_send: bool = False,
) -> None:
    """Add the open scan's lists and the picks the settings resolve to."""
    frame = scans.load_frame(run)
    if page["tab"] == "images":
        _fill_images(page, run, frame, settings, experiment)
    picks = scans.columns(frame)
    page["columns"] = picks
    page["uid"] = run.summary.uid
    page["logbook_send"] = logbook_send
    page["xnames"] = {c: scans.display_name(c, settings.x_alias) for c in picks}
    page["ynames"] = {c: scans.display_name(c, settings.y_alias) for c in picks}
    page["shots"] = len(frame)
    # GEECSplotter's "autofill par1": with x_auto (the default) X is the
    # scan variable of whichever scan is open. With it off, an unset X is
    # the scan variable too and X "" is the shot number.
    x = settings.x
    chosen_x = scans.default_x(run, picks) if settings.x_auto or x is None else x
    page["x"] = chosen_x if chosen_x in picks else ""
    # Default Y: the first measurement, not the bookkeeping.
    others = [
        c
        for c in picks
        if c not in {page["x"], *_BOOKKEEPING} and not c.endswith(" shot #")
    ]
    # A scan stopped before its first shot has no picks at all.
    y = settings.y
    page["y"] = y if y in picks else next(iter(others or picks), "")
    # Y2 and Y3, when the count shows them: the saved pick, else the next
    # measurements after Y. On a scan without a saved pick's column they
    # draw nothing, but the list still holds it ("kept"), so the next click
    # here posts it back instead of erasing it.
    spare = iter([c for c in others if c != page["y"]] or picks)
    lists = (("y2", settings.y2_alias, 2), ("y3", settings.y3_alias, 3))
    for key, alias, n in lists:
        page[f"{key}names"] = {c: scans.display_name(c, alias) for c in picks}
        default = next(spare, "")
        if settings.y_count < n:
            continue
        picked = getattr(settings, key) or default
        page[key] = picked if picked in picks else ""
        page[f"{key}_kept"] = "" if picked in picks else picked
    # The drawn Y slots (0-2: the display drawer's colour per slot) and
    # the logbook caption: what the plot shows, as the panel heads it.
    shown = [(0, page["y"]), (1, page["y2"]), (2, page["y3"])]
    page["slots"] = [i for i, c in shown if c]
    names = [page[f"{k}names"][c] for k, (_, c) in zip(("y", "y2", "y3"), shown) if c]
    x_name = page["xnames"][page["x"]] if page["x"] else "shot #"
    page["caption"] = f"{', '.join(names)} vs {x_name}" + (
        " (binned)" if page["view"] == "bin" else ""
    )


def _fill_images(
    page: dict, run: RunDetail, frame, settings: PlotterSettings, experiment: str
) -> None:
    """Add the device list and, for a picture device, the strip's tiles.

    The saved device, when this scan has no folder for it, is kept in the
    list ("not in this scan") so the next click does not erase it, and the
    strip says so. With none saved the page asks for one: finding a camera
    by probing the folders one by one costs tens of seconds cold over the
    share on a scan of trace and scalar devices.
    """
    folder = scans.scan_folder(run)
    devices = images.devices(folder)
    page["devices"] = devices
    saved = settings.image_device
    if saved and saved not in devices:
        page["device_kept"] = saved
        page["device_problem"] = f"{saved} is not in this scan"
        return
    device = saved or ""
    page["device"] = device
    if not devices:
        page["device_problem"] = "this scan has no device folders"
        return
    if not device:
        page["device_problem"] = "pick a camera from the list"
        return
    if not images.is_image_device(folder, device, devices):
        page["device_problem"] = f"{device} has no pictures (traces or a vendor format)"
        return
    day, scan = page["day"], page["scan"]
    try:
        page["tiles"] = strip_tiles(
            run, frame, page["strip"], day=day, scan=scan, device=device
        )
    except HTTPException as missing:
        page["device_problem"] = missing.detail
        return
    page["strip_code"] = strip_code(day, scan, device, page["strip"], experiment)
    roi = settings.image_rois.get(device)
    page["roi"] = ",".join(str(v) for v in roi) if roi else ""
    page["display_query"] = display_query(page)
    page["levels_url"] = "api/images/levels?" + urlencode(
        {
            "day": day.isoformat(),
            "scan": scan,
            "device": device,
            "strip": page["strip"],
            "roi": page["roi"],
        }
    )


def display_query(page: dict) -> str:
    """The display half of a tile's URL (plotter.js builds the same).

    The colour map, the ROI and, on a fixed z scale with both values set,
    the black and white; a shared z scale's values come from
    ``/api/images/levels`` in the browser.
    """
    query = {"cmap": page["cmap"]}
    if page["roi"]:
        query["roi"] = page["roi"]
    if page["zscale"] == "fixed" and None not in (page["zmin"], page["zmax"]):
        query.update(zmin=page["zmin"], zmax=page["zmax"])
    return urlencode(query)
