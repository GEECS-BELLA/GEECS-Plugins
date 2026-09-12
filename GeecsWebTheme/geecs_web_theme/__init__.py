"""The shared GEECS web palette and layout kit: the files, and what they offer.

Two layers ship here. ``theme.css`` settles **colour** — the token
vocabulary and the palettes. ``kit.css`` settles **everything else** — the
page shell, the three containers, the status chip, controls, tables, the
five pane states and the overlay ladder. They are separate files because
they answer separate questions and a surface may adopt the first without
the second, but the kit is the layer that stops a third web surface from
inventing a third set of answers.

``kit.html`` is the kit's own reference page: every component rendered in
the real theme, at whichever palette and density the viewer picks. It is a
static file beside the stylesheets, so a host that mounts this package
already serves it — the portal reaches it at ``/theme/kit.html`` with no
route of its own.

This package ships stylesheets and small scripts and nothing else. It has
no runtime dependencies on purpose — anything that can serve a static
directory can use it, and nothing it serves needs a Python import to work.

Usage from a host application::

    from geecs_web_theme import static_dir
    app.mount("/theme", StaticFiles(directory=str(static_dir())), name="theme")

then, in a template's ``<head>`` — the boot script NOT deferred, so the
palette is stamped before first paint; the picker script deferred::

    <script src="/theme/theme-boot.js"></script>
    <link rel="stylesheet" href="/theme/theme.css">
    <link rel="stylesheet" href="/theme/kit.css">
    <script src="/theme/theme.js" defer></script>
    <script src="/theme/kit.js" defer></script>

Behind a reverse proxy build those from the request's ``root_path``; the
portal does this with its ``{{ root }}`` idiom.

Give ``<body>`` the ``kit`` class so the kit's base rules apply, and put
the two controls wherever they belong::

    <div data-theme-picker></div>
    <div data-density-picker></div>
"""

from pathlib import Path

__all__ = [
    "THEMES",
    "DEFAULT_THEME",
    "DENSITIES",
    "DEFAULT_DENSITY",
    "STATES",
    "PANE_STATES",
    "static_dir",
    "theme_css",
    "theme_js",
    "theme_boot_js",
    "kit_css",
    "kit_js",
    "kit_html",
]

#: The palettes on offer. Kept here as well as in the stylesheet so a host
#: can name them in a menu or a preference without parsing CSS.
THEMES: dict[str, str] = {
    "bella": "BELLA Center — red on black",
    "laser": "Laser room — 532 nm pump green, breadboard red for criticals",
    "plasma": "Hydrogen plasma — Balmer H-beta cyan, H-alpha for the agent",
}

#: What a viewer gets before they choose. The runtime authority is
#: ``theme-boot.js``; ``tests/test_no_literal_colours.py`` pins the two.
DEFAULT_THEME = "laser"

#: The spacing scales ``kit.css`` implements. Same arrangement as THEMES:
#: named here so a host can offer them, defined for real in
#: ``theme-boot.js``, and the two pinned together by a test.
DENSITIES: dict[str, str] = {
    "comfortable": "Roomier rows — reading and writing",
    "compact": "Tighter rows — tables and live panels",
}

#: What a viewer gets before they choose a density.
DEFAULT_DENSITY = "comfortable"

#: The status vocabulary, in the order a thing moves through it. These
#: replace the fifteen class names the portal and the logbook each invented
#: (``done``/``success``/``ok``, ``fail``/``failed``/``aborted``, …), and a
#: surface names one through ``data-state`` on ``.chip`` or ``.dot``.
#:
#: Named here, and not only in the CSS, for the reason a mistyped state is
#: dangerous: ``data-state="no_data"`` matches no rule and still renders a
#: plausible neutral pill, so it survives review and the browser alike. A
#: host that renders these from a constant cannot mistype one, and
#: ``tests/test_no_literal_colours.py`` pins the list to ``kit.css``.
STATES: dict[str, str] = {
    "queued": "Accepted, not started",
    "running": "In progress now",
    "ok": "Finished as intended",
    "degraded": "Finished, but less than asked",
    "failed": "Did not finish",
    "unknown": "We have no information",
    "agent": "Written by software, not a person",
}

#: The states every pane owes its reader, named on ``.state`` and
#: ``.banner``. ``loading`` and ``empty`` exist on both surfaces today in
#: private forms; ``error``, ``stale`` and ``denied`` exist on neither, and
#: a control surface cannot open without the last two — a live value that
#: silently stops updating is worse than no value.
PANE_STATES: dict[str, str] = {
    "loading": "Named, so the reader knows whether to wait",
    "empty": "The query succeeded and matched nothing",
    "error": "The request failed; nothing was lost",
    "stale": "Showing a value older than it should be",
    "denied": "Someone else holds it, or you may only read",
}


def static_dir() -> Path:
    """Return the directory holding ``theme.css`` and ``theme.js``.

    Returns
    -------
    Path
        A directory suitable for mounting as static files.
    """
    return Path(__file__).parent / "static"


def theme_css() -> Path:
    """Return the path to the stylesheet."""
    return static_dir() / "theme.css"


def theme_js() -> Path:
    """Return the path to the picker script (load deferred)."""
    return static_dir() / "theme.js"


def theme_boot_js() -> Path:
    """Return the path to the boot script (load in ``<head>``, not deferred)."""
    return static_dir() / "theme-boot.js"


def kit_css() -> Path:
    """Return the path to the layout kit stylesheet.

    Load it *after* ``theme.css``: the kit styles through that file's
    tokens and defines none of its own.
    """
    return static_dir() / "kit.css"


def kit_js() -> Path:
    """Return the path to the kit script (load deferred).

    Optional. Without it the page still renders and ``<details>`` still
    opens; only the drawer, the dialog helper and the density control go
    missing.
    """
    return static_dir() / "kit.js"


def kit_html() -> Path:
    """Return the path to the kit's reference page.

    A static file in the same directory, so any host already mounting
    :func:`static_dir` serves it at ``<mount>/kit.html`` without a route.
    """
    return static_dir() / "kit.html"
