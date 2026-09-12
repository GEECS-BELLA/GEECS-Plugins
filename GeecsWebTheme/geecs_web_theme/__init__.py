"""The shared GEECS web palette: where the files are, and what they offer.

This package ships a stylesheet and a small script and nothing else. It has
no runtime dependencies on purpose — anything that can serve a static
directory can use it, and nothing it serves needs a Python import to work.

Usage from a host application::

    from geecs_web_theme import static_dir
    app.mount("/theme", StaticFiles(directory=str(static_dir())), name="theme")

then, in a template's ``<head>`` — the boot script NOT deferred, so the
palette is stamped before first paint; the picker script deferred::

    <script src="/theme/theme-boot.js"></script>
    <link rel="stylesheet" href="/theme/theme.css">
    <script src="/theme/theme.js" defer></script>

Behind a reverse proxy build those from the request's ``root_path``; the
portal does this with its ``{{ root }}`` idiom.

and wherever the control belongs::

    <div data-theme-picker></div>
"""

from pathlib import Path

__all__ = [
    "THEMES",
    "DEFAULT_THEME",
    "static_dir",
    "theme_css",
    "theme_js",
    "theme_boot_js",
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
