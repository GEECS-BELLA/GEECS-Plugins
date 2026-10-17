"""Where the three services are: ``config.ini`` ``[mcp]`` and nothing else.

The server talks to the scanner (:8300), the Data Portal (:8200) and the
logbook (:8400) and to nothing else; their addresses come from the fleet
contract file, ``~/.config/geecs_python_api/config.ini``, under ``[mcp]``::

    [mcp]
    scanner_url = http://<services host>:8300
    portal_url  = http://<services host>:8200
    logbook_url = http://<services host>:8400

``deploy/bootstrap_host.sh`` renders them from ``site.env``'s
``GEECS_SERVICES_HOST`` (ports are fleet constants).  An unset key leaves
that service's verbs answering ``service_unreachable`` with a message
naming the key — the server always starts (DESIGN.md §10).
"""

from __future__ import annotations

import configparser
import threading
from pathlib import Path
from typing import Optional

#: The fleet contract file (GEECS-Core/DESIGN.md: the name is permanent).
CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")

#: The services this server is a client of, and their ``[mcp]`` keys.
SERVICES = ("scanner", "portal", "logbook")
URL_KEYS = {name: f"{name}_url" for name in SERVICES}

_cache: dict[str, Optional[str]] = {}
_lock = threading.Lock()


def _read_urls() -> dict[str, Optional[str]]:
    path = CONFIG_PATH.expanduser()
    urls: dict[str, Optional[str]] = {name: None for name in SERVICES}
    if not path.exists():
        return urls
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except (OSError, configparser.Error):
        return urls
    for name in SERVICES:
        value = parser.get("mcp", URL_KEYS[name], fallback="").strip()
        urls[name] = value.rstrip("/") or None
    return urls


def service_url(service: str) -> Optional[str]:
    """The configured base URL of *service*, or ``None`` when unset.

    Read once per process and cached; tests monkeypatch this function on
    the module (tools call ``config.service_url`` through the module
    attribute) or call :func:`clear_config_cache`.
    """
    if service not in SERVICES:  # programmer error — fail loudly
        raise ValueError(f"unknown service {service!r}")
    with _lock:
        if not _cache:
            _cache.update(_read_urls())
        return _cache[service]


def clear_config_cache() -> None:
    """Forget the cached URLs (tests; the file is re-read on the next call)."""
    with _lock:
        _cache.clear()
